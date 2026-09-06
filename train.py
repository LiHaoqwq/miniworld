"""Train PixelWM / RSSM / JEPA on the same PushArena dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from env import collect_episodes
from models import JEPA, PixelWM, RSSM


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
DATA.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)


class PairDataset(Dataset):
    def __init__(self, bundle: dict) -> None:
        obs, act, rew, done = bundle["obs"], bundle["act"], bundle["rew"], bundle["done"]
        # Pairs (t, t+1) that do not cross episode boundaries.
        idx = np.where(~done)[0]
        self.obs = obs[idx]
        self.next_obs = obs[idx + 1]
        self.act = act[idx]
        self.rew = rew[idx]

    def __len__(self) -> int:
        return len(self.act)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.obs[i].transpose(2, 0, 1)),
            int(self.act[i]),
            torch.from_numpy(self.next_obs[i].transpose(2, 0, 1)),
            float(self.rew[i]),
        )


class SeqDataset(Dataset):
    def __init__(self, bundle: dict, seq_len: int = 8) -> None:
        obs, act, rew, done = bundle["obs"], bundle["act"], bundle["rew"], bundle["done"]
        self.seq_len = seq_len
        starts = []
        i, n = 0, len(done)
        while i + seq_len <= n:
            if not done[i : i + seq_len - 1].any():
                starts.append(i)
            if done[i]:
                i += 1
            else:
                i += 1
        self.starts = np.asarray(starts)
        self.obs, self.act, self.rew = obs, act, rew

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        e = s + self.seq_len
        obs = torch.from_numpy(self.obs[s:e].transpose(0, 3, 1, 2).copy())
        # a_t is the action that produced obs[t] (previous action). First step: noop.
        act = np.zeros(self.seq_len, dtype=np.int64)
        act[1:] = self.act[s : e - 1]
        rew = torch.from_numpy(self.rew[s:e].copy())
        return obs, torch.from_numpy(act), rew


def to_chw(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def load_or_collect(seed: int = 0) -> dict:
    path = DATA / "arena.npz"
    if path.exists():
        raw = np.load(path)
        return {k: raw[k] for k in raw.files}
    print("collecting dataset …")
    bundle = collect_episodes(n_episodes=120, horizon=64, seed=seed)
    np.savez_compressed(path, **bundle)
    print(f"saved {path}  obs={bundle['obs'].shape}")
    return bundle


def train_pixel(bundle, steps: int, device: torch.device, lr: float = 3e-4) -> PixelWM:
    model = PixelWM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(PairDataset(bundle), batch_size=32, shuffle=True, drop_last=True)
    it = iter(loader)
    model.train()
    log = {}
    for step in range(1, steps + 1):
        try:
            obs, act, nxt, rew = next(it)
        except StopIteration:
            it = iter(loader)
            obs, act, nxt, rew = next(it)
        obs, nxt = obs.to(device), nxt.to(device)
        act, rew = act.to(device), rew.to(device).float()
        loss, log = model.loss(obs, act, nxt, rew)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"[pixel] {step:4d}/{steps}  recon={log['recon']:.4f}  reward={log['reward']:.4f}")
    return model


def train_rssm(bundle, steps: int, device: torch.device, lr: float = 3e-4) -> RSSM:
    model = RSSM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(SeqDataset(bundle, seq_len=8), batch_size=16, shuffle=True, drop_last=True)
    it = iter(loader)
    model.train()
    log = {}
    for step in range(1, steps + 1):
        try:
            obs, act, rew = next(it)
        except StopIteration:
            it = iter(loader)
            obs, act, rew = next(it)
        obs, act, rew = obs.to(device), act.to(device), rew.to(device)
        loss, log = model.loss(obs, act, rew)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(
                f"[rssm ] {step:4d}/{steps}  recon={log['recon']:.4f}  "
                f"reward={log['reward']:.4f}  kl={log['kl']:.4f}"
            )
    return model


def train_jepa(bundle, steps: int, device: torch.device, lr: float = 3e-4) -> JEPA:
    model = JEPA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(PairDataset(bundle), batch_size=32, shuffle=True, drop_last=True)
    it = iter(loader)
    model.train()
    log = {}
    for step in range(1, steps + 1):
        try:
            obs, act, nxt, _ = next(it)
        except StopIteration:
            it = iter(loader)
            obs, act, nxt, _ = next(it)
        obs, nxt, act = obs.to(device), nxt.to(device), act.to(device)
        loss, log = model.loss(obs, act, nxt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"[jepa ] {step:4d}/{steps}  cos={log['cos']:.4f}  var={log['var']:.4f}")
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pixel", "rssm", "jepa", "all"], default="all")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    bundle = load_or_collect(args.seed)

    trained = {}
    if args.model in ("pixel", "all"):
        trained["pixel"] = train_pixel(bundle, args.steps, device)
        torch.save(trained["pixel"].state_dict(), CKPT / "pixel.pt")
    if args.model in ("rssm", "all"):
        trained["rssm"] = train_rssm(bundle, args.steps, device)
        torch.save(trained["rssm"].state_dict(), CKPT / "rssm.pt")
    if args.model in ("jepa", "all"):
        trained["jepa"] = train_jepa(bundle, args.steps, device)
        torch.save(trained["jepa"].state_dict(), CKPT / "jepa.pt")
    meta = {"steps": args.steps, "seed": args.seed, "models": list(trained)}
    (CKPT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done", meta)


if __name__ == "__main__":
    main()
