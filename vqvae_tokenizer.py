"""Discrete video tokenization (VQ-VAE) on the same PushArena frames.

MiniWorld's three world models (PixelWM, RSSM, JEPA) all predict in
continuous space. A fourth lineage — Genie, VQGAN-based video models —
tokenizes frames into a discrete codebook first and predicts *token*
sequences, the same move language models make. This script builds the
tokenizer half of that pipeline and studies its best-known failure mode:
codebook collapse, where only a handful of codes ever get used and the
rest sit dead.

Two runs on the same data and the same architecture:
  naive — textbook VQ-VAE-1 codebook loss (commitment + codebook MSE),
          straight-through gradient, no protection against dead codes
  fixed — EMA codebook updates (van den Oord et al., VQ-VAE-2 / DeepMind
          recipe) plus dead-code reinitialization from live encoder outputs

If collapse is real here, the naive run should end with most probability
mass on a small fraction of the codebook; the fixed run should not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CKPT.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CODEBOOK_K = 48
EMBED_D = 64


def foreground_mse(pred: torch.Tensor, target: torch.Tensor, bg: float = 0.10, gain: float = 12.0) -> torch.Tensor:
    fg = (target - bg).abs().mean(1, keepdim=True)
    w = 1.0 + gain * (fg > 0.05).float()
    return (w * (pred - target).pow(2)).mean()


class Encoder(nn.Module):
    def __init__(self, out_dim: int = EMBED_D) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, out_dim, 4, 2, 1), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, D, 8, 8]


class Decoder(nn.Module):
    def __init__(self, in_dim: int = EMBED_D) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    """K codes of dim D. `use_ema` switches between the two failure/fix
    regimes described in the module docstring."""

    def __init__(self, k: int = CODEBOOK_K, d: int = EMBED_D, use_ema: bool = False,
                 decay: float = 0.99, dead_after: int = 200, beta: float = 0.25) -> None:
        super().__init__()
        self.k, self.d, self.use_ema, self.decay, self.beta = k, d, use_ema, decay, beta
        self.dead_after = dead_after
        embed = torch.randn(k, d) * 0.5
        self.embed = nn.Parameter(embed, requires_grad=not use_ema)
        self.register_buffer("cluster_size", torch.zeros(k))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("steps_unused", torch.zeros(k))

    def forward(self, z_e: torch.Tensor):
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)  # [B*H*W]
        onehot = F.one_hot(idx, self.k).float()
        quant = onehot @ self.embed  # [B*H*W, D]

        if self.training and self.use_ema:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = (self.cluster_size + 1e-5) / (n + self.k * 1e-5) * n
                self.embed.data.copy_(self.embed_avg / size.unsqueeze(1))
                used = onehot.sum(0) > 0
                self.steps_unused[used] = 0
                self.steps_unused[~used] += 1
                dead = self.steps_unused > self.dead_after
                if dead.any():
                    replace = flat[torch.randint(0, flat.size(0), (int(dead.sum().item()),), device=flat.device)]
                    self.embed.data[dead] = replace
                    self.embed_avg.data[dead] = replace
                    self.cluster_size.data[dead] = 1.0
                    self.steps_unused[dead] = 0

        quant = quant.view(B, H, W, D).permute(0, 3, 1, 2)
        commit_loss = F.mse_loss(z_e, quant.detach())
        if self.use_ema:
            vq_loss = self.beta * commit_loss
        else:
            codebook_loss = F.mse_loss(quant, z_e.detach())
            vq_loss = codebook_loss + self.beta * commit_loss
        z_q = z_e + (quant - z_e).detach()  # straight-through
        return z_q, vq_loss, idx.view(B, H * W)


class VQVAE(nn.Module):
    def __init__(self, use_ema: bool) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.vq = VectorQuantizer(use_ema=use_ema)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx

    def loss(self, x: torch.Tensor):
        recon, vq_loss, idx = self.forward(x)
        recon_loss = foreground_mse(recon, x)
        return recon_loss + vq_loss, {"recon": recon_loss.item(), "vq": float(vq_loss)}, idx


def load_frames() -> np.ndarray:
    path = DATA / "arena.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run train.py once to collect the shared dataset first")
    raw = np.load(path)
    return raw["obs"]  # [N, 64, 64, 3] float32 in [0,1]


def train_vqvae(obs: np.ndarray, use_ema: bool, steps: int = 1500, lr: float = 3e-4) -> VQVAE:
    x = torch.from_numpy(obs.transpose(0, 3, 1, 2))
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True, drop_last=True)
    model = VQVAE(use_ema=use_ema)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    it = iter(loader)
    tag = "ema " if use_ema else "naive"
    for step in range(1, steps + 1):
        try:
            (xb,) = next(it)
        except StopIteration:
            it = iter(loader)
            (xb,) = next(it)
        loss, log, _ = model.loss(xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 300 == 0 or step == steps:
            print(f"[vqvae:{tag}] {step:4d}/{steps}  recon={log['recon']:.4f}  vq={log['vq']:.4f}")
    return model


@torch.no_grad()
def codebook_stats(model: VQVAE, obs: np.ndarray, n: int = 2000) -> dict:
    model.eval()
    x = torch.from_numpy(obs[:n].transpose(0, 3, 1, 2))
    recon, _, idx = model.forward(x)
    recon_mse = foreground_mse(recon, x).item()
    counts = torch.bincount(idx.reshape(-1), minlength=model.vq.k).float()
    probs = counts / counts.sum()
    active = int((counts > 0).sum())
    nonzero = probs[probs > 0]
    perplexity = float(torch.exp(-(nonzero * nonzero.log()).sum()))
    top1_share = float(probs.max())
    return {
        "recon_fg_mse": recon_mse,
        "active_codes": active,
        "codebook_size": model.vq.k,
        "perplexity": perplexity,
        "top1_code_share": top1_share,
    }


def save_recon_strip(models: dict, obs: np.ndarray, path: Path, n: int = 6) -> None:
    from PIL import Image

    x = torch.from_numpy(obs[:n].transpose(0, 3, 1, 2))
    rows = [np.concatenate([np.clip(im, 0, 1) for im in obs[:n]], axis=1)]
    for m in models.values():
        m.eval()
        with torch.no_grad():
            recon, _, _ = m.forward(x)
        recon = recon.numpy().transpose(0, 2, 3, 1)
        rows.append(np.concatenate([np.clip(im, 0, 1) for im in recon], axis=1))
    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray((grid * 255).astype(np.uint8))
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    img.save(path)
    print("wrote", path)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    obs = load_frames()
    print(f"frames={len(obs)}")

    naive = train_vqvae(obs, use_ema=False, steps=1500)
    fixed = train_vqvae(obs, use_ema=True, steps=1500)

    stats_naive = codebook_stats(naive, obs)
    stats_fixed = codebook_stats(fixed, obs)
    print("naive:", stats_naive)
    print("fixed:", stats_fixed)

    torch.save(naive.state_dict(), CKPT / "vqvae_naive.pt")
    torch.save(fixed.state_dict(), CKPT / "vqvae_ema.pt")
    save_recon_strip({"naive": naive, "ema": fixed}, obs, RESULTS / "vqvae_recon_strip.png")

    out = {"naive": stats_naive, "ema_deadcode_fix": stats_fixed}
    (RESULTS / "vqvae.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", RESULTS / "vqvae.json")


if __name__ == "__main__":
    main()
