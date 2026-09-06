"""Compare the three world models on rollout error and CEM planning.

Metrics
-------
pixel-MSE @ 1/5/10 : open-loop next-frame error (JEPA has no pixels; skipped)
embed-cos @ 1/5/10 : cosine between predicted and target-encoder embeddings
plan-success       : CEM over imagined action sequences, executed in the real env
action-ablation    : pixel model retrained? we instead zero the action at eval
                     for PixelWM/RSSM by feeding noop — drop in accuracy shows
                     the predictor actually uses a_t.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env import PushArena
from models import JEPA, PixelWM, RSSM
from train import CKPT, DATA, load_or_collect, to_chw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_models(device: torch.device):
    pixel, rssm, jepa = PixelWM(), RSSM(), JEPA()
    pixel.load_state_dict(torch.load(CKPT / "pixel.pt", map_location=device, weights_only=True))
    rssm.load_state_dict(torch.load(CKPT / "rssm.pt", map_location=device, weights_only=True))
    jepa.load_state_dict(torch.load(CKPT / "jepa.pt", map_location=device, weights_only=True))
    for m in (pixel, rssm, jepa):
        m.to(device).eval()
    return pixel, rssm, jepa


def _fg_mask(gt: torch.Tensor, bg: float = 0.10) -> torch.Tensor:
    return ((gt - bg).abs().mean(dim=2, keepdim=True) > 0.05).float()


def _fg_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [B, T, C, H, W] -> [T]"""
    w = _fg_mask(gt)
    num = (w * (pred - gt).pow(2).mean(dim=2, keepdim=True)).sum(dim=(0, 2, 3, 4))
    den = w.sum(dim=(0, 2, 3, 4)).clamp(min=1.0)
    return (num / den)


@torch.no_grad()
def rollout_table(pixel: PixelWM, rssm: RSSM, jepa: JEPA, n: int = 32, horizon: int = 8) -> dict:
    env = PushArena(seed=123)
    mse = {"pixel": np.zeros(horizon), "rssm": np.zeros(horizon)}
    fg = {"pixel": np.zeros(horizon), "rssm": np.zeros(horizon), "pixel_noop": np.zeros(horizon)}
    cos = {"pixel": np.zeros(horizon), "rssm": np.zeros(horizon), "jepa": np.zeros(horizon)}

    for _ in range(n):
        frames, actions = [], []
        obs = env.reset()
        for t in range(horizon + 1):
            frames.append(obs)
            a = int(env.rng.integers(0, 5))
            actions.append(a)
            obs, _, _, _ = env.step(a)
        ctx = to_chw(frames[0])[None]
        act = torch.tensor(actions[:horizon], dtype=torch.long)[None]
        gt = torch.stack([to_chw(f) for f in frames[1 : horizon + 1]])[None]
        noop = torch.zeros_like(act)

        pred_p = pixel.rollout(ctx, act)
        pred_r = rssm.rollout(ctx, act)
        pred_n = pixel.rollout(ctx, noop)
        mse["pixel"] += ((pred_p - gt) ** 2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
        mse["rssm"] += ((pred_r - gt) ** 2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
        fg["pixel"] += _fg_mse(pred_p, gt).cpu().numpy()
        fg["rssm"] += _fg_mse(pred_r, gt).cpu().numpy()
        fg["pixel_noop"] += _fg_mse(pred_n, gt).cpu().numpy()

        tgt = jepa.encode(gt.reshape(-1, *gt.shape[2:]), target=True).reshape(1, horizon, -1)
        for name, pred in ("pixel", pred_p), ("rssm", pred_r):
            emb = jepa.encode(pred.reshape(-1, *pred.shape[2:]), target=True).reshape(1, horizon, -1)
            cos[name] += (emb * tgt).sum(-1).mean(0).cpu().numpy()
        pred_j = jepa.rollout_embed(ctx, act)
        cos["jepa"] += (pred_j * tgt).sum(-1).mean(0).cpu().numpy()

    out = {}
    for name, d in ("mse", mse), ("fg_mse", fg), ("cos", cos):
        out[name] = {k: (v / n).tolist() for k, v in d.items()}
    return out


@torch.no_grad()
def position_probe(jepa: JEPA, bundle: dict) -> dict:
    """Linear readout of (agent, ball, goal) xy from frozen embeddings.

    If JEPA collapsed, R² is near 0. If it kept object state, ball xy should
    be linearly decodable — the JEPA argument in miniature.
    """
    obs = bundle["obs"][:2048]
    y = np.concatenate(
        [bundle["agent"][:2048], bundle["ball"][:2048], bundle["goal"][:2048]], axis=1
    )
    x = torch.from_numpy(obs.transpose(0, 3, 1, 2)).float()
    zs = []
    for i in range(0, len(x), 128):
        zs.append(jepa.encode(x[i : i + 128], target=True).cpu().numpy())
    z = np.concatenate(zs, 0)
    n = len(z)
    n_train = int(n * 0.8)
    z_tr, z_te = z[:n_train], z[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    # Ridge via normal equations
    lam = 1e-2
    d = z_tr.shape[1]
    w = np.linalg.solve(z_tr.T @ z_tr + lam * np.eye(d), z_tr.T @ y_tr)
    pred = z_te @ w
    ss_res = ((y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0).clip(min=1e-6)
    r2 = 1.0 - ss_res / ss_tot
    names = ["agent_x", "agent_y", "ball_x", "ball_y", "goal_x", "goal_y"]
    return {k: float(v) for k, v in zip(names, r2)}


@torch.no_grad()
def _cem_actions(score_fn, horizon: int, pop: int = 32, elites: int = 8, iters: int = 3, rng=None) -> np.ndarray:
    """Discrete CEM: keep a categorical distribution over 5 actions per timestep."""
    rng = np.random.default_rng() if rng is None else rng
    logits = np.zeros((horizon, 5), dtype=np.float32)
    for _ in range(iters):
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)
        samples = torch.multinomial(probs.expand(pop, -1, -1).reshape(-1, 5), 1).view(pop, horizon)
        scores = score_fn(samples)
        top = torch.topk(scores, k=min(elites, pop)).indices
        elite = samples[top].cpu().numpy()
        counts = np.zeros_like(logits)
        for seq in elite:
            for t, a in enumerate(seq):
                counts[t, a] += 1
        logits = np.log(counts + 0.05)
    return elite[0]


@torch.no_grad()
def plan_success(pixel: PixelWM, rssm: RSSM, jepa: JEPA, n: int = 24, horizon: int = 8) -> dict:
    """Plan in each model, execute the first action sequence in the real arena.

    Goal for pixel/RSSM: imagined last frame should have the red ball on green.
    We approximate that by encoding a synthetic 'success' frame? Too fake.
    Instead: minimize predicted *reward-to-go* for RSSM/pixel (they have reward
    heads) and for JEPA minimize embedding distance to a goal image rendered
    with the ball on the goal (oracle rendering of the *current* goal — this
    is the V-JEPA-2-AC 'goal image' protocol at toy scale).
    """
    env = PushArena(seed=7)
    device = next(pixel.parameters()).device
    hits = {"pixel": 0, "rssm": 0, "jepa": 0, "random": 0}
    improve = {"pixel": 0.0, "rssm": 0.0, "jepa": 0.0, "random": 0.0}

    def exec_seq(env: PushArena, seq: np.ndarray) -> tuple[bool, float, float]:
        info = {"success": False, "ball_goal": 0.0}
        start = float(np.linalg.norm(env.ball - env.goal))
        for a in seq:
            _, _, _, info = env.step(int(a))
        delta = start - float(info["ball_goal"])
        ok = bool(info["success"] or delta > 8.0)
        return ok, delta, float(info["ball_goal"])

    for i in range(n):
        start_obs = env.reset()
        snapshot = (env.agent.copy(), env.ball.copy(), env.ball_vel.copy(), env.goal.copy())
        ctx = to_chw(start_obs)[None].to(device)

        # Pixel: maximize predicted reward along the rollout (reward head).
        def score_pixel(samples: torch.Tensor) -> torch.Tensor:
            samples = samples.to(device)
            B = samples.size(0)
            x = ctx.expand(B, -1, -1, -1)
            total = torch.zeros(B, device=device)
            for t in range(samples.size(1)):
                x, r = pixel.forward(x, samples[:, t])
                total = total + r
            return total

        def score_rssm(samples: torch.Tensor) -> torch.Tensor:
            samples = samples.to(device)
            B = samples.size(0)
            dummy = torch.zeros(B, dtype=torch.long, device=device)
            ctx_b = ctx.expand(B, -1, -1, -1)[:, None]
            state, _, _, _, _ = rssm.observe(ctx_b, dummy[:, None], sample=False)
            total = torch.zeros(B, device=device)
            for t in range(samples.size(1)):
                state = rssm.imagine_step(state, F.one_hot(samples[:, t], 5).float(), sample=False)
                feat = rssm._feature(state["h"], state["z"])
                total = total + rssm.reward(feat).squeeze(-1)
            return total

        # Goal image: same agent/floor, ball snapped onto the goal.
        goal_env_obs = start_obs.copy()
        # Re-render with ball at goal using a throwaway env copy.
        tmp = PushArena(seed=0)
        tmp.agent, tmp.ball, tmp.goal = snapshot[0], snapshot[3], snapshot[3]
        tmp.ball_vel[:] = 0
        goal_img = to_chw(tmp.render())[None].to(device)
        goal_z = jepa.encode(goal_img, target=True)

        def score_jepa(samples: torch.Tensor) -> torch.Tensor:
            samples = samples.to(device)
            z = jepa.rollout_embed(ctx.expand(samples.size(0), -1, -1, -1), samples)[:, -1]
            return (z * goal_z).sum(-1)

        plans = {
            "pixel": _cem_actions(score_pixel, horizon),
            "rssm": _cem_actions(score_rssm, horizon),
            "jepa": _cem_actions(score_jepa, horizon),
            "random": np.random.randint(0, 5, size=horizon),
        }
        for name, seq in plans.items():
            env.agent, env.ball, env.ball_vel, env.goal = (
                snapshot[0].copy(), snapshot[1].copy(), snapshot[2].copy(), snapshot[3].copy(),
            )
            ok, delta, _ = exec_seq(env, seq)
            if ok:
                hits[name] += 1
            improve[name] += delta
        if (i + 1) % 6 == 0:
            print(f"planning {i+1}/{n}  { {k: v/(i+1) for k,v in hits.items()} }")

    return {
        "success": {k: v / n for k, v in hits.items()},
        "mean_distance_gain": {k: v / n for k, v in improve.items()},
    }


def save_rollout_strip(pixel: PixelWM, rssm: RSSM, path: Path, horizon: int = 6) -> None:
    """Save a GT / Pixel / RSSM strip for the README."""
    from PIL import Image

    env = PushArena(seed=3)
    frames = [env.reset()]
    actions = []
    for _ in range(horizon):
        a = 4  # always push right — dynamics are visible
        actions.append(a)
        obs, _, _, _ = env.step(a)
        frames.append(obs)
    ctx = to_chw(frames[0])[None]
    act = torch.tensor(actions, dtype=torch.long)[None]
    with torch.no_grad():
        pred_p = pixel.rollout(ctx, act)[0].cpu().numpy().transpose(0, 2, 3, 1)
        pred_r = rssm.rollout(ctx, act)[0].cpu().numpy().transpose(0, 2, 3, 1)
    rows = []
    for seq in (frames[1:], list(pred_p), list(pred_r)):
        row = np.concatenate([np.clip(im, 0, 1) for im in seq], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray((grid * 255).astype(np.uint8))
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    img.save(path)
    print("wrote", path)


def main() -> None:
    device = torch.device("cpu")
    load_or_collect(0)
    pixel, rssm, jepa = load_models(device)
    print("rollout eval …")
    table = rollout_table(pixel, rssm, jepa, n=24, horizon=8)
    print(json.dumps(table, indent=2))
    print("position probe …")
    bundle = load_or_collect(0)
    probe = position_probe(jepa, bundle)
    print("probe", probe)
    print("planning eval …")
    plan = plan_success(pixel, rssm, jepa, n=16, horizon=8)
    print("plan", plan)
    save_rollout_strip(pixel, rssm, OUT / "rollout_strip.png")
    result = {"rollout": table, "probe_r2": probe, "plan": plan}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
