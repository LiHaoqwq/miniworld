"""CEM planning with a learned dynamics model — done in state space, not pixels.

MiniWorld's pixel/RSSM CEM planner (eval.py) never beat random. Two suspects:
the model plans on top of *learned* image/latent dynamics whose errors are
large to begin with, and it plans the whole action sequence once, open-loop,
so those errors compound for the entire horizon before anything is corrected.

This script isolates the planning question from the perception question by
reusing PushArena's ground-truth low-dimensional state (agent xy, ball xy,
ball velocity, goal xy — 8 numbers) instead of pixels. A small MLP learns
one-step state dynamics; CEM searches over actions scored by the environment's
own (known, analytic) reward function evaluated on imagined rollouts.

Two planning regimes are compared on purpose:
  open-loop  — plan a full-episode action sequence once at t=0, execute blind
  mpc        — replan every step from the true current state, execute only
               the first action (receding horizon)

If compounding model error is really the failure mode, open-loop should
degrade over the episode and MPC should recover most of the gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from env import PushArena

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CKPT.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

ACTIONS = 5
STATE_DIM = 8  # agent_x, agent_y, ball_x, ball_y, ball_vx, ball_vy, goal_x, goal_y
SUCCESS_R = 7.5 + 6.5 + 1.5  # ball_r + goal_r + 1.5, matches env.step


def _state(env: PushArena) -> np.ndarray:
    return np.concatenate([env.agent, env.ball, env.ball_vel, env.goal]).astype(np.float32)


def collect(n_episodes: int, horizon: int, seed: int, epsilon_goal: float = 0.35):
    """Same heuristic-mixed-with-random policy as the pixel dataset, but we
    log full state (including ball velocity, which pixels never expose)."""
    env = PushArena(seed=seed)
    s, ns, a, done = [], [], [], []
    for _ in range(n_episodes):
        env.reset()
        for t in range(horizon):
            cur = _state(env)
            if env.rng.random() < epsilon_goal:
                delta = env.ball - env.agent
                act = (4 if delta[0] > 0 else 3) if abs(delta[0]) > abs(delta[1]) else (2 if delta[1] > 0 else 1)
            else:
                act = int(env.rng.integers(0, ACTIONS))
            env.step(act)
            s.append(cur)
            a.append(act)
            ns.append(_state(env))
            done.append(t == horizon - 1)
    return (
        np.stack(s).astype(np.float32),
        np.asarray(a, dtype=np.int64),
        np.stack(ns).astype(np.float32),
        np.asarray(done, dtype=bool),
    )


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, np.maximum(std, 1e-3)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - torch.as_tensor(self.mean)) / torch.as_tensor(self.std)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.as_tensor(self.std) + torch.as_tensor(self.mean)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class DynamicsModel(nn.Module):
    """MLP(state_norm, action_onehot) -> normalized *delta*. Predicting the
    change in state (not the absolute next state) trains much better here:
    most of a state's variance across the dataset is "where is the ball",
    but what the network actually needs to learn is "how much did it move,"
    which has far smaller and more uniform scale — direct next-state
    regression spends most of its capacity re-deriving the identity map."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTIONS, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def step_norm(self, s_norm: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(action.long(), ACTIONS).float()
        delta = self.net(torch.cat([s_norm, oh], -1))
        return s_norm + delta

    forward = step_norm


def train_dynamics(s, a, ns, steps: int = 4000, lr: float = 1e-3) -> tuple[DynamicsModel, Normalizer]:
    mean, std = s.mean(0), s.std(0)
    norm = Normalizer(mean, std)
    delta = ns - s
    delta_std = np.maximum(delta.std(0), 1e-3)
    s_t = norm.norm(torch.from_numpy(s))
    ns_t = norm.norm(torch.from_numpy(ns))
    a_t = torch.from_numpy(a)
    loader = DataLoader(TensorDataset(s_t, a_t, ns_t), batch_size=128, shuffle=True, drop_last=True)
    model = DynamicsModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    dw = torch.as_tensor(std / delta_std, dtype=torch.float32)  # reweight loss toward the delta scale
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            sb, ab, nsb = next(it)
        except StopIteration:
            it = iter(loader)
            sb, ab, nsb = next(it)
        pred = model.step_norm(sb, ab)
        loss = (dw * (pred - nsb)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % 500 == 0 or step == steps:
            print(f"[dynamics] {step:4d}/{steps}  loss={loss.item():.5f}")
    return model, norm


@torch.no_grad()
def rollout_error(model: DynamicsModel, norm: Normalizer, n: int = 200, horizons=(1, 5, 10)) -> dict:
    """Open-loop state-space rollout error, same protocol as the pixel/RSSM
    table in eval.py, so the numbers are directly comparable in the writeup."""
    env = PushArena(seed=999)
    max_h = max(horizons)
    errs = {h: [] for h in horizons}
    for _ in range(n):
        env.reset()
        s0 = _state(env)
        acts = [int(env.rng.integers(0, ACTIONS)) for _ in range(max_h)]
        gt = []
        for act in acts:
            env.step(act)
            gt.append(_state(env))
        gt = np.stack(gt)
        s_norm = norm.norm(torch.from_numpy(s0)[None])
        preds = []
        for act in acts:
            s_norm = model(s_norm, torch.tensor([act]))
            preds.append(norm.denorm(s_norm)[0].numpy())
        preds = np.stack(preds)
        for h in horizons:
            errs[h].append(float(((preds[h - 1] - gt[h - 1]) ** 2).mean()))
    return {str(h): float(np.mean(v)) for h, v in errs.items()}


def _reward(before: float, after: float, success: bool) -> float:
    return (before - after) * 0.25 + (1.0 if success else 0.0) - 0.01


@torch.no_grad()
def _cem_plan(model, norm, state0: np.ndarray, goal_xy: np.ndarray, horizon: int, pop=48, elites=8, iters=3,
              approach_weight: float = 0.08, rng=None):
    """CEM over imagined rollouts, scored by the environment's own (known)
    reward function — plus a shaping term that rewards the agent for closing
    the gap to the ball.

    First version scored *only* ball-to-goal distance, exactly like env.step.
    It never beat the ball-chasing heuristic baseline: when the agent starts
    far from the ball, every candidate action sequence in the population
    predicts zero ball movement, so all of them score identically and CEM has
    no gradient toward "go get the ball" — it only starts working once a
    sampled sequence gets lucky and stumbles into contact. The shaping term
    below gives it something to climb before contact happens.
    """
    rng = np.random.default_rng() if rng is None else rng
    logits = np.zeros((horizon, ACTIONS), dtype=np.float32)
    for _ in range(iters):
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)
        samples = torch.multinomial(probs.expand(pop, -1, -1).reshape(-1, ACTIONS), 1).view(pop, horizon)
        s = norm.norm(torch.from_numpy(state0)[None]).expand(pop, -1).clone()
        scores = torch.zeros(pop)
        goal_t = torch.as_tensor(goal_xy)
        prev_goal_dist = torch.full((pop,), float(np.linalg.norm(state0[2:4] - goal_xy)))
        prev_ab_dist = torch.full((pop,), float(np.linalg.norm(state0[0:2] - state0[2:4])))
        for t in range(horizon):
            s = model(s, samples[:, t])
            raw = norm.denorm(s)
            agent_xy, ball_xy = raw[:, 0:2], raw[:, 2:4]
            goal_dist = torch.linalg.norm(ball_xy - goal_t, dim=-1)
            ab_dist = torch.linalg.norm(agent_xy - ball_xy, dim=-1)
            success = (goal_dist < SUCCESS_R).float()
            scores = scores + (prev_goal_dist - goal_dist) * 0.25 + success - 0.01
            scores = scores + approach_weight * (prev_ab_dist - ab_dist).clamp(min=0.0)
            prev_goal_dist, prev_ab_dist = goal_dist, ab_dist
        top = torch.topk(scores, k=min(elites, pop)).indices
        elite = samples[top].numpy()
        counts = np.zeros_like(logits)
        for seq in elite:
            for t, act in enumerate(seq):
                counts[t, act] += 1
        logits = np.log(counts + 0.05)
    return elite[0]


def _run_episode(env: PushArena, policy_step, max_steps: int) -> tuple[bool, float]:
    hit = False
    for _ in range(max_steps):
        _, _, _, info = policy_step(env)
        hit = hit or info["success"]
    final = float(np.linalg.norm(env.ball - env.goal))
    return hit, final


def evaluate_policies(model, norm, n_episodes: int = 20, ep_len: int = 40, plan_horizon: int = 10, seed: int = 42):
    names = ("random", "heuristic_blind", "open_loop_cem", "mpc_cem_noshape", "mpc_cem")
    results = {name: {"success": 0, "final_dist": 0.0} for name in names}
    seeds = np.random.default_rng(seed).integers(0, 1_000_000, size=n_episodes)

    for ep_seed in seeds:
        # random
        env = PushArena(seed=int(ep_seed))
        env.reset()
        hit, final = _run_episode(env, lambda e: e.step(int(e.rng.integers(0, ACTIONS))), ep_len)
        results["random"]["success"] += hit
        results["random"]["final_dist"] += final

        # heuristic: always walks toward the ball, never "knows" the goal
        env = PushArena(seed=int(ep_seed))
        env.reset()

        def heuristic_step(e):
            delta = e.ball - e.agent
            act = (4 if delta[0] > 0 else 3) if abs(delta[0]) > abs(delta[1]) else (2 if delta[1] > 0 else 1)
            return e.step(act)

        hit, final = _run_episode(env, heuristic_step, ep_len)
        results["heuristic_blind"]["success"] += hit
        results["heuristic_blind"]["final_dist"] += final

        # open-loop CEM: plan the whole episode once, execute blind
        env = PushArena(seed=int(ep_seed))
        env.reset()
        plan = _cem_plan(model, norm, _state(env), env.goal.copy(), horizon=ep_len)
        hit, final = False, 0.0
        for act in plan:
            _, _, _, info = env.step(int(act))
            hit = hit or info["success"]
        final = float(np.linalg.norm(env.ball - env.goal))
        results["open_loop_cem"]["success"] += hit
        results["open_loop_cem"]["final_dist"] += final

        # MPC CEM, no approach shaping: replan every step, score = goal distance only
        env = PushArena(seed=int(ep_seed))
        env.reset()

        def mpc_step_noshape(e):
            plan = _cem_plan(model, norm, _state(e), e.goal.copy(), horizon=plan_horizon, pop=64, iters=4, approach_weight=0.0)
            return e.step(int(plan[0]))

        hit, final = _run_episode(env, mpc_step_noshape, ep_len)
        results["mpc_cem_noshape"]["success"] += hit
        results["mpc_cem_noshape"]["final_dist"] += final

        # MPC CEM, with approach shaping: replan every step from the true current state
        env = PushArena(seed=int(ep_seed))
        env.reset()

        def mpc_step(e):
            plan = _cem_plan(model, norm, _state(e), e.goal.copy(), horizon=plan_horizon, pop=64, iters=4)
            return e.step(int(plan[0]))

        hit, final = _run_episode(env, mpc_step, ep_len)
        results["mpc_cem"]["success"] += hit
        results["mpc_cem"]["final_dist"] += final

    for name, d in results.items():
        d["success"] = d["success"] / n_episodes
        d["final_dist"] = d["final_dist"] / n_episodes
    return results


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    print("collecting state dataset …")
    s, a, ns, done = collect(n_episodes=150, horizon=64, seed=0)
    print(f"transitions={len(a)}")
    model, norm = train_dynamics(s, a, ns, steps=4000)
    torch.save(model.state_dict(), CKPT / "state_dynamics.pt")

    print("rollout error …")
    err = rollout_error(model, norm)
    print(err)

    print("planning eval (20 episodes x 4 policies, this takes a minute) …")
    plan = evaluate_policies(model, norm, n_episodes=20)
    print(json.dumps(plan, indent=2))

    out = {
        "rollout_state_mse": err,
        "planning": plan,
        "normalizer": norm.to_dict(),
        "config": {"ep_len": 40, "plan_horizon": 10, "n_episodes": 20},
    }
    (RESULTS / "state_planning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", RESULTS / "state_planning.json")


if __name__ == "__main__":
    main()
