"""Does the data-collection policy decide whether a world model learns to act?

Every world-model paper spends most of its engineering budget on data, and the
argument is always about the *collection policy*, not the model. This script
makes that argument measurable at toy scale.

The collector mixes two behaviours with probability epsilon:
  epsilon = 0.0  pure uniform-random actions
  epsilon = 1.0  pure "walk toward the ball" heuristic
and anything in between.

Both extremes should fail, for opposite reasons:

  Random    the agent almost never reaches the ball, so the dataset contains
            almost no contact events — the model never sees the one piece of
            dynamics that actions can actually influence.

  Heuristic contacts are frequent, but the action is now a deterministic
            function of the state (the agent always walks toward the ball).
            Action and state are confounded: the model can predict the future
            from the state alone and safely ignore a_t. This is the toy version
            of "expert demonstrations have no counterfactual coverage."

The evaluation is the part that matters. Every model is scored on ONE shared
held-out set, and the headline metric is measured only on transitions where a
collision actually happened — the transitions where the action was allowed to
matter:

  ball MSE (true action)   how well it predicts the ball after a hit
  ball MSE (wrong action)  same transitions, deliberately wrong action fed in
  action gap = wrong - true

A model that genuinely conditions on a_t is hurt by a wrong action, so the gap
is large. A model that has quietly learned "actions are irrelevant, just
extrapolate the ball's momentum" scores a gap near zero — no matter how good
its plain MSE looks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from env import ACTION_VEC, PushArena
from state_planning import ACTIONS, DynamicsModel, Normalizer, STATE_DIM, _state, train_dynamics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

BALL = slice(2, 4)  # ball xy inside the 8-dim state vector
EPSILONS = (0.0, 0.15, 0.35, 0.6, 1.0)


def heuristic_action(env: PushArena) -> int:
    delta = env.ball - env.agent
    if abs(delta[0]) > abs(delta[1]):
        return 4 if delta[0] > 0 else 3
    return 2 if delta[1] > 0 else 1


def would_contact(env: PushArena, action: int) -> bool:
    """Replicates env.step's collision test: the agent moves first, then the
    overlap check runs against the *new* agent position."""
    new_agent = env._clip(env.agent + ACTION_VEC[action] * env.action_scale, env.agent_r)
    return float(np.linalg.norm(env.ball - new_agent)) < (env.agent_r + env.ball_r)


def collect(n_episodes: int, horizon: int, seed: int, epsilon: float) -> dict:
    """Collect transitions and log, per step, whether the action caused contact."""
    env = PushArena(seed=seed)
    s, a, ns, contact = [], [], [], []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(horizon):
            cur = _state(env)
            act = heuristic_action(env) if env.rng.random() < epsilon else int(env.rng.integers(0, ACTIONS))
            hit = would_contact(env, act)
            env.step(act)
            s.append(cur)
            a.append(act)
            ns.append(_state(env))
            contact.append(hit)
    return {
        "s": np.stack(s).astype(np.float32),
        "a": np.asarray(a, dtype=np.int64),
        "ns": np.stack(ns).astype(np.float32),
        "contact": np.asarray(contact, dtype=bool),
    }


def heuristic_action_from_state(s: np.ndarray) -> np.ndarray:
    """Vectorised version of heuristic_action, computed from logged states."""
    delta = s[:, 2:4] - s[:, 0:2]  # ball - agent
    horizontal = np.abs(delta[:, 0]) > np.abs(delta[:, 1])
    return np.where(
        horizontal,
        np.where(delta[:, 0] > 0, 4, 3),
        np.where(delta[:, 1] > 0, 2, 1),
    )


def dataset_stats(data: dict) -> dict:
    """What the collection policy actually put in the dataset, before any model
    is trained. These are the numbers a data-collection argument lives on."""
    counts = np.bincount(data["a"], minlength=ACTIONS).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    ball_delta = np.linalg.norm(data["ns"][:, BALL] - data["s"][:, BALL], axis=1)
    # How well a_t can be guessed from s_t alone. At epsilon=1 the action is a
    # deterministic function of the state, so state and action are fully
    # confounded and the model can drop a_t without paying for it.
    predictability = float((data["a"] == heuristic_action_from_state(data["s"])).mean())
    return {
        "contact_rate": float(data["contact"].mean()),
        "ball_moved_rate": float((ball_delta > 0.5).mean()),
        "action_entropy": entropy,
        "action_entropy_max": float(np.log(ACTIONS)),
        "action_predictable_from_state": predictability,
        "n_transitions": int(len(data["a"])),
    }


def set_env_state(env: PushArena, s: np.ndarray) -> None:
    """The 8-dim state vector IS the complete env state, so counterfactuals
    can be simulated exactly: rewind to s_t, take a different action, look."""
    env.agent = s[0:2].astype(np.float32).copy()
    env.ball = s[2:4].astype(np.float32).copy()
    env.ball_vel = s[4:6].astype(np.float32).copy()
    env.goal = s[6:8].astype(np.float32).copy()


def oracle_action_effect(eval_data: dict, seed: int = 0) -> dict:
    """The ceiling any model could possibly score on the action gap.

    A perfect model predicts the true next state under a_t (error 0) and the
    true counterfactual next state under a wrong action. Its gap is therefore
    exactly how much the ball's true future actually changes when the action
    changes — the causal effect size of the action. Without this number the
    model gaps below are unreadable: a small gap could mean the model ignores
    actions, or that the action genuinely does not matter much.
    """
    rng = np.random.default_rng(seed)
    env = PushArena(seed=0)
    wrong = (eval_data["a"] + rng.integers(1, ACTIONS, size=len(eval_data["a"]))) % ACTIONS
    cf_ball = np.empty((len(wrong), 2), dtype=np.float32)
    for i, (s, a_w) in enumerate(zip(eval_data["s"], wrong)):
        set_env_state(env, s)
        env.step(int(a_w))
        cf_ball[i] = env.ball
    sq = ((cf_ball - eval_data["ns"][:, BALL]) ** 2).mean(axis=1)
    contact = eval_data["contact"]
    return {
        "oracle_action_gap": float(sq.mean()),
        "oracle_action_gap_contact": float(sq[contact].mean()),
    }


def rebalance(data: dict, target_contact_fraction: float = 0.5) -> dict:
    """Oversample contact transitions.

    Contacts are rare but carry all of the action-conditioned information. Under
    a plain uniform loss they are averaged away by the overwhelming majority of
    transitions where nothing touches anything. This is the toy version of
    "rare-but-informative events need to be resampled, not just collected."
    """
    contact = data["contact"]
    p = float(contact.mean())
    if p <= 0 or p >= 1 or target_contact_fraction <= p:
        return data
    q = target_contact_fraction
    k = max(int(round(q * (1 - p) / (p * (1 - q)))), 1)
    idx = np.concatenate([np.arange(len(contact))] + [np.where(contact)[0]] * (k - 1))
    return {key: data[key][idx] for key in ("s", "a", "ns", "contact")}


@torch.no_grad()
def evaluate(model: DynamicsModel, norm: Normalizer, eval_data: dict, seed: int = 0) -> dict:
    """Score one model on the shared held-out set.

    Reported twice: over every transition, and over the contact subset only.
    The contact subset is where a_t is allowed to change the ball's future, so
    it is the only place an action-sensitivity gap can legitimately appear.
    """
    rng = np.random.default_rng(seed)
    s = torch.from_numpy(eval_data["s"])
    ns = torch.from_numpy(eval_data["ns"])
    a = torch.from_numpy(eval_data["a"])
    # A deliberately wrong action: uniformly resampled among the other four.
    wrong = torch.from_numpy((eval_data["a"] + rng.integers(1, ACTIONS, size=len(eval_data["a"]))) % ACTIONS)

    s_norm = norm.norm(s)
    pred_true = norm.denorm(model(s_norm, a))
    pred_wrong = norm.denorm(model(s_norm, wrong))

    def ball_mse(pred: torch.Tensor, mask: np.ndarray | None = None) -> float:
        err = (pred[:, BALL] - ns[:, BALL]).pow(2).mean(dim=1)
        if mask is not None:
            err = err[torch.from_numpy(mask)]
        return float(err.mean())

    contact = eval_data["contact"]
    true_all, wrong_all = ball_mse(pred_true), ball_mse(pred_wrong)
    true_hit, wrong_hit = ball_mse(pred_true, contact), ball_mse(pred_wrong, contact)
    return {
        "ball_mse_true": true_all,
        "ball_mse_wrong": wrong_all,
        "action_gap": wrong_all - true_all,
        "ball_mse_true_contact": true_hit,
        "ball_mse_wrong_contact": wrong_hit,
        "action_gap_contact": wrong_hit - true_hit,
        "action_gap_contact_ratio": wrong_hit / max(true_hit, 1e-9),
        "eval_contact_n": int(contact.sum()),
    }


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # One shared evaluation set for every configuration. It uses a mixed policy
    # (epsilon=0.4) and a held-out seed, so no training config gets to be
    # evaluated on exactly its own distribution.
    print("collecting shared eval set …")
    eval_data = collect(n_episodes=40, horizon=64, seed=4242, epsilon=0.4)
    print(f"  eval transitions={len(eval_data['a'])}  contacts={int(eval_data['contact'].sum())}")

    oracle = oracle_action_effect(eval_data)
    print(f"  oracle action gap (contact) = {oracle['oracle_action_gap_contact']:.4f}  "
          f"← the ceiling a perfect model would score")

    def run_one(tag: str, data: dict) -> dict:
        stats = dataset_stats(data)
        print(f"  data: n={stats['n_transitions']}  contact_rate={stats['contact_rate']:.3f}  "
              f"a|s predictable={stats['action_predictable_from_state']:.3f}")
        model, norm = train_dynamics(data["s"], data["a"], data["ns"], steps=4000)
        model.eval()
        scores = evaluate(model, norm, eval_data)
        scores["recovered_fraction_contact"] = (
            scores["action_gap_contact"] / max(oracle["oracle_action_gap_contact"], 1e-9)
        )
        print(f"  eval: ball_mse(contact)={scores['ball_mse_true_contact']:.4f}  "
              f"action_gap(contact)={scores['action_gap_contact']:.4f}  "
              f"= {100*scores['recovered_fraction_contact']:.1f}% of oracle")
        return {"data": stats, "eval": scores}

    runs = {}
    for eps in EPSILONS:
        print(f"\n=== epsilon={eps} ===")
        runs[str(eps)] = run_one(str(eps), collect(120, 64, seed=0, epsilon=eps))

    # Fix condition: same collection policy as the best mixed setting, but the
    # rare contact transitions are oversampled to half the training set.
    print("\n=== epsilon=0.35 + contact rebalancing (target 50%) ===")
    base = collect(120, 64, seed=0, epsilon=0.35)
    runs["0.35_rebalanced"] = run_one("0.35_rebalanced", rebalance(base, 0.5))

    out = {
        "protocol": {
            "train_episodes": 120,
            "horizon": 64,
            "eval_episodes": 40,
            "eval_epsilon": 0.4,
            "eval_seed": 4242,
            "train_steps": 4000,
            "note": "epsilon = P(goal-directed heuristic action); 1-epsilon = P(uniform random)",
            "action_gap": "ball MSE(wrong action) - ball MSE(true action), on contact transitions",
        },
        "oracle": oracle,
        "runs": runs,
    }
    (RESULTS / "data_collection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", RESULTS / "data_collection.json")


if __name__ == "__main__":
    main()
