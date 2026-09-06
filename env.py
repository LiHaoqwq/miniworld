"""Controllable 2D push-arena used as a toy physical world.

The agent is kinematic (action sets its velocity). The ball is dynamic:
it integrates velocity, damps, bounces on walls, and receives an impulse
when the agent collides with it. Reward is dense distance-to-goal of the
ball. Observations are 64x64 RGB rasterizations — enough visual nuisance
(antialiased discs, overlapping colors) that a world model must decide
*what* to predict.

This environment exists so three modeling choices can be compared on the
same data: pixel next-frame prediction, RSSM latent dynamics, and JEPA
representation prediction.
"""

from __future__ import annotations

import numpy as np


ACTIONS = ("noop", "up", "down", "left", "right")
ACTION_VEC = np.array(
    [[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
    dtype=np.float32,
)


def _disc(canvas: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray) -> None:
    h, w, _ = canvas.shape
    y0 = max(int(cy - radius - 1), 0)
    y1 = min(int(cy + radius + 2), h)
    x0 = max(int(cx - radius - 1), 0)
    x1 = min(int(cx + radius + 2), w)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
    alpha = np.clip(radius + 0.6 - dist, 0.0, 1.0)[..., None]
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = patch * (1.0 - alpha) + color * alpha


class PushArena:
    """64x64 RGB arena: blue agent pushes a red ball toward a green goal."""

    def __init__(self, size: int = 64, seed: int | None = None) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.agent_r = 6.5
        self.ball_r = 7.5
        self.goal_r = 6.5
        self.margin = 8.0
        self.action_scale = 2.4
        self.damping = 0.90
        self.impulse = 1.8
        self.max_ball_speed = 3.2
        self.reset()

    def reset(self) -> np.ndarray:
        lo, hi = self.margin, self.size - self.margin
        self.agent = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.ball = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        # Keep objects from spawning on top of each other.
        for _ in range(8):
            if np.linalg.norm(self.ball - self.agent) < 14:
                self.ball = self.rng.uniform(lo, hi, size=2).astype(np.float32)
            if np.linalg.norm(self.goal - self.ball) < 16:
                self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.ball_vel = self.rng.normal(0.0, 0.15, size=2).astype(np.float32)
        return self.render()

    def _clip(self, pos: np.ndarray, radius: float) -> np.ndarray:
        lo, hi = radius + 1.0, self.size - radius - 1.0
        return np.clip(pos, lo, hi)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        force = ACTION_VEC[int(action)] * self.action_scale
        self.agent = self._clip(self.agent + force, self.agent_r)

        delta = self.ball - self.agent
        dist = float(np.linalg.norm(delta) + 1e-6)
        min_dist = self.agent_r + self.ball_r
        if dist < min_dist:
            normal = delta / dist
            self.ball_vel = self.ball_vel + normal * self.impulse
            self.ball = self.agent + normal * min_dist

        self.ball_vel = self.ball_vel * self.damping
        speed = float(np.linalg.norm(self.ball_vel))
        if speed > self.max_ball_speed:
            self.ball_vel *= self.max_ball_speed / speed

        prev = self.ball.copy()
        self.ball = self.ball + self.ball_vel
        for i in range(2):
            lo, hi = self.ball_r + 1.0, self.size - self.ball_r - 1.0
            if self.ball[i] < lo or self.ball[i] > hi:
                self.ball[i] = np.clip(self.ball[i], lo, hi)
                self.ball_vel[i] *= -0.65

        # Dense shaping: improvement in ball-to-goal distance, plus a bonus
        # when the ball is parked on the goal. This gives the reward head
        # something non-trivial to predict without requiring a sparse success.
        before = float(np.linalg.norm(prev - self.goal))
        after = float(np.linalg.norm(self.ball - self.goal))
        success = after < (self.ball_r + self.goal_r + 1.5)
        reward = (before - after) * 0.25 + (1.0 if success else 0.0) - 0.01
        obs = self.render()
        info = {
            "success": success,
            "ball_goal": after,
            "agent": self.agent.copy(),
            "ball": self.ball.copy(),
            "goal": self.goal.copy(),
        }
        return obs, float(reward), False, info

    def render(self) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 0.10, dtype=np.float32)
        # Soft floor grid. Strong enough to be clutter, weak enough that a
        # foreground-weighted recon loss can still see the discs.
        img[::8, :, :] = np.minimum(img[::8, :, :] + 0.025, 1.0)
        img[:, ::8, :] = np.minimum(img[:, ::8, :] + 0.025, 1.0)
        _disc(img, *self.goal, self.goal_r, np.array([0.20, 0.78, 0.40], dtype=np.float32))
        _disc(img, *self.ball, self.ball_r, np.array([0.92, 0.28, 0.22], dtype=np.float32))
        _disc(img, *self.agent, self.agent_r, np.array([0.22, 0.50, 0.95], dtype=np.float32))
        return img


def collect_episodes(
    n_episodes: int = 80,
    horizon: int = 64,
    seed: int = 0,
    epsilon_goal: float = 0.35,
) -> dict[str, np.ndarray]:
    """Collect on-policy-ish random data with a weak heuristic.

    With probability `epsilon_goal` the collector moves the agent toward the
    ball (so interactions actually occur). Otherwise it acts uniformly. A
    world model that ignores actions cannot distinguish these two regimes.
    """
    env = PushArena(seed=seed)
    obs_l, act_l, rew_l, done_l = [], [], [], []
    agent_l, ball_l, goal_l = [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        for t in range(horizon):
            if env.rng.random() < epsilon_goal:
                delta = env.ball - env.agent
                if abs(delta[0]) > abs(delta[1]):
                    action = 4 if delta[0] > 0 else 3
                else:
                    action = 2 if delta[1] > 0 else 1
            else:
                action = int(env.rng.integers(0, 5))
            agent_l.append(env.agent.copy())
            ball_l.append(env.ball.copy())
            goal_l.append(env.goal.copy())
            next_obs, reward, _, _ = env.step(action)
            obs_l.append(obs)
            act_l.append(action)
            rew_l.append(reward)
            done_l.append(t == horizon - 1)
            obs = next_obs
    return {
        "obs": np.stack(obs_l, axis=0).astype(np.float32),
        "act": np.asarray(act_l, dtype=np.int64),
        "rew": np.asarray(rew_l, dtype=np.float32),
        "done": np.asarray(done_l, dtype=np.bool_),
        "agent": np.stack(agent_l).astype(np.float32),
        "ball": np.stack(ball_l).astype(np.float32),
        "goal": np.stack(goal_l).astype(np.float32),
    }
