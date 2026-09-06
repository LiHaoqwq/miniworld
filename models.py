"""Three world-model families on the same observation/action interface.

PixelWM  — Camp "video as simulator": predict x_{t+1} in pixel space.
RSSM     — Dreamer-style latent dynamics: GRU deter state + Gaussian stoch
           state, reconstruct pixels, predict reward, KL-balance prior/posterior.
JEPA     — Predict the *representation* of x_{t+1}, never decode pixels.
           Target encoder is an EMA copy; variance hinge fights collapse.

The point is not SOTA. The point is that the three losses ask the network
for three different things, and those differences show up in long-horizon
rollout and in planning.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(action: torch.Tensor, n: int = 5) -> torch.Tensor:
    return F.one_hot(action.long(), num_classes=n).float()


def foreground_mse(pred: torch.Tensor, target: torch.Tensor, bg: float = 0.10, gain: float = 12.0) -> torch.Tensor:
    """Up-weight disc pixels. Unweighted MSE is dominated by the floor."""
    fg = (target - bg).abs().mean(1, keepdim=True)
    w = 1.0 + gain * (fg > 0.05).float()
    return (w * (pred - target).pow(2)).mean()


class ConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 64, 64] in [0, 1]
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 64 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 8, 8)
        return self.deconv(h)


class PixelWM(nn.Module):
    """Action-conditioned next-frame predictor. No recurrent state."""

    def __init__(self, action_dim: int = 5, embed_dim: int = 64) -> None:
        super().__init__()
        self.encoder = ConvEncoder(embed_dim)
        self.action_dim = action_dim
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.decoder = ConvDecoder(128)
        self.reward = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        e = self.encoder(obs)
        h = self.fuse(torch.cat([e, one_hot(action, self.action_dim)], -1))
        return self.decoder(h), self.reward(h).squeeze(-1)

    def loss(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, rew: torch.Tensor):
        pred, pred_r = self.forward(obs, action)
        recon = foreground_mse(pred, next_obs)
        reward = F.mse_loss(pred_r, rew)
        return recon + 0.1 * reward, {"recon": recon.item(), "reward": reward.item()}

    @torch.no_grad()
    def rollout(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Open-loop pixel rollout. Predicted frame is fed back as input."""
        frames = []
        x = obs
        for t in range(actions.size(1)):
            x, _ = self.forward(x, actions[:, t])
            frames.append(x)
        return torch.stack(frames, 1)


def _split_gauss(x: torch.Tensor, min_std: float = 0.1):
    mean, std_raw = x.chunk(2, dim=-1)
    std = F.softplus(std_raw) + min_std
    return mean, std


def _kl_gauss(mean_p, std_p, mean_q, std_q):
    """KL(p || q) for diagonal Gaussians, mean over batch and dims."""
    var_p, var_q = std_p ** 2, std_q ** 2
    kl = (var_p / var_q + (mean_q - mean_p) ** 2 / var_q - 1.0 + torch.log(var_q) - torch.log(var_p)) * 0.5
    return kl.sum(-1).mean()


class RSSM(nn.Module):
    """Compact Dreamer-style recurrent state-space world model.

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])          # deterministic
    z_t ~ q(z_t | h_t, e_t)  during observe         # posterior
    z_t ~ p(z_t | h_t)       during imagine         # prior
    x̂_t, r̂_t = decode(h_t, z_t)

    Training uses KL balancing (DreamerV3): the prior is pulled toward the
    posterior more strongly than the reverse, plus a free-nats floor so the
    stochastic units do not collapse to the prior.
    """

    def __init__(
        self,
        action_dim: int = 5,
        embed_dim: int = 64,
        deter_dim: int = 64,
        stoch_dim: int = 16,
        kl_balance: float = 0.8,
        free_nats: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.encoder = ConvEncoder(embed_dim)
        self.decoder = ConvDecoder(deter_dim + stoch_dim)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(nn.Linear(deter_dim, 64), nn.SiLU(), nn.Linear(64, 2 * stoch_dim))
        self.post_net = nn.Sequential(nn.Linear(deter_dim + embed_dim, 64), nn.SiLU(), nn.Linear(64, 2 * stoch_dim))
        self.reward = nn.Sequential(nn.Linear(deter_dim + stoch_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def initial(self, batch: int, device: torch.device):
        return {
            "h": torch.zeros(batch, self.deter_dim, device=device),
            "z": torch.zeros(batch, self.stoch_dim, device=device),
        }

    def _feature(self, h, z):
        return torch.cat([h, z], -1)

    def observe_step(self, prev, action_oh, embed, sample: bool = True):
        h = self.gru(torch.cat([prev["z"], action_oh], -1), prev["h"])
        prior_mean, prior_std = _split_gauss(self.prior_net(h))
        post_mean, post_std = _split_gauss(self.post_net(torch.cat([h, embed], -1)))
        if sample:
            z = post_mean + post_std * torch.randn_like(post_mean)
        else:
            z = post_mean
        return {"h": h, "z": z}, (prior_mean, prior_std), (post_mean, post_std)

    def imagine_step(self, prev, action_oh, sample: bool = False):
        h = self.gru(torch.cat([prev["z"], action_oh], -1), prev["h"])
        prior_mean, prior_std = _split_gauss(self.prior_net(h))
        z = prior_mean + prior_std * torch.randn_like(prior_mean) if sample else prior_mean
        return {"h": h, "z": z}

    def observe(self, obs, actions, sample: bool = True):
        """obs: [B, T, 3, H, W], actions: [B, T] action that *led to* obs[:, t] (a_{t-1})."""
        B, T = obs.shape[:2]
        embed = self.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
        state = self.initial(B, obs.device)
        posts, priors, feats, recs, rews = [], [], [], [], []
        for t in range(T):
            a = one_hot(actions[:, t], self.action_dim)
            state, prior, post = self.observe_step(state, a, embed[:, t], sample=sample)
            feat = self._feature(state["h"], state["z"])
            recs.append(self.decoder(feat))
            rews.append(self.reward(feat).squeeze(-1))
            posts.append(post)
            priors.append(prior)
            feats.append(feat)
        recon = torch.stack(recs, 1)
        pred_r = torch.stack(rews, 1)
        return state, recon, pred_r, posts, priors

    def loss(self, obs, actions, rewards):
        _, recon, pred_r, posts, priors = self.observe(obs, actions, sample=True)
        recon_loss = foreground_mse(recon.reshape(-1, *obs.shape[2:]), obs.reshape(-1, *obs.shape[2:]))
        reward_loss = F.mse_loss(pred_r, rewards)
        kl_dyn = kl_rep = 0
        for (p_mean, p_std), (q_mean, q_std) in zip(priors, posts):
            # KL balancing: sg(post)||prior trains dynamics; post||sg(prior) trains encoder.
            kl_dyn = kl_dyn + _kl_gauss(q_mean.detach(), q_std.detach(), p_mean, p_std)
            kl_rep = kl_rep + _kl_gauss(q_mean, q_std, p_mean.detach(), p_std.detach())
        t = max(len(posts), 1)
        kl_dyn, kl_rep = kl_dyn / t, kl_rep / t
        kl = self.kl_balance * kl_dyn + (1.0 - self.kl_balance) * kl_rep
        kl = torch.clamp(kl, min=self.free_nats)
        loss = recon_loss + 0.1 * reward_loss + 0.1 * kl
        return loss, {
            "recon": recon_loss.item(),
            "reward": reward_loss.item(),
            "kl": float(kl.detach()),
        }

    @torch.no_grad()
    def rollout(self, obs_context, actions_future):
        """Encode a 1-step context, then imagine `actions_future` [B, H]."""
        B = obs_context.size(0)
        dummy_a = torch.zeros(B, dtype=torch.long, device=obs_context.device)
        state, _, _, _, _ = self.observe(obs_context[:, None], dummy_a[:, None], sample=False)
        frames = []
        for t in range(actions_future.size(1)):
            state = self.imagine_step(state, one_hot(actions_future[:, t], self.action_dim), sample=False)
            feat = self._feature(state["h"], state["z"])
            frames.append(self.decoder(feat))
        return torch.stack(frames, 1)


class JEPA(nn.Module):
    """Joint-embedding predictive architecture, toy scale.

    context encoder  e_t = Enc(x_t)
    target encoder   s_{t+1} = Enc_EMA(x_{t+1})     # stop-grad + EMA
    predictor        ŝ_{t+1} = Pred(e_t, a_t)
    loss             1 - cosine(ŝ, s) + variance hinge

    No decoder. Planning compares predicted embeddings to a goal embedding.
    """

    def __init__(self, action_dim: int = 5, embed_dim: int = 64, ema: float = 0.99) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.ema = ema
        self.encoder = ConvEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, embed_dim),
        )

    @torch.no_grad()
    def update_target(self) -> None:
        for p, q in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            q.data.mul_(self.ema).add_(p.data, alpha=1.0 - self.ema)

    def encode(self, obs: torch.Tensor, target: bool = False, normalize: bool = True) -> torch.Tensor:
        enc = self.target_encoder if target else self.encoder
        z = enc(obs)
        return F.normalize(z, dim=-1) if normalize else z

    def forward(self, obs: torch.Tensor, action: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        e = self.encode(obs, target=False, normalize=True)
        pred = self.predictor(torch.cat([e, one_hot(action, self.action_dim)], -1))
        return F.normalize(pred, dim=-1) if normalize else pred

    def loss(self, obs, action, next_obs):
        pred_raw = self.forward(obs, action, normalize=False)
        pred = F.normalize(pred_raw, dim=-1)
        with torch.no_grad():
            target = self.encode(next_obs, target=True, normalize=True)
        cos = (pred * target).sum(-1).mean()
        # Variance on *unnormalized* predictions. After L2-norm, per-dim std
        # of a 64-d unit vector is ~1/sqrt(64) and a hinge at 1 can never fire.
        std = pred_raw.std(dim=0) + 1e-4
        var_loss = F.relu(1.0 - std).mean()
        loss = (1.0 - cos) + 0.5 * var_loss
        return loss, {"cos": float(cos.detach()), "var": float(var_loss.detach())}

    @torch.no_grad()
    def rollout_embed(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Closed-loop embedding rollout. No pixels exist to feed back, so the
        predictor is applied to its own previous prediction — the JEPA analogue
        of compounding error."""
        z = self.encode(obs, target=False)
        outs = []
        for t in range(actions.size(1)):
            z = F.normalize(self.predictor(torch.cat([z, one_hot(actions[:, t], self.action_dim)], -1)), dim=-1)
            outs.append(z)
        return torch.stack(outs, 1)
