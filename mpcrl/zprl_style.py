from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class BottleneckBasePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=16, hidden=256):
        super().__init__()
        self.register_buffer('obs_mean', torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer('obs_std', torch.ones(obs_dim, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    @torch.no_grad()
    def set_obs_normalizer(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.obs_mean.device)
        if x.ndim != 2 or x.shape[1] != self.obs_mean.numel():
            raise ValueError(
                f'expected observation array [N,{self.obs_mean.numel()}], got {tuple(x.shape)}'
            )
        self.obs_mean.copy_(x.mean(dim=0))
        self.obs_std.copy_(x.std(dim=0, unbiased=False).clamp_min(1e-3))

    def normalize_obs(self, obs):
        # Clipping only protects the clone from rare OOD spikes; the training
        # distribution remains effectively unchanged after standardization.
        return ((obs - self.obs_mean) / self.obs_std).clamp(-10.0, 10.0)

    def encode(self, obs):
        return self.encoder(self.normalize_obs(obs))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, obs):
        return self.decode(self.encode(obs))


def fit_behavior_clone(
    policy,
    obs,
    target_action_norm,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    device='cpu',
    show_progress=True,
    update_obs_stats=True,
):
    """Fit/refit the ZPRL-style base policy on an expert action dataset.

    The clone uses full-dataset epochs and observation standardization. The
    function can be called repeatedly after DAgger-style dataset aggregation;
    parameters are temporarily unfrozen for each refit and frozen afterwards.
    """
    policy.to(device)
    for p in policy.parameters():
        p.requires_grad_(True)
    policy.train()

    o = torch.as_tensor(obs, dtype=torch.float32, device=device)
    a = torch.as_tensor(target_action_norm, dtype=torch.float32, device=device)
    n = len(o)
    if n == 0:
        raise ValueError('behavior-clone dataset is empty')
    if update_obs_stats:
        policy.set_obs_normalizer(o)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    batch_size = max(1, min(int(batch_size), n))
    losses = []
    iterator = tqdm(
        range(int(epochs)),
        desc='ZPRL base-policy BC',
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
    )
    for _ in iterator:
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = F.mse_loss(policy(o[idx]), a[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        if show_progress:
            iterator.set_postfix(mse=f'{epoch_loss:.5f}')

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()
    return float(np.mean(losses[-min(5, len(losses)):]))
