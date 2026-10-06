"""LeWorldModel: ViT encoder + autoregressive causal transformer predictor + SIGReg.

Faithful to the official method (github.com/lucas-maes/le-wm, arXiv:2511.19312):
a from-scratch ViT encoder producing patch tokens, an autoregressive predictor
that is causal over an observation history with AdaLN-Zero action conditioning,
and the objective ``L_pred + lambda * SIGReg`` with no EMA teacher and no
stop-gradient on the target (that absence is LeJEPA's headline claim, not a bug).

What the previous implementation did instead, and why each part mattered
-----------------------------------------------------------------------
1. ``PixelEncoder`` was a 3-layer conv stack ending in ``AdaptiveAvgPool2d(1)``.
   Global average pooling discards spatial layout, which is most of what a world
   model needs from an image.
2. It ended in ``BatchNorm1d(latent_dim, affine=False)``. That standardises every
   marginal before SIGReg sees it, so the Gaussianity test measures essentially
   nothing. Measured on random embeddings: a deliberately anisotropic embedding
   scored 0.00111 and a uniform one 0.00114, both *better* (lower) than a true
   N(0,I) at 0.00213 -- the regulariser was inverted, not merely weak. Only literal
   rank-1 collapse (0.084) was penalised, and BatchNorm already prevents the
   per-dimension form of that.
3. The encoder's terminal ``BatchNorm1d(latent_dim, affine=False)`` made SIGReg
   measure essentially nothing (see ``SIGReg`` for the measured numbers), so the
   two-term objective was effectively single-term. With lambda=0.1 the regulariser
   contributed ~0.2% of the loss against a prediction loss of ~0.93 -- where a
   predictor emitting zeros scores 0.987, i.e. the predictor had learned almost
   nothing either.
4. The conv stack was hardcoded to 128 channels regardless of ``latent_dim``, so
   ``Linear(128, latent_dim)`` with ``latent_dim: 256`` (every shipped rtx5090
   LeWM config) mapped into a 256-d space whose image is a 128-d subspace.
5. A single RGB frame cannot encode velocity, so the latent was structurally
   blind to it on every MuJoCo locomotion task. Official LeWM carries history in
   the *predictor* (length 3 on PushT/OGBench-Cube, 1 on TwoRoom), which is what
   ``history`` does here.

``CNNPixelEncoder``/``MLPActionPredictor`` are kept so the old behaviour stays
reproducible as an explicitly-labelled ablation, selected by
``lewm.encoder: cnn`` / ``lewm.predictor: mlp``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- encoders
class CNNPixelEncoder(nn.Module):
    """Legacy 3-layer conv encoder. Kept only as a labelled ablation.

    Unlike the original, the conv width scales with ``latent_dim`` so the
    projection is not a hidden 128-d bottleneck, and the terminal BatchNorm is
    replaced by LayerNorm so it does not neutralise SIGReg.
    """

    def __init__(self, in_channels=3, latent_dim=192, tokens=1):
        super().__init__()
        w = max(32, latent_dim // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, w, 5, 2, 2), nn.GroupNorm(8, w), nn.GELU(),
            nn.Conv2d(w, w * 2, 5, 2, 2), nn.GroupNorm(8, w * 2), nn.GELU(),
            nn.Conv2d(w * 2, latent_dim, 3, 2, 1), nn.GroupNorm(8, latent_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(latent_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        self.tokens = int(tokens)
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        z = self.norm(self.proj(self.conv(x)))
        return z.unsqueeze(1)  # [B, 1, D] so downstream code is token-shaped


class ViTEncoder(nn.Module):
    """Patch-token ViT encoder (no global pooling, no terminal BatchNorm)."""

    def __init__(self, image_size=96, patch_size=8, in_channels=3, latent_dim=192,
                 depth=6, heads=3, mlp_ratio=4.0):
        super().__init__()
        if image_size % patch_size:
            raise ValueError(
                f'image_size {image_size} must be divisible by patch_size {patch_size}')
        if latent_dim % heads:
            raise ValueError(
                f'lewm.latent_dim ({latent_dim}) must be divisible by '
                f'lewm.encoder_heads ({heads}); e.g. latent_dim=192 with heads in '
                '{1,2,3,4,6,8,12}')
        self.grid = image_size // patch_size
        self.tokens = self.grid * self.grid
        self.latent_dim = int(latent_dim)
        self.patch = nn.Conv2d(in_channels, latent_dim, patch_size, patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.tokens, latent_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=heads, dim_feedforward=int(latent_dim * mlp_ratio),
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)  # [B, T, D]


# -------------------------------------------------------------------- predictors
class MLPActionPredictor(nn.Module):
    """Legacy MLP predictor over a pooled latent. Ablation only."""

    def __init__(self, latent_dim, action_dim, hidden=256, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim))

    def forward(self, z, a, history=None):
        pooled = z.mean(1)
        return self.net(torch.cat([pooled, a], -1)).unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with AdaLN-Zero action conditioning.

    The modulation projection is zero-initialised so the block starts as the
    identity and the action's influence is learned from zero -- this is what makes
    deep action-conditioned predictors trainable without warmup tricks.
    """

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, cond, attn_mask=None):
        s1, sc1, g1, s2, sc2, g2 = self.modulation(cond).unsqueeze(1).chunk(6, dim=-1)
        h = self._mod(self.norm1(x), s1, sc1)
        attn, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g1 * attn
        h = self._mod(self.norm2(x), s2, sc2)
        return x + g2 * self.mlp(h)


class ARPredictor(nn.Module):
    """Autoregressive predictor, causal over an observation history.

    Input is ``history`` frames' worth of patch tokens, flattened into one
    sequence with a learned per-frame temporal embedding and a block-causal mask
    so frame i attends only to frames <= i. The action conditions every block
    through AdaLN-Zero. Output is the predicted next-frame token grid.
    """

    def __init__(self, latent_dim, action_dim, tokens, depth=4, heads=3,
                 history=2, mlp_ratio=4.0, hidden=None):
        super().__init__()
        if latent_dim % heads:
            raise ValueError(
                f'lewm.latent_dim ({latent_dim}) must be divisible by '
                f'lewm.predictor_heads ({heads})')
        self.tokens = int(tokens)
        self.history = max(1, int(history))
        self.latent_dim = int(latent_dim)
        self.action = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.temporal = nn.Parameter(torch.zeros(1, self.history, 1, latent_dim))
        nn.init.trunc_normal_(self.temporal, std=0.02)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(latent_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(latent_dim)
        self.head = nn.Linear(latent_dim, latent_dim)
        self.register_buffer('mask', self._block_causal_mask(), persistent=False)

    def _block_causal_mask(self):
        n = self.history * self.tokens
        frame = torch.arange(n) // self.tokens
        # nn.MultiheadAttention bool mask: True == this position may NOT be
        # attended to. Query at frame i may see keys at frames <= i, so block
        # exactly the keys that live in a strictly later frame.
        return frame.unsqueeze(0) > frame.unsqueeze(1)

    def forward(self, z, a, history=None):
        """z: [B, T, D] most recent frame. history: [B, H-1, T, D] older frames."""
        b = z.shape[0]
        frames = [z] if history is None or self.history == 1 else list(
            torch.unbind(history[:, -(self.history - 1):], dim=1)) + [z]
        while len(frames) < self.history:            # pad by repeating the oldest
            frames.insert(0, frames[0])
        h = torch.stack(frames, dim=1) + self.temporal
        h = h.reshape(b, self.history * self.tokens, self.latent_dim)
        cond = self.action(a)
        mask = self.mask if self.mask.shape[0] == h.shape[1] else self._block_causal_mask().to(h.device)
        for blk in self.blocks:
            h = blk(h, cond, attn_mask=mask)
        h = h.reshape(b, self.history, self.tokens, self.latent_dim)[:, -1]
        return self.head(self.norm(h))


# ------------------------------------------------------------------------ SIGReg
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularisation (Epps-Pulley on random 1-D sketches).

    Compares the empirical characteristic function of random 1-D projections of the
    embedding against the standard normal CF ``exp(-t^2/2)``, integrated against a
    Gaussian weight by trapezoidal quadrature. The CF of a real random variable is
    Hermitian, so integrating t over [0, t_max] and doubling equals the symmetric
    integral.

    Why the old version was inert, precisely
    ----------------------------------------
    Not because of the quadrature, which was fine, but because the encoder fed it
    ``BatchNorm1d(latent_dim, affine=False)`` output. Per-dimension standardisation
    makes every marginal exactly standard, and random 1-D projections of
    standardised high-dimensional data are near-Gaussian by CLT *regardless of the
    covariance*. Measured on 256x128 samples: anisotropic-then-BatchNormed scored
    0.579 and uniform-then-BatchNormed 0.577, both BELOW a true N(0,I) at 0.987 --
    the test ranked bad embeddings as better than the ideal one. On raw embeddings
    the same statistic separates cleanly: anisotropic 28.7 (29x the ideal) and
    rank-1 collapse 43.0 (44x). Removing the BatchNorm is therefore the fix; no
    amount of reweighting would have recovered the signal.

    ``scale_by_n`` multiplies by the *independent* sample count to give the
    textbook Epps-Pulley test statistic. This is the DEFAULT and is the
    batch-size-invariant choice: the empirical CF converges at rate 1/sqrt(N), so
    the unscaled squared error decays as 1/N and the regulariser would silently
    weaken as the batch grows. With the N factor a true N(0,I) scores ~1.0 at any
    N, which makes that value a usable calibration anchor in a training log.

    ``n_independent`` matters for token embeddings: patch tokens from one image
    are strongly correlated, so the number of independent draws is the number of
    *frames*, not the number of tokens. Scaling by the token count instead
    overstates the statistic by the token multiplier (36x at patch 16 on 96px),
    which inflates the gradient by the same factor and makes lambda meaningless.
    All tokens are still used to estimate the characteristic function.
    """

    def __init__(self, knots=17, num_proj=512, t_max=3.0, scale_by_n=True):
        super().__init__()
        self.num_proj = int(num_proj)
        self.scale_by_n = bool(scale_by_n)
        t = torch.linspace(0.0, t_max, knots)
        dx = (t[1] - t[0]) if knots > 1 else torch.tensor(1.0)
        trap = torch.ones_like(t)
        trap[[0, -1]] = 0.5
        window = torch.exp(-0.5 * t.square())
        self.register_buffer('t', t)
        # 2x for the [-t_max, 0] half recovered by Hermitian symmetry.
        self.register_buffer('weights', 2.0 * trap * window * dx)
        self.register_buffer('phi', torch.exp(-0.5 * t.square()))

    def forward(self, z, n_independent=None):
        if n_independent is None:
            # A [B, T, D] input is B independent draws of T correlated tokens.
            n_independent = z.shape[0] if z.dim() >= 2 else 1
        z = z.reshape(-1, z.shape[-1])
        d = z.shape[-1]
        dirs = torch.randn(d, self.num_proj, device=z.device, dtype=z.dtype)
        dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
        h = z @ dirs
        angle = h.unsqueeze(-1) * self.t
        c = angle.cos().mean(0)
        s = angle.sin().mean(0)
        err = (c - self.phi).square() + s.square()
        stat = (err * self.weights).sum(-1).mean()
        return stat * float(n_independent) if self.scale_by_n else stat


# ------------------------------------------------------------------- world model
class LeWorldModel(nn.Module):
    def __init__(self, action_dim, latent_dim=192, hidden=256, sigreg_lambda=0.1,
                 sigreg_projections=512, sigreg_knots=17, encoder='vit', predictor='ar_transformer',
                 image_size=96, patch_size=8, encoder_depth=6, encoder_heads=3,
                 predictor_depth=4, predictor_heads=3, history=2, frame_channels=3,
                 sigreg_scale_by_n=True):
        super().__init__()
        self.history = max(1, int(history))
        self.encoder_kind = encoder
        self.predictor_kind = predictor
        if encoder == 'vit':
            self.encoder = ViTEncoder(image_size, patch_size, frame_channels, latent_dim,
                                      encoder_depth, encoder_heads)
        elif encoder == 'cnn':
            self.encoder = CNNPixelEncoder(frame_channels, latent_dim)
        else:
            raise ValueError(f"encoder must be 'vit' or 'cnn', got {encoder!r}")
        self.tokens = self.encoder.tokens
        if predictor == 'ar_transformer':
            self.predictor = ARPredictor(latent_dim, action_dim, self.tokens,
                                         predictor_depth, predictor_heads, self.history)
        elif predictor == 'mlp':
            self.predictor = MLPActionPredictor(latent_dim, action_dim, hidden)
        else:
            raise ValueError(f"predictor must be 'ar_transformer' or 'mlp', got {predictor!r}")
        self.sigreg = SIGReg(sigreg_knots, sigreg_projections, scale_by_n=sigreg_scale_by_n)
        self.sigreg_lambda = sigreg_lambda
        self.latent_dim = int(latent_dim)

    def encode(self, frames):
        x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames
        return self.encoder(x)

    def pooled(self, z):
        """Mean over patch tokens. Used by the reward probe and diagnostics."""
        return z.mean(1)

    def loss(self, frame_t, action_t, frame_tp1, history=None):
        z = self.encode(frame_t)
        target = self.encode(frame_tp1)          # no stop-grad: LeJEPA by design
        hist = self.encode(history.flatten(0, 1)).reshape(
            history.shape[0], history.shape[1], self.tokens, self.latent_dim
        ) if history is not None else None
        pred = self.predictor(z, action_t, history=hist)
        pred_loss = F.mse_loss(pred.float(), target.float())
        sr = self.sigreg(torch.cat([z, target], 0).float())
        total = pred_loss + self.sigreg_lambda * sr
        return total, {'total': float(total.detach()), 'pred': float(pred_loss.detach()),
                       'sigreg': float(sr.detach())}

    def latent_step(self, z, a, history=None):
        return self.predictor(z, a, history=history)


class RewardProbe(nn.Module):
    """MuJoCo adaptation layer: predicts reward from a frozen pooled latent.

    Not part of LeWM's two-term objective; documented as an adaptation in
    docs/LEWM_REPRODUCTION.md.
    """

    def __init__(self, latent_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z, a):
        pooled = z.mean(1) if z.dim() == 3 else z
        return self.net(torch.cat([pooled, a], -1))
