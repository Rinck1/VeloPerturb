"""Corrected SIMPLE linear kinetic control, not the proposed full GFG model.

Legacy W2 models and checkpoints remain unchanged. New checkpoints must include
the fitted coefficient, prepared flag and the shared log-state projection.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from w2_gfg_v2 import GFGv2, GFGv2Core, nb_logprob
from w3_validation import LogStateProjection


class CorrectedCore(GFGv2Core):
    def forward(self, u, s, u_mean, u_std, s_mean, s_std, u_innov, gamma_ss, *, track=False):
        un, sn = (u - u_mean) / u_std, (s - s_mean) / s_std
        z, code_loss = self.manifold_codebook(self.manifold_encoder(torch.stack((un, sn), -1)))
        recon = self.decoder.net(z.reshape(-1, z.shape[-1])).reshape(*u.shape, 2)
        # Positive raw-count means with no lower bound at the training mean.
        def positive_mean(raw, mean, std):
            m = mean.clamp_min(1e-4)
            offset = m + torch.log(-torch.expm1(-m))  # stable inverse softplus
            return F.softplus(raw * std + offset)
        mu = positive_mean(recon[..., 0], u_mean, u_std)
        ms = positive_mean(recon[..., 1], s_mean, s_std)
        recon_nb = -(nb_logprob(u, mu, F.softplus(self.log_theta_u) + .01).mean()
                     + nb_logprob(s, ms, F.softplus(self.log_theta_s) + .01).mean())
        vs = self.vel_head(torch.stack((un, sn, u_innov / u_std), -1)).squeeze(-1)
        beta, gamma = self.log_beta.exp(), self.log_gamma.exp()
        dyn = ((vs - (beta * u - gamma * s)) / (1 + s)).square().mean()
        anchor = ((self.log_gamma - self.log_beta) - gamma_ss.log()).square().mean()
        return dict(native=recon_nb + code_loss + 20 * dyn + 50 * anchor,
                    vs=vs, vu=torch.zeros_like(vs), recon_nb=recon_nb.detach(),
                    dyn_loss=dyn.detach(), anchor=anchor.detach(), mu=mu.detach(), ms=ms.detach())


class FrozenInnovationControl(GFGv2):
    def __init__(self, genes, state_dim=50, dim=8, codes=32, hidden=(256, 512, 512, 256)):
        super().__init__(genes, state_dim, dim, codes, hidden)
        self.core = CorrectedCore(genes, dim, codes, hidden)
        self.register_buffer('innovation_slope', torch.zeros(genes))
        self.register_buffer('state_log_mean', torch.zeros(genes))
        self.register_buffer('prepared', torch.tensor(False))

    @torch.no_grad()
    def fit_source(self, train_us, *, cell_ids, allowed_ids):
        if bool(self.prepared):
            raise ValueError('Transform already frozen; construct a new model to refit')
        if len(cell_ids) != len(train_us) or len(set(cell_ids)) != len(cell_ids) or not set(cell_ids) <= set(allowed_ids):
            raise ValueError('Source training scope violation')
        if train_us.ndim != 2 or train_us.shape[1] != 2 * self.genes or not torch.isfinite(train_us).all() or (train_us < 0).any():
            raise ValueError('Invalid source U/S')
        u, s = train_us.chunk(2, -1)
        p = LogStateProjection.fit(s.detach().cpu().numpy(), dimensions=len(self.components))
        if p.components.shape != tuple(self.components.shape):
            raise ValueError('Too few source cells/genes for requested PCA dimension')
        components = torch.as_tensor(p.components, dtype=train_us.dtype, device=train_us.device)
        super().prepare(train_us, components)
        # Fixed training ratio is a kinetic residual, NOT a general E[U|S] model.
        self.innovation_slope.copy_(self.gamma_ss)
        self.core.log_gamma.copy_(self.gamma_ss.log())
        self.state_log_mean.copy_(torch.as_tensor(p.mean, dtype=train_us.dtype, device=train_us.device))
        self.velocity_scale.fill_(p.scale)
        self.prepared.fill_(True)

    def prepare(self, *args, **kwargs):
        raise ValueError('Use fit_source with an explicit training scope')

    def innovation(self, u, s):
        if not bool(self.prepared):
            raise ValueError('Source transform not fitted')
        return u - self.innovation_slope * s

    def state(self, gene_us):
        if not bool(self.prepared):
            raise ValueError('Source transform not fitted')
        s = gene_us[:, self.genes:]
        return (torch.log1p(s) - self.state_log_mean) @ self.components.T / self.velocity_scale


def seeded_control(genes, *, seed=0, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return FrozenInnovationControl(genes, **kwargs)
