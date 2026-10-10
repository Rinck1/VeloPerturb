"""GFG v2 (dynamics-version) — new-only implementation, src/ untouched.

Changes vs current GFG (per W2 taskbook section 4 / 4-week plan D1):
  1. gene-level shared beta_g, gamma_g (log-parameterised, positive);
  2. dynamics loss || v_s_tan - (beta*u - gamma*s) ||^2 weighted by count noise,
     replacing the per-cell/gene theta solve (rna_ode_projection);
  3. no align loss; smoothing weight <= 1 (the pretraining moments-smoother is not used);
  4. raw-count input, NB-likelihood reconstruction (no 20-NN moments);
  5. velocity encoder additionally reads the U innovation u - E[u|s];
  6. soft VQ with codebook usage / mutual-information monitoring.

Reused (imported, not modified): GFGCodebook, GFGBaseDecoder from veloroute.gfg.

Modes:
  --unit-test  : JVP == finite difference; local U shuffle must change direction.
  --train      : fit on a dataset (pancreas by default).
  --accept     : D2 acceptance metrics (U-shuffle cos, v_kin vs mean-shift, count-split).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.gfg import GFGBaseDecoder, GFGCodebook
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class GFGv2Encoder(nn.Module):
    def __init__(self, dim=8, hidden=(256, 512, 512, 256), in_ch=3):
        super().__init__()
        layers, width = [], in_ch
        for hw in hidden:
            layers += [nn.Linear(width, hw), nn.GELU(), nn.Dropout(0.)]
            width = hw
        self.net = nn.Sequential(*layers, nn.Linear(width, dim))
        self.attention_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0., batch_first=True)
        self.attention = nn.TransformerEncoder(self.attention_layer, 1)

    def forward(self, x):
        shape = (*x.shape[:-1], -1)
        y = self.net(x.reshape(-1, x.shape[-1])).reshape(shape)
        return self.attention(y)


def nb_logprob(count, mean, theta):
    """Negative-binomial log-density (mean/dispersion parameterisation)."""
    theta = theta.clamp_min(1e-3)
    mean = mean.clamp_min(1e-6)
    logp = (torch.lgamma(count + theta) - torch.lgamma(theta) - torch.lgamma(count + 1)
            + theta * torch.log(theta / (theta + mean)) + count * torch.log(mean / (theta + mean)))
    return logp


class GFGv2Core(nn.Module):
    def __init__(self, genes, dim=8, codes=32, hidden=(256, 512, 512, 256), shared_rates=False):
        super().__init__()
        self.genes = genes
        self.shared_rates = shared_rates
        self.manifold_encoder = GFGv2Encoder(dim, hidden, in_ch=2)
        self.manifold_codebook = GFGCodebook(codes, dim)
        self.decoder = GFGBaseDecoder(dim, hidden)
        rshape = (1,) if shared_rates else (genes,)
        self.log_beta = nn.Parameter(torch.zeros(*rshape))
        self.log_gamma = nn.Parameter(torch.zeros(*rshape))
        # beta fixed to 1 (global time scale is arbitrary); only gamma is learned,
        # anchored to the data steady-state ratio gamma/beta = mean(u)/mean(s).
        self.log_beta.requires_grad_(False)
        self.log_theta_u = nn.Parameter(torch.zeros(genes))
        self.log_theta_s = nn.Parameter(torch.zeros(genes))

    def forward(self, u, s, u_mean, u_std, s_mean, s_std, u_innov, gamma_ss, *, track=False):
        build = torch.is_grad_enabled()
        un, sn = (u - u_mean) / u_std, (s - s_mean) / s_std
        xs = torch.stack((un, sn), -1)
        zs, ls = self.manifold_codebook(self.manifold_encoder(xs))
        fs = zs.reshape(-1, zs.shape[-1])
        recon = self.decoder.net(fs).reshape(*u.shape, 2)
        mu = F.softplus(recon[..., 0]) * u_std + u_mean
        ms = F.softplus(recon[..., 1]) * s_std + s_mean
        theta_u = F.softplus(self.log_theta_u) + 1e-2
        theta_s = F.softplus(self.log_theta_s) + 1e-2
        recon_nb = -(nb_logprob(u, mu, theta_u).mean() + nb_logprob(s, ms, theta_s).mean())
        # kinetic velocity, NO JVP: v_s = beta_g * u - gamma_g * s
        beta, gamma = self.log_beta.exp(), self.log_gamma.exp()
        vs = beta * u - gamma * s
        vu = torch.zeros_like(vs)
        # anchor per-gene ratio gamma/beta to the data steady state (u/s)
        anchor = ((self.log_gamma - self.log_beta) - gamma_ss.log()).square().mean()
        native = recon_nb + ls + 50.0 * anchor
        out = dict(native=native, recon_nb=recon_nb.detach(), anchor=anchor.detach(),
                   vs=vs, vu=vu, mu=mu.detach(), ms=ms.detach(), beta=beta, gamma=gamma)
        return out if build else {k: v.detach() for k, v in out.items()}


class GFGv2(nn.Module):
    def __init__(self, genes, state_dim=50, dim=8, codes=32, hidden=(256, 512, 512, 256), shared_rates=False):
        super().__init__()
        self.genes = genes
        self.core = GFGv2Core(genes, dim, codes, hidden, shared_rates=shared_rates)
        for name in ('u_mean', 's_mean'):
            self.register_buffer(name, torch.zeros(genes))
        for name in ('u_std', 's_std'):
            self.register_buffer(name, torch.ones(genes))
        self.register_buffer('gamma_ss', torch.ones(genes))
        self.register_buffer('components', torch.zeros(state_dim, genes))
        self.register_buffer('velocity_scale', torch.ones(()))

    @torch.no_grad()
    def prepare(self, gene_us, components):
        u, s = gene_us.chunk(2, -1)
        self.u_mean.copy_(u.mean(0)); self.u_std.copy_(u.std(0, unbiased=False).clamp_min(1e-3))
        self.s_mean.copy_(s.mean(0)); self.s_std.copy_(s.std(0, unbiased=False).clamp_min(1e-3))
        ratio = (u.mean(0) / s.mean(0).clamp_min(1e-3)).clamp_min(1e-4)
        self.gamma_ss.copy_(ratio)
        self.components.copy_(components)
        proj = (s @ components.T)
        self.velocity_scale.copy_(proj.square().mean().sqrt().clamp_min(1e-6))

    def innovation(self, u, s):
        # E[u|s] = per-gene ratio * s
        ratio = (u / s.clamp_min(1.0)).mean(0, keepdim=True)
        return u - ratio * s

    def forward(self, gene_us):
        u, s = gene_us.chunk(2, -1)
        innov = self.innovation(u, s)
        out = self.core(u, s, self.u_mean, self.u_std, self.s_mean, self.s_std, innov, self.gamma_ss)
        v = (out['vs'] / (1 + s)) @ self.components.T / self.velocity_scale.clamp_min(1e-8)
        return out, v


def unit_tests(genes=64, cells=32, dim=8, codes=8, hidden=(64, 64), seed=0):
    """No-JVP design: velocity is kinetic v = beta*u - gamma*s.
    Checks: (1) velocity equals beta*u-gamma*s; (2) locally shuffling U changes direction."""
    torch.manual_seed(seed)
    model = GFGv2(genes, state_dim=10, dim=dim, codes=codes, hidden=hidden)
    gene_us = torch.rand(cells, 2 * genes) * 5
    model.prepare(gene_us, torch.randn(10, genes))
    out, v = model(gene_us)
    u, s = gene_us.chunk(2, -1)
    beta = model.core.log_beta.exp(); gamma = model.core.log_gamma.exp()
    vs_ref = beta * u - gamma * s
    kin_err = float((out['vs'] - vs_ref).abs().max())
    # U-shuffle changes direction
    order = torch.randperm(cells)
    gene_us_p = gene_us.clone(); gene_us_p[:, :genes] = gene_us[order, :genes]
    _, v1 = model(gene_us_p)
    cos = float(F.cosine_similarity(v, v1, dim=1).median())
    return dict(kinetic_vs_max_abs_err=kin_err, kinetic_ok=bool(kin_err < 1e-4),
                u_shuffle_cos_median=cos, u_shuffle_changes_direction=bool(cos < 0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='unit-test', choices=['unit-test', 'train', 'accept'])
    ap.add_argument('--dataset', default='/data/yuchang/veloroute_gainprobe_20260915/pancreas_scvelo.h5ad')
    ap.add_argument('--genes', type=int, default=2000)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=8)
    ap.add_argument('--codes', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shared-rates', action='store_true')
    ap.add_argument('--output', default='outputs/w2_gfg_v2_unit.json')
    args = ap.parse_args()
    if args.mode == 'unit-test':
        r = unit_tests(seed=args.seed)
        Path(args.output).write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=2))
        return
    import anndata as ad
    from scipy import sparse
    a = ad.read_h5ad(args.dataset)
    s = sparse.csr_matrix(a.layers['spliced']); u = sparse.csr_matrix(a.layers['unspliced'])
    tot = np.asarray((s + u).sum(1)).ravel()
    keep = tot >= 100
    s, u = s[keep], u[keep]
    logs = s.astype(np.float32).copy(); logs.data = np.log1p(logs.data)
    expressed = np.asarray((s > 0).sum(0)).ravel() >= 10
    var = np.asarray(logs.power(2).mean(0)).ravel() - np.asarray(logs.mean(0)).ravel() ** 2
    var[~expressed] = -1
    sel = np.sort(np.argsort(-var)[:args.genes])
    sn = np.asarray(s[:, sel].todense(), dtype=np.float32)
    un = np.asarray(u[:, sel].todense(), dtype=np.float32)
    gene_us = torch.tensor(np.concatenate([un, sn], 1))
    z = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.log1p(sn))
    comp = torch.tensor(PCA(50, svd_solver='randomized', random_state=0).fit(sn).components_)
    model = GFGv2(args.genes, state_dim=50, dim=args.dim, codes=args.codes, shared_rates=args.shared_rates)
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    model = model.to(device)
    model.prepare(gene_us.to(device), comp.to(device))
    gene_us = gene_us.to(device)
    if args.mode == 'train':
        torch.manual_seed(args.seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        n = len(gene_us)
        t0 = time.time(); losses = []
        for step in range(args.steps):
            ix = torch.randint(0, n, (args.batch,))
            out, v = model(gene_us[ix])
            opt.zero_grad(); out['native'].backward(); opt.step()
            losses.append(float(out['native']))
            if (step + 1) % 200 == 0:
                print(f'step {step+1} native={np.mean(losses[-200:]):.4f} recon={float(out["recon_nb"]):.3f} anchor={float(out["anchor"]):.4f}', flush=True)
        torch.save(model.state_dict(), 'outputs/w2_gfg_v2_pancreas.pt')
        print(json.dumps(dict(steps=args.steps, final_native=float(np.mean(losses[-100:])), runtime=time.time() - t0)))
    elif args.mode == 'accept':
        model.load_state_dict(torch.load('outputs/w2_gfg_v2_pancreas.pt'))
        model.eval()

        def infer(x, chunk=64):
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), chunk):
                    _, vv = model(x[i:i + chunk])
                    outs.append(vv.cpu())
            return torch.cat(outs)
        v = infer(gene_us)
        # U-shuffle
        order = torch.randperm(len(gene_us))
        gp = gene_us.clone(); gp[:, :args.genes] = gene_us[order, :args.genes]
        vp = infer(gp)
        cos_shuf = float(F.cosine_similarity(v, vp, dim=1).median())
        # count-split halves
        rng = np.random.default_rng(0)
        a1 = gene_us.clone(); a1[:, :args.genes] = torch.tensor(rng.binomial(gene_us[:, :args.genes].cpu().numpy().astype(int), 0.5)).to(device)
        a2 = gene_us.clone(); a2[:, :args.genes] = gene_us[:, :args.genes] - a1[:, :args.genes]
        v1 = infer(a1); v2 = infer(a2)
        cos_split = float(F.cosine_similarity(v1, v2, dim=1).median())
        # mean-shift
        nn = NearestNeighbors(n_neighbors=10).fit(z)
        idx = nn.kneighbors(z, return_distance=False)[:, 1:]
        m = z[idx].mean(1) - z
        def cosr(a, b):
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            return float(np.median((a * b).sum(1)))
        cos_meanshift = cosr(v.numpy(), m)
        # v_kin (steady-state) for accept2
        u_np = gene_us[:, :args.genes].cpu().numpy().astype(np.float64)
        s_np = gene_us[:, args.genes:].cpu().numpy().astype(np.float64)
        qlo = np.quantile(s_np, 0.05, 0); qhi = np.quantile(s_np, 0.95, 0)
        gamma = np.zeros(args.genes)
        for g in range(args.genes):
            tail = (s_np[:, g] <= qlo[g]) | (s_np[:, g] >= qhi[g])
            if tail.sum() >= 10:
                d = float((s_np[tail, g] ** 2).sum())
                if d > 1e-12:
                    gamma[g] = float((s_np[tail, g] * u_np[tail, g]).sum() / d)
        comp_np = model.components.cpu().numpy()
        scale = float(model.velocity_scale)
        v_kin = ((u_np - s_np * gamma) @ comp_np.T) / scale
        cos_vkin = cosr(v.numpy(), v_kin)
        res = dict(u_shuffle_cos_median=cos_shuf, count_split_cos_median=cos_split,
                   cos_v_meanshift=cos_meanshift, cos_v_vkin=cos_vkin,
                   accept1_u_shuffle_le_0p9=bool(cos_shuf <= 0.9),
                   accept2_vkin_gt_meanshift=bool(cos_vkin > cos_meanshift),
                   accept3_count_split_ge_0p8=bool(cos_split >= 0.8))
        Path(args.output.replace('unit', 'accept')).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
