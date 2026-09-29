"""GFG dual-encoder / soft-VQ / decoder-JVP backend.

Architecture and state-dict names mirror /data/yuchang/GFG/model/{Encoder,
Codebook,Decoder,model}.py. The local RNA branch has gene-shared weights, so
new genes use newly fitted training-only scaling, not MouseBrain statistics.
Deterministic dropout-free execution ensures reconstruction and JVP describe
the same decoder. We retain the original pointwise RNA ODE projection penalty;
its underdetermination is recorded, not treated as independent velocity truth.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import sha256
from .contracts import FitScope


class GFGEncoder(nn.Module):
    def __init__(self, dim=8, hidden=(256, 512, 512, 256)):
        super().__init__()
        layers, width = [], 2
        for next_width in hidden:
            layers += [nn.Linear(width, next_width), nn.GELU(), nn.Dropout(0.)]
            width = next_width
        self.net = nn.Sequential(*layers, nn.Linear(width, dim))
        # Original checkpoints contain both the prototype and cloned layer.
        self.attention_layer = nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=0., batch_first=True)
        self.attention = nn.TransformerEncoder(self.attention_layer, 1)
        self.attention_layer.requires_grad_(False)  # unused original prototype

    def forward(self, u, s):
        shape = (*u.shape, -1)
        x = self.net(torch.stack((u, s), -1).reshape(-1, 2)).reshape(shape)
        return self.attention(x)


class GFGCodebook(nn.Module):
    def __init__(self, codes, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(codes, dim))

    def forward(self, z):
        flat = z.reshape(-1, z.shape[-1])
        e = self.embedding
        distance = flat.square().sum(-1, keepdim=True)+e.square().sum(-1)[None]-2*flat@e.T
        probability = (-distance).softmax(-1)
        quantized = probability@e
        mean = probability.mean(0)
        penalty = F.mse_loss(flat, quantized.detach())+mean.new_tensor(len(e)).log()+(mean*(mean+1e-10).log()).sum()
        return quantized.reshape_as(z), penalty


class GFGBaseDecoder(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        layers, width = [], dim
        for next_width in hidden:
            layers += [nn.Linear(width, next_width), nn.GELU(), nn.Dropout(0.)]
            width = next_width
        self.net = nn.Sequential(*layers, nn.Linear(width, 2))


class GFGDecoder(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.statedecoder = GFGBaseDecoder(dim, hidden)


def rna_ode_projection(u, s, vu, vs):
    """Original per-cell/gene ridge-projected nonnegative-rate penalty.

    Solve in float64: high U/S counts make the original float32 normal equations
    poorly conditioned. This changes numerical precision, not fitted biology.
    """
    dtype = u.dtype
    u, s, vu, vs = [x.double() for x in (u, s, vu, vs)]
    one, zero = torch.ones_like(u), torch.zeros_like(u)
    matrix = torch.stack((torch.stack((one, -u, zero), -1), torch.stack((zero, u, -s), -1)), -2)
    observed = torch.stack((vu, vs), -1)
    normal = matrix.transpose(-1, -2)@matrix + .001*torch.eye(3, device=u.device, dtype=u.dtype)
    rhs = (matrix.transpose(-1, -2)@observed[..., None])
    rates = torch.linalg.solve(normal, rhs).squeeze(-1)
    residual = observed-(matrix@rates[..., None]).squeeze(-1)
    # Normalize only loss units; the decoder velocities themselves are unchanged.
    scale = torch.stack((u.abs()+1., s.abs()+1.), -1)
    return ((residual/scale).square().mean()+.001*F.relu(-rates).mean()).to(dtype)


class GFGCore(nn.Module):
    def __init__(self, dim=8, codes=32, hidden=(256, 512, 512, 256)):
        super().__init__()
        self.manifold_encoder = GFGEncoder(dim, hidden)
        self.velocity_encoder = GFGEncoder(dim, hidden)
        self.manifold_codebook = GFGCodebook(codes, dim)
        self.velocity_codebook = GFGCodebook(codes, dim)
        self.decoder = GFGDecoder(dim, hidden)

    def forward(self, u, s, u_mean, u_std, s_mean, s_std):
        build_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            un, sn = (u-u_mean)/u_std, (s-s_mean)/s_std
            zs, ls = self.manifold_codebook(self.manifold_encoder(un, sn))
            zv, lv = self.velocity_codebook(self.velocity_encoder(un, sn))
            flat_s, flat_v = zs.reshape(-1, zs.shape[-1]), zv.reshape(-1, zv.shape[-1])
            reconstruction, velocity = torch.autograd.functional.jvp(
                self.decoder.statedecoder.net, (flat_s,), (flat_v,), create_graph=build_graph)
            reconstruction, velocity = reconstruction.reshape(*u.shape, 2), velocity.reshape(*u.shape, 2)
            uhat, shat = reconstruction[..., 0]*u_std+u_mean, reconstruction[..., 1]*s_std+s_mean
            vu, vs = velocity[..., 0]*u_std, velocity[..., 1]*s_std
            recon_u, recon_s = ((uhat-u)/u_std).square(), ((shat-s)/s_std).square()
            native = 7.5*recon_u.mean()+2.5*recon_s.mean()+ls+lv+20*rna_ode_projection(u, s, vu, vs)
            per_cell_error = ((recon_u+recon_s).mean(-1)/2).clamp_min(1e-12).sqrt()
        output = dict(vs=vs, vu=vu, shat=shat, uhat=uhat, native=native, reconstruction_error=per_cell_error)
        return output if build_graph else {key: value.detach() for key, value in output.items()}

    def load_pretrained(self, path, *, expected_sha256=None):
        digest = sha256(path)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('GFG checkpoint hash mismatch')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(weights, strict=True)
        return digest


class GFGDynamics(nn.Module):
    """Gene-level [U,S] -> actual JVP velocity -> PCA -> router representation.

    The first argument is the existing S PCA state for shape/alignment only.
    It is never used to fabricate reconstructed gene counts. No c argument.
    """
    reference_batch_size = 8

    def __init__(self, config):
        super().__init__()
        self.genes = config.gfg_genes
        self.core = GFGCore(config.gfg_gene_dim, config.gfg_codes, tuple(config.gfg_hidden))
        self.representation = nn.Sequential(nn.Linear(config.state_dim, config.representation_dim), nn.SiLU())
        self.uncertainty_head = nn.Linear(config.state_dim, 1)
        for name in ('u_mean', 's_mean'):
            self.register_buffer(name, torch.zeros(self.genes))
        for name in ('u_std', 's_std'):
            self.register_buffer(name, torch.ones(self.genes))
        self.register_buffer('components', torch.zeros(config.state_dim, self.genes))
        self.register_buffer('velocity_scale', torch.ones(()))
        self.register_buffer('prepared', torch.tensor(False))
        self.fit_ids_hash = None
        self.last_native_loss = None

    @torch.no_grad()
    def prepare(self, gene_us, components, *, cell_ids, scope: FitScope):
        from .artifacts import object_hash
        ids = scope.validate(cell_ids)
        if len(ids) != len(gene_us) or gene_us.shape[1] != 2*self.genes or not torch.isfinite(gene_us).all() or (gene_us < 0).any():
            raise ValueError('GFG requires aligned nonnegative training gene counts')
        if components.shape != self.components.shape:
            raise ValueError('GFG/PCA gene dimensions differ')
        u, s = gene_us.chunk(2, -1)
        for name, values in (('u', u), ('s', s)):
            getattr(self, name+'_mean').copy_(values.mean(0))
            getattr(self, name+'_std').copy_(values.std(0, unbiased=False).clamp_min(1e-3))
        self.components.copy_(components)
        self.fit_ids_hash = object_hash(sorted(ids))
        self.prepared.fill_(True)

    def forward(self, spliced_pca, gene_us):
        if not self.prepared:
            raise ValueError('GFG training-only normalization/PCA must be prepared first')
        if gene_us.shape != (len(spliced_pca), 2*self.genes) or not torch.isfinite(gene_us).all() or (gene_us < 0).any():
            raise ValueError('GFG input must be nonnegative gene-level [U,S], not PCA coordinates')
        u, s = gene_us.chunk(2, -1)
        output = self.core(u, s, self.u_mean, self.u_std, self.s_mean, self.s_std)
        v = (output['vs']/(1+s))@self.components.T/self.velocity_scale.clamp_min(1e-8)
        r = self.representation(v)
        rho = F.softplus(self.uncertainty_head(v.detach())).squeeze(-1)
        self.last_native_loss = output['native']+.1*(rho-output['reconstruction_error'].detach()).square().mean()
        return r, v, rho

    def reference_loss(self, spliced, gene_us, unused_reference):
        self(spliced, gene_us)
        return self.last_native_loss

    def corrupt(self, gene_us, method, generator=None):
        values = gene_us.clone()
        u = values[:, :self.genes]
        if method == 'shift':
            u.copy_(u.roll(1, 0))
        elif method == 'zero_u':
            u.zero_()
        elif method == 'gaussian':
            noise = torch.randn(u.shape, device=u.device, dtype=u.dtype, generator=generator)
            u.copy_((u+noise*u.square().mean().sqrt()).clamp_min(0))
        else:
            raise ValueError('Unknown GFG corruption')
        return values
