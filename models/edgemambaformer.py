"""
models/edgemambaformer.py — one class, eleven variants.

VARIANTS
--------
  full              reference model
  learned_conv      A: Haar -> parameter-matched LEARNED depthwise conv   <-- THE PAPER
  random_filter     B: Haar -> frozen RANDOM filters (fixed, but not wavelet)
  no_decomp         C: no decomposition at all; f1 straight into the gate
  no_aux            D: lambda_edge = 0 (handled in the loss, flag kept for logging)
  no_eag            X1: f_e := upsample(proj_high(f4)); no wavelet, no gate
  no_csmm           X2: BiMamba -> identity; projections retained
  no_dbtd           X3: dual-branch decoder -> parameter-matched conv decoder
  image_dwt         E: DWT on the INPUT IMAGE + stride-2 stem (not on f1)
  transformer_csmm  F: BiMamba -> 2-layer transformer over the same tokens
  boundary_target   G: identical model; only the edge target changes in the loss


"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

VARIANTS = [
    'full', 'learned_conv', 'random_filter', 'no_decomp', 'no_aux',
    'no_eag', 'no_csmm', 'no_dbtd', 'image_dwt', 'transformer_csmm',
    'boundary_target',
]



#  Fixed Haar DWT (zero learnable parameters)


def haar_filters(channels, device=None, dtype=None):
    """Four separable 2x2 Haar filters, stacked as a depthwise conv weight."""
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
    ll = torch.outer(lo, lo)
    lh = torch.outer(lo, hi)
    hl = torch.outer(hi, lo)
    hh = torch.outer(hi, hi)
    base = torch.stack([ll, lh, hl, hh], 0)                 # (4,2,2)
    w = base.unsqueeze(1).repeat(channels, 1, 1, 1)         # (4C,1,2,2)
    return w.to(device=device, dtype=dtype)


class FixedHaarDWT(nn.Module):
    """Depthwise stride-2 Haar. Returns the three high-frequency subbands."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.register_buffer('weight', haar_filters(channels), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        y = F.conv2d(x, self.weight.to(x.dtype), stride=2, groups=C)
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 1:].reshape(B, 3 * C, y.shape[-2], y.shape[-1])   # LH,HL,HH


class RandomFixedFilters(nn.Module):
    """Variant B: same shape as Haar, frozen random values. Fixed but not wavelet."""

    def __init__(self, channels, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(3 * channels, 1, 2, 2, generator=g)
        w = w / w.flatten(1).norm(dim=1).view(-1, 1, 1, 1)     # unit norm, like Haar
        self.register_buffer('weight', w, persistent=True)
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        return F.conv2d(x, self.weight.to(x.dtype), stride=2, groups=C)


class LearnedEdgeExtractor(nn.Module):
    """
    Variant A: structurally identical to the Haar path, but the filter values
    are learned. 3*C*2*2 = 768 params at C=64.

    This is the control the paper's central claim requires and currently lacks.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 3 * channels, kernel_size=2, stride=2,
                              groups=channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        return self.conv(x)



#  Wavelet Edge Attention Gate


class WaveletEAG(nn.Module):
    """
    Fuses wavelet edge evidence with deep semantic context through a sigmoid gate.

        f_e = edge_feat * sigma + f4_up * (1 - sigma)

    Reference config has exactly 54,337 learnable parameters (verified against
    the paper's stated figure).
    """

    def __init__(self, c_low=64, c_high=512, variant='full'):
        super().__init__()
        self.variant = variant

        if variant == 'learned_conv':
            self.extract = LearnedEdgeExtractor(c_low)
            edge_in = 3 * c_low
        elif variant == 'random_filter':
            self.extract = RandomFixedFilters(c_low)
            edge_in = 3 * c_low
        elif variant == 'no_decomp':
            self.extract = None                       # f1 passed straight through
            edge_in = c_low
        else:                                          # full / everything else
            self.extract = FixedHaarDWT(c_low)
            edge_in = 3 * c_low

        self.proj_high = nn.Conv2d(c_high, c_low, 1)

        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_in, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        self.edge_refine = nn.Sequential(
            nn.Conv2d(c_low, c_low, 3, padding=1, groups=c_low, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c_low, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_low, 1, 1),
        )

    def forward(self, f1, f4):
        size = f1.shape[-2:]

        if self.extract is None:                       # variant C
            e = f1
        else:
            e = self.extract(f1)
            e = F.interpolate(e, size=size, mode='bilinear', align_corners=False)

        edge_feat = self.edge_refine(self.edge_proj(e))
        f4_up = F.interpolate(self.proj_high(f4), size=size,
                              mode='bilinear', align_corners=False)

        gate_logit = self.gate(torch.cat([edge_feat, f4_up], 1))
        sigma = torch.sigmoid(gate_logit)
        f_e = edge_feat * sigma + f4_up * (1.0 - sigma)
        return f_e, gate_logit


class ImageDWTBranch(nn.Module):
    """
    Variant E: DWT on the INPUT IMAGE (352 -> 176), then a stride-2 stem to
    reach f1 resolution (88) and c_low channels.

    The paper argues the encoder has "already partially discarded" high
    frequencies. Applying the DWT to f1 (a stride-4 encoder output) cannot
    recover them. This variant makes the motivation literally true.
    """

    def __init__(self, c_low=64):
        super().__init__()
        self.dwt = FixedHaarDWT(3)                     # 3 RGB -> 9 subband channels
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, c_low, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_low), nn.ReLU(inplace=True),
        )

    def forward(self, img, size):
        e = self.stem(self.dwt(img))                   # 352 -> 176 -> 88
        if e.shape[-2:] != size:
            e = F.interpolate(e, size=size, mode='bilinear', align_corners=False)
        return e



#  Selective state-space block (S6)


class S6Block(nn.Module):
    """
    Chunked selective scan.

    Within a chunk the recurrence is resolved in parallel via a segment-sum
    decay matrix (all entries <= 1, so numerically stable); state is carried
    across chunks sequentially. Chunk 32 over ~2541 tokens gives ~80 sequential
    steps instead of 2541.
    """

    def __init__(self, d_model, d_state=8, d_conv=3, expand=1, chunk=64):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.chunk = chunk
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _chunk_step(self, h, dA_c, dBx_c, C_c):
        """One chunk: parallel within, state carried across."""
        Lc = dA_c.shape[1]
        cs = torch.cumsum(dA_c, dim=1)                           # (b,Lc,d,n)

        # decay[t,s'] = exp(cs_t - cs_s') for s' <= t, else 0. All <= 1.
        seg = cs.unsqueeze(2) - cs.unsqueeze(1)                  # (b,Lc,Lc,d,n)
        mask = torch.tril(torch.ones(Lc, Lc, device=dA_c.device, dtype=torch.bool))
        decay = torch.exp(seg.clamp(max=0.0)) * mask.view(1, Lc, Lc, 1, 1)

        h_all = (decay * dBx_c.unsqueeze(1)).sum(2) \
                + torch.exp(cs.clamp(max=0.0)) * h.unsqueeze(1)
        return (h_all * C_c.unsqueeze(2)).sum(-1), h_all[:, -1]

    def _scan(self, x, dt, B, C):
        """x,dt: (b,L,d)  B,C: (b,L,n)  ->  (b,L,d)"""
        b, L, d = x.shape
        n = B.shape[-1]
        A = -torch.exp(self.A_log.float())                       # (d,n), negative

        dA = dt.unsqueeze(-1) * A                                # (b,L,d,n)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        h = x.new_zeros(b, d, n)
        ys = []
        for s in range(0, L, self.chunk):
            e = min(s + self.chunk, L)
            args = (h, dA[:, s:e], dBx[:, s:e], C[:, s:e])
            # Checkpointing stores only the chunk-boundary state; the interior
            # decay matrix is O(chunk^2) and is recomputed during backward.
            if self.training and torch.is_grad_enabled():
                y_c, h = torch.utils.checkpoint.checkpoint(
                    self._chunk_step, *args, use_reentrant=False)
            else:
                y_c, h = self._chunk_step(*args)
            ys.append(y_c)

        return torch.cat(ys, 1) + x * self.D.float()

    def forward(self, x):
        b, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)

        xi = self.conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2)
        xi = F.silu(xi)

        dbc = self.x_proj(xi)
        dt, B, C = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))

        y = self._scan(xi.float(), dt.float(), B.float(), C.float())
        y = y.to(x.dtype) * F.silu(z)
        return self.out_proj(y)


class BiMambaBlock(nn.Module):
    """Forward + backward S6 with a residual."""

    def __init__(self, d_model, d_state=8, chunk=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = S6Block(d_model, d_state, chunk=chunk)
        self.bwd = S6Block(d_model, d_state, chunk=chunk)

    def forward(self, x):
        h = self.norm(x)
        return x + self.fwd(h) + self.bwd(h.flip(1)).flip(1)


#  Cross-Scale Mamba Module


class CSMM(nn.Module):
    """f2,f3,f4 -> shared d -> one token sequence -> mixer -> split back."""

    def __init__(self, chans=(128, 320, 512), d_model=64, d_state=8,
                 chunk=64, variant='full'):
        super().__init__()
        self.variant = variant
        self.proj = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in chans])

        if variant == 'no_csmm':                    # X2: identity, projections kept
            self.mixer = None
        elif variant == 'transformer_csmm':         # F: attention instead of Mamba
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                batch_first=True, norm_first=True, dropout=0.0)
            self.mixer = nn.TransformerEncoder(layer, num_layers=2)
        else:
            self.mixer = BiMambaBlock(d_model, d_state, chunk)

    def forward(self, feats):
        proj = [p(f) for p, f in zip(self.proj, feats)]
        if self.mixer is None:
            return proj

        shapes = [f.shape[-2:] for f in proj]
        lens = [h * w for h, w in shapes]
        tokens = torch.cat([f.flatten(2).transpose(1, 2) for f in proj], 1)

        tokens = self.mixer(tokens)

        out, i = [], 0
        for (h, w), L in zip(shapes, lens):
            out.append(tokens[:, i:i + L].transpose(1, 2).reshape(
                tokens.shape[0], -1, h, w))
            i += L
        return out


#  Decoder


class WindowAttention(nn.Module):
    def __init__(self, dim, window=8, heads=4):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window
        ph, pw = (w - H % w) % w, (w - W % w) % w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Hp, Wp = x.shape[-2:]

        x = x.view(B, C, Hp // w, w, Wp // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, C)

        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))

        x = x.view(B, Hp // w, Wp // w, w, w, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return x[:, :, :H, :W]


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, t):
        h = self.norm1(t)
        t = t + self.attn(h, h, h, need_weights=False)[0]
        return t + self.mlp(self.norm2(t))


class DualBranchDecoder(nn.Module):
    """Local windowed attention + global self-attention, merged by cross-attention."""

    def __init__(self, c_low=64, d_model=64, dec=64, window=8, heads=4,
                 max_tokens=1024):
        super().__init__()
        self.in_proj = nn.Conv2d(2 * c_low, dec, 1)
        self.local = WindowAttention(dec, window, heads)
        self.ctx_proj = nn.Conv2d(d_model, dec, 1)

        # FIX: the original allocated 4096 slots but only ever indexed 484,
        # leaving 231,168 parameters that never received a gradient — 45% of
        # the non-encoder budget, in a paper whose claim is parameter efficiency.
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, dec))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.global_attn = SelfAttentionBlock(dec, heads)
        self.norm_q = nn.LayerNorm(dec)
        self.norm_kv = nn.LayerNorm(dec)
        self.cross = nn.MultiheadAttention(dec, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dec, dec, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(dec, 1, 1)

    def _pos(self, N):
        if N <= self.pos_emb.shape[1]:
            return self.pos_emb[:, :N]
        p = self.pos_emb.transpose(1, 2)
        return F.interpolate(p, size=N, mode='linear', align_corners=False).transpose(1, 2)

    def forward(self, f1, f_e, ctx):
        local = self.local(self.in_proj(torch.cat([f1, f_e], 1)))
        B, C, H, W = local.shape

        f2, f3, f4 = ctx
        size = f3.shape[-2:]
        g = (F.adaptive_avg_pool2d(f2, size) + f3
             + F.interpolate(f4, size=size, mode='bilinear', align_corners=False))
        g = self.ctx_proj(g).flatten(2).transpose(1, 2)
        g = self.global_attn(g + self._pos(g.shape[1]))

        q = local.flatten(2).transpose(1, 2)
        a = self.cross(self.norm_q(q), self.norm_kv(g), self.norm_kv(g),
                       need_weights=False)[0]
        a = a.transpose(1, 2).reshape(B, C, H, W)

        return self.head(self.fuse(torch.cat([local, a], 1)))


class ConvDecoder(nn.Module):
    """X3 control: plain conv decoder, parameter-matched to the dual-branch one."""

    def __init__(self, c_low=64, d_model=64, dec=78):
        super().__init__()
        self.in_proj = nn.Conv2d(2 * c_low, dec, 1)
        self.ctx_proj = nn.Conv2d(d_model, dec, 1)
        self.block = nn.Sequential(
            nn.Conv2d(2 * dec, dec, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec), nn.ReLU(inplace=True),
            nn.Conv2d(dec, dec, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec), nn.ReLU(inplace=True),
            nn.Conv2d(dec, dec, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec, 1, 1)

    def forward(self, f1, f_e, ctx):
        local = self.in_proj(torch.cat([f1, f_e], 1))
        f2, f3, f4 = ctx
        size = f3.shape[-2:]
        g = (F.adaptive_avg_pool2d(f2, size) + f3
             + F.interpolate(f4, size=size, mode='bilinear', align_corners=False))
        g = F.interpolate(self.ctx_proj(g), size=local.shape[-2:],
                          mode='bilinear', align_corners=False)
        return self.head(self.block(torch.cat([local, g], 1)))



#  Full model


class EdgeMambaFormer(nn.Module):

    def __init__(self, variant='full', backbone='pvt_v2_b2', pretrained=True,
                 d_model=64, d_state=8, chunk=64, dec=64, window=8, heads=4):
        super().__init__()
        assert variant in VARIANTS, f'unknown variant: {variant}'
        self.variant = variant

        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         features_only=True, out_indices=(0, 1, 2, 3))
        c1, c2, c3, c4 = self.encoder.feature_info.channels()

        if variant == 'no_eag':
            self.eag = None
            self.proj_high = nn.Conv2d(c4, c1, 1)
            self.img_dwt = None
        elif variant == 'image_dwt':
            self.eag = WaveletEAG(c1, c4, 'no_decomp')   # gate fed by the image branch
            self.img_dwt = ImageDWTBranch(c1)
        else:
            self.eag = WaveletEAG(c1, c4, variant)
            self.img_dwt = None

        self.csmm = CSMM((c2, c3, c4), d_model, d_state, chunk, variant)

        if variant == 'no_dbtd':
            self.decoder = ConvDecoder(c1, d_model)
        else:
            self.decoder = DualBranchDecoder(c1, d_model, dec, window, heads)

    def forward(self, x):
        size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)

        if self.eag is None:                              # X1
            f_e = F.interpolate(self.proj_high(f4), size=f1.shape[-2:],
                                mode='bilinear', align_corners=False)
            gate = None
        elif self.img_dwt is not None:                    # E
            e = self.img_dwt(x, f1.shape[-2:])
            f_e, gate = self.eag(e, f4)
        else:
            f_e, gate = self.eag(f1, f4)

        ctx = self.csmm([f2, f3, f4])
        logit = self.decoder(f1, f_e, ctx)

        out = {'pred': F.interpolate(logit, size=size, mode='bilinear',
                                     align_corners=False)}
        if gate is not None:
            out['edge'] = F.interpolate(gate, size=size, mode='bilinear',
                                        align_corners=False)
        return out


def count_params(model):
    """Per-module learnable parameter breakdown, for the paper's efficiency table."""
    def n(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    d = {'total': n(model), 'encoder': n(model.encoder), 'csmm': n(model.csmm),
         'decoder': n(model.decoder)}
    if model.eag is not None:
        d['eag'] = n(model.eag)
    if getattr(model, 'img_dwt', None) is not None:
        d['img_dwt'] = n(model.img_dwt)
    d['non_encoder'] = d['total'] - d['encoder']
    return d
