import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# PraNet and SANet both say "Res2Net-50" [Gao et al., TPAMI]. timm's
# res2net50_26w_4s is that architecture with ImageNet-1K weights.
BACKBONE = 'res2net50_26w_4s'


# =============================================================================
#  U-Net  (Ronneberger et al., MICCAI 2015)
# =============================================================================
class DoubleConv(nn.Module):
    """Sec. 2: two 3x3 convolutions, each followed by ReLU. No BatchNorm."""
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    """Fig. 1 / Sec. 2. Channels double at each downsampling step (64 -> 1024).
    Expansive path: 2x2 up-convolution halving the channels, concatenation with
    the contracting-path feature map, two 3x3 convolutions. Final 1x1 conv to
    the class count. 23 convolutional layers in total.

    Deviation: convolutions are padded (the paper uses unpadded convolutions with
    an overlap-tile strategy for large microscopy images), so the output matches
    the 352x352 input without cropping.
    """
    def __init__(self, base=64, n_classes=2, dropout=0.5):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([DoubleConv(3 if i == 0 else ch[i - 1], ch[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)                              # 2x2 max pool, stride 2
        self.drop = nn.Dropout(dropout)                          # "drop-out ... end of contracting path"
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], n_classes, 1)
        for m in self.modules():                                 # Sec. 3: N(0, sqrt(2/N))
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.drop(skips.pop())
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([skips.pop(), up(x)], 1))
        return self.head(x)                                      # (B, 2, H, W) class scores


# =============================================================================
#  UNet++  (Zhou et al., DLMIA 2018)
# =============================================================================
class UNetPP(nn.Module):
    """Eq. 1. Node x^{i,j}: i indexes downsampling depth, j the convolution along
    the skip pathway. H(.) = two 3x3 conv + ReLU, with k = 32 * 2^i kernels.
    U(.) is a 2x2 transposed convolution; this reading reproduces the paper's
    reported 9.04 M parameters (Table 3) exactly, where parameter-free
    upsampling gives 9.16 M. Deep supervision (Sec. 3.2): a 1x1 conv on each of
    x^{0,1..4}; the four maps are averaged at inference ("accurate mode")."""
    def __init__(self):
        super().__init__()
        k = [32 * 2 ** i for i in range(5)]
        self.k = k
        self.pool = nn.MaxPool2d(2)
        self.node = nn.ModuleDict()
        self.upc = nn.ModuleDict()
        for i in range(5):
            self.node[f'{i}_0'] = DoubleConv(3 if i == 0 else k[i - 1], k[i])
        for j in range(1, 5):
            for i in range(0, 5 - j):
                self.upc[f'{i}_{j}'] = nn.ConvTranspose2d(k[i + 1], k[i], 2, 2)
                self.node[f'{i}_{j}'] = DoubleConv((j + 1) * k[i], k[i])
        self.heads = nn.ModuleList([nn.Conv2d(k[0], 1, 1) for _ in range(4)])

    def forward(self, x):
        X = {}
        for i in range(5):
            X[(i, 0)] = self.node[f'{i}_0'](x if i == 0 else self.pool(X[(i - 1, 0)]))
        for j in range(1, 5):                                    # Eq. 1, j > 0
            for i in range(0, 5 - j):
                prev = [X[(i, t)] for t in range(j)]
                X[(i, j)] = self.node[f'{i}_{j}'](
                    torch.cat(prev + [self.upc[f'{i}_{j}'](X[(i + 1, j - 1)])], 1))
        return [h(X[(0, j)]) for h, j in zip(self.heads, range(1, 5))]


# =============================================================================
#  PraNet  (Fan et al., MICCAI 2020)
# =============================================================================
class BConv(nn.Module):
    """Fig. 1 legend: 'Conv + BN'."""
    def __init__(self, i, o, k=1, p=0, d=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(o)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFB(nn.Module):
    """Receptive-field block of the partial decoder. PraNet Sec. 2.1 uses the
    partial decoder of Wu et al. [29] (CPD), which reduces each high-level
    feature with a multi-branch dilated block before aggregation."""
    def __init__(self, i, o):
        super().__init__()
        self.b0 = BConv(i, o, 1)
        self.b1 = nn.Sequential(BConv(i, o, 1), BConv(o, o, (1, 3), (0, 1)),
                                BConv(o, o, (3, 1), (1, 0)), BConv(o, o, 3, 3, 3))
        self.b2 = nn.Sequential(BConv(i, o, 1), BConv(o, o, (1, 5), (0, 2)),
                                BConv(o, o, (5, 1), (2, 0)), BConv(o, o, 3, 5, 5))
        self.b3 = nn.Sequential(BConv(i, o, 1), BConv(o, o, (1, 7), (0, 3)),
                                BConv(o, o, (7, 1), (3, 0)), BConv(o, o, 3, 7, 7))
        self.cat = BConv(4 * o, o, 3, 1)
        self.res = BConv(i, o, 1)

    def forward(self, x):
        y = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1)
        return F.relu(self.cat(y) + self.res(x))


def _to(t, ref):
    return F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class PartialDecoder(nn.Module):
    """pd(f3, f4, f5) -> global map S_g (Sec. 2.1), following CPD [29]."""
    def __init__(self, c=32):
        super().__init__()
        self.u1, self.u2, self.u3, self.u4 = (BConv(c, c, 3, 1) for _ in range(4))
        self.u5 = BConv(2 * c, 2 * c, 3, 1)
        self.c2 = BConv(2 * c, 2 * c, 3, 1)
        self.c3 = BConv(3 * c, 3 * c, 3, 1)
        self.c4 = BConv(3 * c, 3 * c, 3, 1)
        self.c5 = nn.Conv2d(3 * c, 1, 1)

    def forward(self, x5, x4, x3):
        x4_1 = self.u1(_to(x5, x4)) * x4
        x3_1 = self.u2(_to(x5, x3)) * self.u3(_to(x4, x3)) * x3
        x4_2 = self.c2(torch.cat([x4_1, self.u4(_to(x5, x4))], 1))
        x3_2 = self.c3(torch.cat([x3_1, self.u5(_to(x4_2, x3))], 1))
        return self.c5(self.c4(x3_2))


class ReverseAttention(nn.Module):
    """Sec. 2.2. A_i = 1 - sigmoid(P(S_{i+1}))  (Eq. 2),  R_i = f_i * A_i  (Eq. 1).
    Fig. 1: R_i passes through convolutions and is added to the upsampled deeper
    map to give S_i. The paper does not state the convolution widths."""
    def __init__(self, c_in, mid):
        super().__init__()
        self.red = BConv(c_in, mid, 1)
        self.convs = nn.ModuleList([BConv(mid, mid, 3, 1) for _ in range(3)])
        self.out = BConv(mid, 1, 3, 1)

    def forward(self, f, S_up):
        R = f * (1 - torch.sigmoid(S_up))
        x = F.relu(self.red(R))
        for c in self.convs:
            x = F.relu(c(x))
        return self.out(x) + S_up


class PraNet(nn.Module):
    """Returns (S_g, S_5, S_4, S_3) upsampled to the input. Deep supervision on
    all four (Sec. 2.3); the prediction is S_3."""
    def __init__(self, pretrained=True, c=32, backbone=BACKBONE):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=pretrained,
                                    features_only=True, out_indices=(2, 3, 4))
        c3, c4, c5 = self.bb.feature_info.channels()
        self.rfb3, self.rfb4, self.rfb5 = RFB(c3, c), RFB(c4, c), RFB(c5, c)
        self.pd = PartialDecoder(c)
        self.ra5 = ReverseAttention(c5, 256)
        self.ra4 = ReverseAttention(c4, 64)
        self.ra3 = ReverseAttention(c3, 64)

    def forward(self, x):
        size = x.shape[-2:]
        f3, f4, f5 = self.bb(x)
        Sg = self.pd(self.rfb5(f5), self.rfb4(f4), self.rfb3(f3))
        S5 = self.ra5(f5, _to(Sg, f5))          # global map down-sampled to f5
        S4 = self.ra4(f4, _to(S5, f4))
        S3 = self.ra3(f3, _to(S4, f3))
        U = lambda t: F.interpolate(t, size=size, mode='bilinear', align_corners=False)
        return U(Sg), U(S5), U(S4), U(S3)


# =============================================================================
#  SANet  (Wei et al., MICCAI 2021)
# =============================================================================
class LinearBlock(nn.Sequential):
    """Fig. 2 legend: '1x1 conv + bn + relu'."""
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SANet(nn.Module):
    """Fig. 2: features of the last three Res2Net blocks (res-3, res-4, res-5).
    Each reduced by a 1x1 conv+BN+ReLU; the shallow attention module (Eq. 1-2)
    gates res-4 with res-5 and res-3 with the result; the three maps are
    concatenated and a 1x1 conv gives the logit. The channel width is not
    stated in the paper."""
    def __init__(self, pretrained=True, c=64, backbone=BACKBONE):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=pretrained,
                                    features_only=True, out_indices=(2, 3, 4))
        c3, c4, c5 = self.bb.feature_info.channels()
        self.l3, self.l4, self.l5 = LinearBlock(c3, c), LinearBlock(c4, c), LinearBlock(c5, c)
        self.predict = nn.Conv2d(3 * c, 1, 1)

    @staticmethod
    def sam(fs, fd):
        """Eq. 1: Att = ReLU(Up(f_d)).  Eq. 2: f_s = Att * f_s."""
        return F.relu(_to(fd, fs)) * fs

    def forward(self, x):
        size = x.shape[-2:]
        f3, f4, f5 = self.bb(x)
        o3, o4, o5 = self.l3(f3), self.l4(f4), self.l5(f5)
        s4 = self.sam(o4, o5)
        s3 = self.sam(o3, s4)
        logit = self.predict(torch.cat([_to(o5, o3), _to(s4, o3), s3], 1))
        return F.interpolate(logit, size=size, mode='bilinear', align_corners=False)


def pcs(logit):
    """SANet Eq. 3-4, per image: positive logits divided by the fraction of
    positive pixels, negative logits by the fraction of negative pixels.
    Inference only."""
    l = logit.flatten(1)
    rp = (l > 0).float().mean(1, keepdim=True).clamp_min(1e-8)
    rn = (l < 0).float().mean(1, keepdim=True).clamp_min(1e-8)
    return torch.where(l > 0, l / rp, torch.where(l < 0, l / rn, l)).view_as(logit)
