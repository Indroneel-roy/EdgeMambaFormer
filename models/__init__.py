"""
One entry point for every model in the paper.

    from models import build_model, to_logit, EMF_VARIANTS, BASELINES

`to_logit` applies each method's own inference rule and returns one logit map,
so every model can be scored by the same evaluation code.
"""

import torch

from .edgemambaformer import EdgeMambaFormer
from .baselines import UNet, UNetPP, PraNet, SANet, pcs
from .polyp_pvt import PolypPVT

# The eight configurations of the ablation (paper Table 1).
EMF_VARIANTS = ['full', 'learned_conv', 'random_filter', 'no_decomp',
                'no_aux', 'no_eag', 'no_csmm', 'no_dbtd']

# The five published methods, each implemented from its paper.
BASELINES = ['unet', 'unetpp', 'pranet', 'sanet', 'polyppvt']

ALL_MODELS = EMF_VARIANTS + BASELINES


def build_model(name, pretrained=True):
    if name in EMF_VARIANTS:
        return EdgeMambaFormer(variant=name, pretrained=pretrained)
    if name == 'unet':     return UNet()                      # trains from scratch
    if name == 'unetpp':   return UNetPP()                    # trains from scratch
    if name == 'pranet':   return PraNet(pretrained=pretrained)
    if name == 'sanet':    return SANet(pretrained=pretrained)
    if name == 'polyppvt': return PolypPVT(pretrained=pretrained)
    raise ValueError(f'unknown model {name!r}; choose from {ALL_MODELS}')


def to_logit(name, out, use_pcs=False):
    """Each method's inference rule, as a single (B,1,H,W) logit."""
    if name in EMF_VARIANTS:
        l = out['pred']
    elif name == 'unet':                                      # softmax class 1
        l = out[:, 1:2] - out[:, 0:1]
    elif name == 'unetpp':                                    # mean of the 4 heads
        p = torch.stack([torch.sigmoid(h.float()) for h in out]).mean(0).clamp(1e-6, 1 - 1e-6)
        l = torch.log(p / (1 - p))
    elif name == 'pranet':                                    # prediction is S3
        l = out[-1]
    elif name == 'sanet':
        l = out
    elif name == 'polyppvt':                                  # P1 + P2
        l = out[0] + out[1]
    else:
        raise ValueError(name)
    l = l.float()
    return pcs(l) if use_pcs else l


class Predictor(torch.nn.Module):
    """Wraps any model so it returns {'pred': logit}, as the evaluator expects."""
    def __init__(self, net, name, use_pcs=False):
        super().__init__()
        self.net, self.name, self.use_pcs = net, name, use_pcs

    def forward(self, x):
        return {'pred': to_logit(self.name, self.net(x), self.use_pcs)}
