import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    """
    F3Net / PraNet structure loss: boundary-weighted BCE + boundary-weighted IoU.

    `pred` is a raw logit; `mask` is a float binary target. Both (B,1,H,W).
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    p = torch.sigmoid(pred)
    inter = ((p * mask) * weit).sum(dim=(2, 3))
    union = ((p + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def boundary_target(mask, band=4):
    """
    Variant G: turn a filled mask into a boundary band of width ~2*band.

    Section 4.8 concedes the edge branch is supervised with the segmentation
    mask rather than boundary annotations. With this, sigma is supervised
    against an actual boundary, so the "boundary-probability map" language in
    Section 3.2 becomes accurate.

    Uses max-pool dilation/erosion so it stays on-GPU and differentiable-free.
    """
    k = 2 * band + 1
    dil = F.max_pool2d(mask, k, stride=1, padding=band)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=band)
    return (dil - ero).clamp(0, 1)


def edgemamba_loss(outputs, target, variant='full',
                   lambda_pred=1.0, lambda_edge=0.3, band=4):
    """
    Total objective.

      no_aux (D)          -> lambda_edge = 0
      boundary_target (G) -> edge head supervised on a boundary band
      no_eag (X1)         -> no edge head exists; prediction term only
    """
    loss = lambda_pred * structure_loss(outputs['pred'], target)
    parts = {'pred': float(loss.detach())}

    if 'edge' in outputs and variant != 'no_aux' and lambda_edge > 0:
        if variant == 'boundary_target':
            tgt = boundary_target(target, band)
            # A thin band is mostly background, so BCE alone behaves better
            # here than the mask-shaped structure loss.
            e = F.binary_cross_entropy_with_logits(outputs['edge'], tgt)
        else:
            e = structure_loss(outputs['edge'], target)
        loss = loss + lambda_edge * e
        parts['edge'] = float(e.detach())

    parts['total'] = float(loss.detach())
    return loss, parts



#  Baseline losses — 


def soft_dice_loss(logit, gt, eps=1.0):
    p = torch.sigmoid(logit.float())
    inter = (p * gt).sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + eps)).mean()


def loss_unet(out, gt):
    """U-Net Eq. 1: soft-max cross entropy weighted by class frequency (w_c).
    The touching-cell term w_0 of Eq. 2 does not apply to single polyps."""
    t = (gt >= 0.5).long().squeeze(1)
    f = t.float().mean().clamp(1e-4, 1 - 1e-4)
    w = torch.stack([0.5 / (1 - f), 0.5 / f]).to(out.device)
    return F.cross_entropy(out.float(), t, weight=w)


def loss_unetpp(outs, gt):
    """UNet++ Eq. 2 on each deep-supervision head: 0.5 * BCE + Dice, averaged."""
    return sum(0.5 * F.binary_cross_entropy_with_logits(o.float(), gt)
               + soft_dice_loss(o, gt) for o in outs) / len(outs)


def loss_pranet(outs, gt):
    """PraNet: L(G, S_g) + sum_i L(G, S_i), L = weighted IoU + weighted BCE."""
    return sum(structure_loss(o.float(), gt) for o in outs)


def loss_sanet(out, gt):
    """SANet Eq. 5: BCE + Dice."""
    return F.binary_cross_entropy_with_logits(out.float(), gt) + soft_dice_loss(out, gt)


def loss_polyppvt(outs, gt):
    """Polyp-PVT Eq. 11: L_main(P2, G) + L_aux(P1, G)."""
    P1, P2 = outs
    return structure_loss(P2, gt) + structure_loss(P1, gt)
