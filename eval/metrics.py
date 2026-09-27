import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

EPS = 1e-8



#  Core region metrics


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Per-image Dice and IoU on binary masks."""
    inter = float((pred_bin * gt_bin).sum())
    psum, gsum = float(pred_bin.sum()), float(gt_bin.sum())

    # Both empty -> perfect agreement. Only one empty -> zero.
    if psum == 0 and gsum == 0:
        return 1.0, 1.0
    dice = 2.0 * inter / (psum + gsum + EPS)
    iou = inter / (psum + gsum - inter + EPS)
    return dice, iou


def mae(pred_prob: np.ndarray, gt_bin: np.ndarray):
    """Mean absolute error on the CONTINUOUS map (not thresholded)."""
    return float(np.abs(pred_prob - gt_bin).mean())


#  Boundary metrics 


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    95th-percentile symmetric Hausdorff distance, in pixels at NATIVE resolution.

    Lower is better. Returns NaN when either mask is empty (undefined) — callers
    must use np.nanmean, and you must report how many images were skipped.
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return float('nan')

    # Distance from every pixel to the nearest foreground pixel of the other mask
    dt_g = distance_transform_edt(~g)
    dt_p = distance_transform_edt(~p)

    # Surface voxels only
    p_surf = p ^ binary_erosion(p)
    g_surf = g ^ binary_erosion(g)

    d_pg = dt_g[p_surf]
    d_gp = dt_p[g_surf]
    if d_pg.size == 0 or d_gp.size == 0:
        return float('nan')

    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def boundary_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, dilation_ratio=0.02):
    """
    Boundary IoU (Cheng et al., CVPR 2021). Higher is better.

    Measures agreement in a band around the contour, so unlike Dice it does not
    saturate on large objects. This is where a boundary-refinement module should
    show its effect most clearly.

    The band width scales with image diagonal (default 2%), matching the paper.
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    h, w = g.shape
    d = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))

    # Boundary region = mask minus its d-times erosion
    p_band = p & ~binary_erosion(p, iterations=d)
    g_band = g & ~binary_erosion(g, iterations=d)

    inter = float((p_band & g_band).sum())
    union = float((p_band | g_band).sum())
    return inter / (union + EPS)

#  S-measure and E-measure 


def _object_score(x: np.ndarray):
    if x.size == 0:
        return 0.0
    mu, sigma = x.mean(), x.std()
    return float(2.0 * mu / (mu ** 2 + 1.0 + sigma + EPS))


def _s_object(pred: np.ndarray, gt: np.ndarray):
    fg = pred[gt > 0.5]
    bg = (1.0 - pred)[gt <= 0.5]
    u = float(gt.mean())
    return u * _object_score(fg) + (1.0 - u) * _object_score(bg)


def _ssim(pred: np.ndarray, gt: np.ndarray):
    n = pred.size
    if n == 0:
        return 0.0
    x, y = pred.mean(), gt.mean()
    sx2 = ((pred - x) ** 2).sum() / (n - 1 + EPS)
    sy2 = ((gt - y) ** 2).sum() / (n - 1 + EPS)
    sxy = ((pred - x) * (gt - y)).sum() / (n - 1 + EPS)

    num = 4.0 * x * y * sxy
    den = (x ** 2 + y ** 2) * (sx2 + sy2)
    if den > EPS:
        return float(num / den)
    return 1.0 if num <= EPS else 0.0


def _s_region(pred: np.ndarray, gt: np.ndarray):
    """Split at the GT centroid into 4 quadrants, area-weighted SSIM."""
    h, w = gt.shape
    total = gt.sum()
    if total == 0:
        cy, cx = h // 2, w // 2
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        cy = int(round((ys * gt).sum() / total))
        cx = int(round((xs * gt).sum() / total))
    cy = min(max(cy, 1), h - 1)
    cx = min(max(cx, 1), w - 1)

    quads = [(slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, w)),
             (slice(cy, h), slice(0, cx)), (slice(cy, h), slice(cx, w))]
    score = 0.0
    for sy, sx in quads:
        wgt = gt[sy, sx].size / float(h * w)
        score += wgt * _ssim(pred[sy, sx], gt[sy, sx])
    return score


def s_measure(pred_prob: np.ndarray, gt_bin: np.ndarray, alpha=0.5):
    """Structure measure (Fan et al., ICCV 2017). Higher is better."""
    y = float(gt_bin.mean())
    if y == 0.0:
        return float(1.0 - pred_prob.mean())
    if y == 1.0:
        return float(pred_prob.mean())
    return float(max(0.0, alpha * _s_object(pred_prob, gt_bin)
                     + (1.0 - alpha) * _s_region(pred_prob, gt_bin)))


def e_measure(pred_prob: np.ndarray, gt_bin: np.ndarray):
    """Enhanced-alignment measure (Fan et al., IJCAI 2018). Higher is better."""
    pm = pred_prob - pred_prob.mean()
    gm = gt_bin - gt_bin.mean()
    align = 2.0 * gm * pm / (gm * gm + pm * pm + EPS)
    enhanced = ((align + 1.0) ** 2) / 4.0
    return float(enhanced.sum() / (pred_prob.size - 1 + EPS))



#  The evaluation driver


@torch.no_grad()
def evaluate_standard(model, loader, device, output_key='pred',
                      threshold=0.5, full_metrics=False, return_per_image=False):
    """
    Args
    ----
    model        : returns dict with `output_key`, or a raw tensor
    loader       : yields (img_352, gt_native, name); batch_size MUST be 1
    full_metrics : also compute HD95, Boundary-IoU, S-measure, E-measure
                   (slower — use False during training, True for final tables)

    Returns dict of means. HD95 uses nanmean; `hd95_skipped` counts undefined cases.
    """
    model.eval()
    acc = {k: [] for k in ['dice', 'iou', 'mae', 'hd95', 'biou', 'sm', 'em']}
    names = []

    for batch in loader:
        img, gt, name = batch
        img = img.to(device, non_blocking=True)

        out = model(img)
        logit = out[output_key] if isinstance(out, dict) else out
        if isinstance(logit, (list, tuple)):
            logit = logit[0]

        # ── native-resolution scoring ────────────────────────────────────────
        gt_np = gt.squeeze().cpu().numpy().astype(np.float32)
        H, W = gt_np.shape
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        p = torch.sigmoid(logit).squeeze().cpu().numpy().astype(np.float32)
        p = (p - p.min()) / (p.max() - p.min() + EPS)      # per-image min-max
        p_bin = (p >= threshold).astype(np.float32)
        g_bin = (gt_np >= 0.5).astype(np.float32)

        d, i = dice_iou(p_bin, g_bin)
        acc['dice'].append(d)
        acc['iou'].append(i)
        acc['mae'].append(mae(p, g_bin))

        if full_metrics:
            acc['hd95'].append(hd95(p_bin, g_bin))
            acc['biou'].append(boundary_iou(p_bin, g_bin))
            acc['sm'].append(s_measure(p, g_bin))
            acc['em'].append(e_measure(p, g_bin))

        names.append(name[0] if isinstance(name, (list, tuple)) else name)

    res = {
        'mDice': float(np.mean(acc['dice'])),
        'mIoU': float(np.mean(acc['iou'])),
        'MAE': float(np.mean(acc['mae'])),
        'n': len(acc['dice']),
    }
    if full_metrics:
        h = np.array(acc['hd95'], dtype=np.float64)
        res.update({
            'HD95': float(np.nanmean(h)) if np.isfinite(h).any() else float('nan'),
            'hd95_skipped': int(np.isnan(h).sum()),
            'BIoU': float(np.mean(acc['biou'])),
            'Smeasure': float(np.mean(acc['sm'])),
            'Emeasure': float(np.mean(acc['em'])),
        })
    if return_per_image:
        res['per_image'] = {'name': names, **{k: v for k, v in acc.items() if v}}
    return res
