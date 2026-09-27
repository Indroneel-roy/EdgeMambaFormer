import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.datasets import (load_splits, make_test_loader, PolypTrainDataset,    # noqa: E402
                           get_train_transforms, get_val_transforms, set_seed, seed_worker)
from data.augment import AUGMENTATIONS, BaselineTrainDataset, PILTrainDataset   # noqa: E402
from eval.metrics import evaluate_standard                                      # noqa: E402
from models import build_model, Predictor, EMF_VARIANTS, ALL_MODELS             # noqa: E402
from models.losses import (edgemamba_loss, loss_unet, loss_unetpp,              # noqa: E402
                           loss_pranet, loss_sanet, loss_polyppvt)

SCALES = (0.75, 1.0, 1.25)


RECIPES = {
    # EdgeMambaFormer, all eight configurations. Frozen before the ablation grid.
    'emf':      dict(data='albu',  opt='adam',  lr=1e-4, wd=1e-4, sched='cosine', batch=16,
                     accum=1, multiscale='random', rng='numpy', align=False,
                     clip=('norm', 5.0), amp=True, drop_last=True),
    # Polyp-PVT, from its paper's Table II.
    'polyppvt': dict(data='pil',   opt='adamw', lr=1e-4, wd=1e-4, sched=None, batch=16,
                     accum=1, multiscale='steps', align=True,
                     clip=('value', 0.5), amp=False, drop_last=False),
    # PraNet, Sec. 2.3.
    'pranet':   dict(data='numpy', opt='adam',  lr=1e-4, wd=0.0, sched=None, batch=16,
                     accum=1, multiscale='steps', align=True,
                     clip=None, amp=False, drop_last=True),
    # SANet, Sec. 4.1. Batch 64 reached as 2 x 32 with gradient accumulation.
    'sanet':    dict(data='numpy', opt='sgd',   lr=0.04, wd=5e-4, sched='linear', batch=32,
                     accum=2, multiscale='random', rng='python', align=True,
                     clip=None, amp=True, drop_last=True, color_exchange=True, pcs=True),
    # U-Net and UNet++: lr and optimiser as reported for this benchmark in PraNet Table 3.
    'unet':     dict(data='numpy', opt='adam',  lr=3e-4, wd=0.0, sched=None, batch=16,
                     accum=1, multiscale=None, align=True,
                     clip=None, amp=False, drop_last=True),
    'unetpp':   dict(data='numpy', opt='adam',  lr=3e-4, wd=0.0, sched=None, batch=16,
                     accum=1, multiscale=None, align=True,
                     clip=None, amp=False, drop_last=True),
}

LOSSES = {'unet': loss_unet, 'unetpp': loss_unetpp, 'pranet': loss_pranet,
          'sanet': loss_sanet, 'polyppvt': loss_polyppvt}

CSV_FIELDS = ['run_id', 'model', 'seed', 'dataset', 'mDice', 'mIoU', 'MAE', 'HD95', 'BIoU',
              'best_epoch', 'val_smooth', 'params', 'epochs', 'minutes', 'timestamp']


def recipe_for(name):
    return RECIPES['emf'] if name in EMF_VARIANTS else RECIPES[name]


def compute_loss(name, out, gt):
    if name in EMF_VARIANTS:
        return edgemamba_loss(out, gt, name, lambda_pred=1.0, lambda_edge=0.3)[0]
    return LOSSES[name](out, gt)


def rescale(img, gt, rate, align):
    s = int(round(352 * rate / 32) * 32)
    if s == img.shape[-1]:
        return img, gt
    kw = dict(mode='bilinear', align_corners=align)
    return F.interpolate(img, (s, s), **kw), F.interpolate(gt, (s, s), **kw)


def make_loaders(name, R, train_pairs, val_pairs, seed):
    g = torch.Generator(); g.manual_seed(seed)
    if R['data'] == 'albu':
        tr = PolypTrainDataset(train_pairs, get_train_transforms())
        va = PolypTrainDataset(val_pairs, get_val_transforms())
    elif R['data'] == 'pil':
        tr, va = PILTrainDataset(train_pairs), PILTrainDataset(val_pairs)
    else:
        tr = BaselineTrainDataset(train_pairs, AUGMENTATIONS[name],
                                  color_exchange_on=R.get('color_exchange', False))
        va = PolypTrainDataset(val_pairs, get_val_transforms())
    tl = DataLoader(tr, batch_size=R['batch'], shuffle=True, num_workers=4, pin_memory=True,
                    drop_last=R['drop_last'], worker_init_fn=seed_worker, generator=g)
    vl = DataLoader(va, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    return tl, vl


def make_optimiser(net, R, steps_per_epoch, epochs):
    p = net.parameters()
    if R['opt'] == 'adam':
        opt = torch.optim.Adam(p, lr=R['lr'], weight_decay=R['wd'])
    elif R['opt'] == 'adamw':
        opt = torch.optim.AdamW(p, lr=R['lr'], weight_decay=R['wd'])
    else:
        opt = torch.optim.SGD(p, lr=R['lr'], momentum=0.9, weight_decay=R['wd'])
    if R['sched'] == 'cosine':                              # stepped once per epoch
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    elif R['sched'] == 'linear':                            # stepped once per update
        total = epochs * steps_per_epoch
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    else:
        sch = None
    return opt, sch


@torch.no_grad()
def quick_val(model, loader, device):
    model.eval(); t = n = 0
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        pb = (torch.sigmoid(model(img)['pred']) >= .5).float(); mb = (msk >= .5).float()
        inter = (pb * mb).sum(dim=(1, 2, 3))
        t += float(((2 * inter) / (pb.sum(dim=(1, 2, 3)) + mb.sum(dim=(1, 2, 3)) + 1e-8)).sum())
        n += img.shape[0]
    return t / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--model', required=True, choices=ALL_MODELS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100, help='100 for every run in the paper')
    ap.add_argument('--out', default=str(ROOT / 'checkpoints'))
    ap.add_argument('--results_csv', default=str(ROOT / 'results' / 'my_runs.csv'))
    ap.add_argument('--smooth_window', type=int, default=5)
    args = ap.parse_args()

    name, R = args.model, recipe_for(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_id = f'{name}_s{args.seed}'
    csv_path = Path(args.results_csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        with open(csv_path) as f:
            if run_id in {r['run_id'] for r in csv.DictReader(f)}:
                print(f'{run_id} already in {csv_path} - skipping'); return

    out = Path(args.out) / run_id; out.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    log = open(ROOT / 'logs' / f'{run_id}.log', 'a')

    def P(*a):
        s = ' '.join(map(str, a)); print(s); log.write(s + '\n'); log.flush()

    set_seed(args.seed)
    P(f'\n=== {run_id} ===  recipe: {json.dumps({k: v for k, v in R.items()})}')
    train_pairs, val_pairs, test_sets = load_splits(args.data_root)
    tl, vl = make_loaders(name, R, train_pairs, val_pairs, args.seed)

    net = build_model(name, pretrained=True).to(device)
    model = Predictor(net, name)
    n_params = sum(p.numel() for p in net.parameters()); P(f'  params {n_params:,}')

    per_step = R['multiscale'] == 'steps'
    steps_ep = len(tl) * 3 if per_step else len(tl) // R['accum']
    opt, sch = make_optimiser(net, R, steps_ep, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=R['amp'])

    start, best, best_ep, hist = 0, -1.0, -1, []
    if (out / 'last.pth').exists():
        ck = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
        net.load_state_dict(ck['model_state']); opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        if sch: sch.load_state_dict(ck['sched'])
        start, best, best_ep, hist = ck['epoch'] + 1, ck['best'], ck['best_epoch'], ck['hist']
        P(f'  resumed at epoch {start}')

    def update():
        if R['clip']:
            scaler.unscale_(opt)
            kind, v = R['clip']
            if kind == 'norm':
                torch.nn.utils.clip_grad_norm_(net.parameters(), v)
            else:
                for p in net.parameters():
                    if p.grad is not None: p.grad.data.clamp_(-v, v)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if R['sched'] == 'linear': sch.step()

    t0 = time.time()
    for ep in range(start, args.epochs):
        net.train(); run_loss = 0.0; n = 0; te = time.time()
        opt.zero_grad(set_to_none=True)
        for it, (img, gt) in enumerate(tqdm(tl, desc=f'{run_id} ep{ep}', leave=False)):
            img, gt = img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            if per_step:
                rates = SCALES
            elif R['multiscale'] == 'random':
                rates = [np.random.choice(SCALES) if R['rng'] == 'numpy' else random.choice(SCALES)]
            else:
                rates = [1.0]
            for rate in rates:
                im, g = rescale(img, gt, rate, R['align'])
                with torch.cuda.amp.autocast(enabled=R['amp']):
                    loss = compute_loss(name, net(im), g)
                scaler.scale(loss if per_step else loss / R['accum']).backward()
                if per_step or (it + 1) % R['accum'] == 0:
                    update()
                if rate == 1.0 or not per_step:
                    run_loss += float(loss.detach()); n += 1
        if R['sched'] == 'cosine':
            sch.step()

        vd = quick_val(model, vl, device); hist.append(vd)
        sm = float(np.mean(hist[-args.smooth_window:]))
        if sm > best:
            best, best_ep = sm, ep
            torch.save({'model_state': net.state_dict(), 'epoch': ep, 'val_smooth': sm}, out / 'best.pth')
        torch.save({'model_state': net.state_dict(), 'opt': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'sched': sch.state_dict() if sch else None,
                    'epoch': ep, 'best': best, 'best_epoch': best_ep, 'hist': hist}, out / 'last.pth')
        if ep == start:
            s = time.time() - te
            P(f'  one epoch {s/60:.1f} min -> about {s*(args.epochs-start)/3600:.1f} h for this run')
        if ep % 5 == 0 or ep == args.epochs - 1:
            P(f'  ep {ep:3d}  loss {run_loss/max(n,1):.4f}  val {vd:.4f}  '
              f'smooth {sm:.4f}  best {best:.4f} @{best_ep}')

    minutes = (time.time() - t0) / 60
    net.load_state_dict(torch.load(out / 'best.pth', map_location='cpu', weights_only=False)['model_state'])
    final = Predictor(net, name, use_pcs=R.get('pcs', False))
    P(f'\n  test (checkpoint from epoch {best_ep}, val smooth {best:.4f})')
    rows = []
    for ds, pairs in test_sets.items():
        r = evaluate_standard(final, make_test_loader(pairs), device, full_metrics=True)
        P(f'    {ds:<20s} mDice {r["mDice"]:.4f}  mIoU {r["mIoU"]:.4f}  '
          f'HD95 {r["HD95"]:7.2f}  BIoU {r["BIoU"]:.4f}')
        rows.append({'run_id': run_id, 'model': name, 'seed': args.seed, 'dataset': ds,
                     'mDice': r['mDice'], 'mIoU': r['mIoU'], 'MAE': r['MAE'],
                     'HD95': r.get('HD95'), 'BIoU': r.get('BIoU'), 'best_epoch': best_ep,
                     'val_smooth': best, 'params': n_params, 'epochs': args.epochs,
                     'minutes': round(minutes, 1), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new: w.writeheader()
        w.writerows(rows)
    torch.save({'model_state': net.state_dict()}, out / f'{run_id}_weights.pth')
    (out / 'last.pth').unlink(missing_ok=True)
    P(f'  done in {minutes:.1f} min -> {csv_path}')
    log.close()


if __name__ == '__main__':
    main()
