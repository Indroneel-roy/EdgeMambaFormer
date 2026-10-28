import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']
ID, ZS = DS[:2], DS[2:]

ABLATION = ['full', 'learned_conv', 'random_filter', 'no_decomp',
            'no_aux', 'no_eag', 'no_csmm', 'no_dbtd']
COMPARISON = ['unet', 'unetpp', 'pranet', 'sanet', 'polyppvt', 'full']
NAME = {'full': 'EdgeMambaFormer (full)', 'learned_conv': 'A  Haar -> learned conv',
        'random_filter': 'B  Haar -> fixed random', 'no_decomp': 'C  no decomposition',
        'no_aux': 'D  lambda_aux = 0', 'no_eag': 'X1 w/o WaveletEAG',
        'no_csmm': 'X2 w/o CSMM', 'no_dbtd': 'X3 w/o DBTD',
        'unet': 'U-Net', 'unetpp': 'UNet++', 'pranet': 'PraNet', 'sanet': 'SANet',
        'polyppvt': 'Polyp-PVT'}
# Published values, obtained under each paper's own protocol — for reference only.
PUBLISHED = {'unet': [.818, .823, .512, .398, .710], 'unetpp': [.821, .794, .483, .401, .707],
             'pranet': [.898, .899, .709, .628, .871], 'sanet': [.904, .916, .753, .750, .888],
             'polyppvt': [.917, .937, .808, .787, .900]}
# Wall-clock minutes per 100-epoch run, as logged during training on one RTX 4090.
# These are NOT the controlled per-step timings of Table 8 in the paper: they
# include data loading, validation and checkpointing, and runs were not
# scheduled under identical machine conditions. Use Table 8 for cost claims.
COST = {'full': (54337, 381674, 129), 'learned_conv': (55105, 382442, 134),
        'random_filter': (54337, 381674, 129), 'no_decomp': (46145, 373482, 129),
        'no_aux': (54337, 381674, 129), 'no_eag': (0, 360169, 93),
        'no_csmm': (54337, 351682, 11), 'no_dbtd': (54337, 380664, 84)}


def cell(d, m, ds, col='mDice'):
    s = d[(d.model == m) & (d.dataset == ds)][col]
    return (s.mean(), s.std()) if len(s) else None


def agg(d, m, sets, col='mDice'):
    s = d[(d.model == m) & (d.dataset.isin(sets))]
    if not len(s): return None
    per_seed = s.groupby('seed')[col].mean()
    return per_seed.mean(), per_seed.std()


def header(t):
    print('\n' + '=' * 80 + f'\n  {t}\n' + '=' * 80)


def table_comparison(d):
    header('TABLE 3 - comparison with published methods (mDice, mean +- std, 3 seeds)')
    print(f'  {"Method":<24s}' + ''.join(f'{x[:11]:>16s}' for x in DS) + f'{"zero-shot":>11s}')
    for m in COMPARISON:
        if agg(d, m, ZS) is None: continue
        row = ''.join(f'{cell(d,m,ds)[0]:9.4f}+-{cell(d,m,ds)[1]:.3f}' for ds in DS)
        print(f'  {NAME[m]:<24s}{row}{agg(d,m,ZS)[0]:11.4f}')
    print('\n  reproduction vs published (mean over the five test sets):')
    for m, pub in PUBLISHED.items():
        if agg(d, m, DS) is None: continue
        diffs = [cell(d, m, ds)[0] - p for ds, p in zip(DS, pub)]
        print(f'    {NAME[m]:<12s} {np.mean(diffs):+.4f}')


def table_miou(d):
    header('TABLE 4 - comparison with published methods (mIoU, 3-seed means)')
    print(f'  {"Method":<24s}' + ''.join(f'{x[:11]:>13s}' for x in DS))
    for m in COMPARISON:
        if agg(d, m, ZS) is None: continue
        print(f'  {NAME[m]:<24s}' + ''.join(f'{cell(d,m,ds,"mIoU")[0]:13.4f}' for ds in DS))


def table_ablation(d):
    header('TABLE 5 - all eight configurations (mDice, mean +- std, 3 seeds)')
    print(f'  {"Configuration":<26s}' + ''.join(f'{x[:11]:>16s}' for x in DS)
          + f'{"in-dist":>9s}{"zero-shot":>11s}')
    for m in ABLATION:
        if agg(d, m, ZS) is None: continue
        row = ''.join(f'{cell(d,m,ds)[0]:9.4f}+-{cell(d,m,ds)[1]:.3f}' for ds in DS)
        print(f'  {NAME[m]:<26s}{row}{agg(d,m,ID)[0]:9.4f}{agg(d,m,ZS)[0]:11.4f}')


def zero_shot_per_seed(d, m):
    s = d[(d.model == m) & (d.dataset.isin(ZS))]
    return s.groupby('seed')['mDice'].mean().values


def welch(x, y, conf=0.90):
    """Welch's t-test of mean(x) - mean(y) and its two-sided `conf` CI."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff = x.mean() - y.mean()
    h = stats.t.ppf(0.5 + conf / 2, df) * se
    p = stats.ttest_ind(x, y, equal_var=False).pvalue
    return diff, diff - h, diff + h, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i])); adj[i] = run
    return adj


def table_significance(d, margin=0.02):
    header('TABLE 6 - each configuration vs the full model (zero-shot mDice)')
    print('  Welch t-test over seeds; p_H Holm-corrected over the comparisons;')
    print(f'  90% CI inside +-m establishes equivalence within m (TOST, alpha=0.05), m = {margin}')
    full = zero_shot_per_seed(d, 'full')
    rows = [(m, *welch(zero_shot_per_seed(d, m), full)) for m in ABLATION[1:]
            if agg(d, m, ZS) is not None]
    adj = holm(np.array([r[4] for r in rows]))
    print(f'  {"Configuration":<26s}{"zero-shot":>10s}{"delta":>9s}{"90% CI":>20s}{"p":>7s}{"p_H":>7s}  equiv. +-{margin}')
    for (m, diff, lo, hi, p), ph in zip(rows, adj):
        eq = 'yes' if (lo > -margin and hi < margin) else 'no'
        print(f'  {NAME[m]:<26s}{agg(d,m,ZS)[0]:10.4f}{diff:+9.4f}   [{lo:+.3f}, {hi:+.3f}]'
              f'{p:7.2f}{ph:7.2f}  {eq}')
    means = [agg(d, m, ZS)[0] for m in ABLATION if agg(d, m, ZS)]
    stds = [agg(d, m, ZS)[1] for m in ABLATION if agg(d, m, ZS)]
    print(f'\n  lowest CI bound: {min(r[2] for r in rows):+.4f}')
    print(f'  spread across configurations: {max(means)-min(means):.4f}   '
          f'typical seed std: {np.mean(stds):.4f}')


def table_boundary(d):
    header('TABLE 7 - boundary metrics (3-seed means): BIoU / HD95')
    sets = ['Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    print(f'  {"Configuration":<26s}' + ''.join(f'{x[:11]:>18s}' for x in sets))
    for m in ABLATION:
        if agg(d, m, ZS) is None: continue
        row = ''.join(f'{cell(d,m,ds,"BIoU")[0]:10.4f} /{cell(d,m,ds,"HD95")[0]:6.1f}' for ds in sets)
        print(f'  {NAME[m]:<26s}{row}')


def table_cost(d):
    header('SUPPLEMENTARY - parameters and wall-clock minutes per run (see note in COST)')
    print(f'  {"Configuration":<26s}{"EAG params":>11s}{"non-enc":>10s}{"min/run":>9s}{"zero-shot":>11s}')
    for m in ABLATION:
        if agg(d, m, ZS) is None: continue
        e, ne, mins = COST[m]
        print(f'  {NAME[m]:<26s}{e:>11,d}{ne:>10,d}{mins:>9d}{agg(d,m,ZS)[0]:11.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(ROOT / 'results' / 'all_runs.csv'))
    d = pd.read_csv(ap.parse_args().csv)
    if 'model' not in d.columns and 'variant' in d.columns:
        d = d.rename(columns={'variant': 'model'})
    print(f'{len(d)} rows, {d.groupby(["model","seed"]).ngroups} runs, {d.model.nunique()} models')
    table_comparison(d)
    table_miou(d)
    table_ablation(d)
    table_significance(d)
    table_boundary(d)
    table_cost(d)
    print()


if __name__ == '__main__':
    main()
