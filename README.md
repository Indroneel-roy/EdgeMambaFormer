# EdgeMambaFormer

> **Do wavelet, state-space and dual-branch modules improve polyp segmentation with a pretrained PVTv2-B2 encoder? A controlled ablation**
> Indroneel Roy, Mohammad Kamruzzaman Khan Prince — Shahjalal University of Science and Technology


We built EdgeMambaFormer, a polyp-segmentation network that couples a pretrained
PVTv2-B2 encoder with three components — a Wavelet Edge Attention Gate
(WaveletEAG), a Cross-Scale Mamba Module (CSMM) and a Dual-Branch Transformer
Decoder (DBTD) — and tested each one with matched controls and three random seeds.
**No component changes accuracy significantly; equivalence tests bound the
zero-shot benefit of each to at most 1.7 mDice points.**

## Main results

All numbers below are regenerated from `results/all_runs.csv` by
`python scripts/make_tables.py` (table numbers match the paper).

**Ablation** — zero-shot mDice (mean over CVC-ColonDB, ETIS, CVC-300), three seeds (Tables 5–6):

| Configuration | Zero-shot | Δ vs full | 90% CI | p (Holm) |
|---|---|---|---|---|
| EdgeMambaFormer (full) | 0.8200 | — | — | — |
| A — Haar → learned convolution | 0.8205 | +0.0006 | [−0.014, +0.015] | 1.00 |
| B — Haar → frozen **random** filters | 0.8248 | +0.0049 | [−0.015, +0.025] | 1.00 |
| C — no decomposition | 0.8199 | −0.0001 | [−0.017, +0.016] | 1.00 |
| D — no auxiliary loss (λ_aux = 0) | 0.8297 | +0.0097 | [−0.005, +0.024] | 1.00 |
| X1 — without WaveletEAG | 0.8225 | +0.0025 | [−0.014, +0.019] | 1.00 |
| X2 — without CSMM | 0.8240 | +0.0041 | [−0.010, +0.019] | 1.00 |
| X3 — without DBTD | 0.8227 | +0.0027 | [−0.012, +0.017] | 1.00 |

Spread across all eight configurations: **0.0098**; typical seed standard deviation: **0.0078**.

**Comparison** — every method trained by us from its own paper, same protocol, three seeds (Table 3, mDice):

| Method | Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 | Zero-shot |
|---|---|---|---|---|---|---|
| U-Net | 0.7633 | 0.8003 | 0.6126 | 0.4066 | 0.7186 | 0.5793 |
| UNet++ | 0.7723 | 0.8204 | 0.5273 | 0.3938 | 0.6186 | 0.5132 |
| PraNet | 0.9004 | 0.9214 | 0.7159 | 0.6553 | 0.8977 | 0.7563 |
| SANet | 0.9081 | 0.9128 | 0.7569 | 0.7779 | 0.8816 | 0.8055 |
| Polyp-PVT | 0.9061 | 0.9239 | 0.7976 | 0.7499 | 0.8882 | 0.8119 |
| **EdgeMambaFormer** | 0.9168 | 0.9182 | 0.7975 | 0.7979 | 0.8644 | 0.8200 |

The top three are within seed variance of one another on the zero-shot mean.

**Cost.** In a controlled per-step benchmark (paper, Table 8), removing CSMM makes a
training step 9.4× faster and inference 2.2× faster in our plain-PyTorch
implementation. The wall-clock minutes per run printed by `make_tables.py` are
logged run times that include data loading, validation and checkpointing under
uncontrolled machine conditions; they are not used for any claim in the paper.

## Install

```bash
git clone https://github.com/Indroneel-roy/EdgemambaFormer.git
cd EdgemambaFormer
pip install -r requirements.txt
```

## Data

Download `TrainDataset` and `TestDataset` from [PraNet](https://github.com/DengPingFan/PraNet)
(the published file lists, so test sets are identical to those of prior work):

```
data/
  TrainDataset/{images,masks}/        1,450 images (Kvasir-SEG + CVC-ClinicDB)
  TestDataset/{Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300}/{images,masks}/
```

145 validation images are taken from `TrainDataset` with a fixed seed. Test
images are never used for model selection. Before every run, `load_splits()` in
`data/datasets.py` verifies by **content hash (MD5)** — not by filename — that no
image is shared between train, validation and test.

## Train

```bash
python train.py --data_root data/ --model full --seed 42      # one run
bash scripts/run_all.sh data/                                 # all 39 runs
```

`--model` can be any of the 13 models used in the paper:

| Model | What it is |
|---|---|
| `full` | EdgeMambaFormer |
| `learned_conv`, `random_filter`, `no_decomp`, `no_aux` | ablations A–D |
| `no_eag`, `no_csmm`, `no_dbtd` | ablations X1–X3 |
| `unet`, `unetpp`, `pranet`, `sanet`, `polyppvt` | baselines, each implemented from its paper |

Each model trains with the exact recipe used in the paper (see `RECIPES` in
`train.py`); the EdgeMambaFormer recipe was frozen before the ablation grid.
Results go to `results/my_runs.csv`; the paper's results are never overwritten.

`models/edgemambaformer.py` also contains three variants (`image_dwt`,
`transformer_csmm`, `boundary_target`) that were implemented for future work.
They were **not run and are not reported** in the paper.

## Reproduce the tables

```bash
python scripts/make_tables.py                                 # from the paper's runs
python scripts/make_tables.py --csv results/my_runs.csv       # from your own runs
```

## Layout

```
models/      EdgeMambaFormer, the five baselines, losses
data/        splits, content-hash leakage check, training pipelines
eval/        metrics (the only place any metric is computed)
scripts/     make_tables.py, run_all.sh
train.py     trains any model
results/     all_runs.csv — every run behind the paper's results tables
```

## Limitations

All results use one pretrained backbone (PVTv2-B2). With three seeds, the tests
exclude effects larger than about 1.7 zero-shot mDice points but cannot resolve
smaller ones. The auxiliary head is supervised with masks, not boundaries.

## Citation

```bibtex
@article{roy2026controlled,
  title  = {Do wavelet, state-space and dual-branch modules improve polyp
            segmentation with a pretrained {PVTv2-B2} encoder? A controlled ablation},
  author = {Roy, Indroneel and Prince, Mohammad Kamruzzaman Khan},
  note   = {Manuscript submitted to PLOS ONE},
  year   = {2026}
}
```

MIT License.
