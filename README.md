# EdgeMambaFormer



> **Do Boundary-Attention Modules Improve Polyp Segmentation? A Controlled Ablation of Wavelet, State-Space and Dual-Branch Components**
> Indroneel Roy, Mohammad Kamruzzaman Khan Prince — Shahjalal University of Science and Technology

We built EdgeMambaFormer, a polyp-segmentation network with three purpose-built
components, and tested each one with matched controls and three random seeds.
**None of them changes accuracy by more than seed variance.**

## Main results

**Ablation** — zero-shot mDice, mean of three seeds:

| Configuration | zero-shot | vs full |
|---|---|---|
| EdgeMambaFormer (full) | 0.8200 | — |
| A — Haar → learned convolution | 0.8205 | +0.0006 |
| B — Haar → frozen **random** filters | 0.8248 | +0.0049 |
| C — no decomposition | 0.8199 | −0.0001 |
| D — no auxiliary loss | 0.8297 | +0.0097 |
| X1 — without the wavelet gate | 0.8225 | +0.0025 |
| X2 — without the Mamba module | 0.8240 | +0.0041 |
| X3 — without the dual-branch decoder | 0.8227 | +0.0027 |

Spread across all eight: **0.0098**. Typical seed standard deviation: **0.0078**.
Removing the Mamba module cuts training from 129 to 11 minutes per run.

**Comparison** — every method trained by us from its own paper, same protocol, three seeds:

| Method | Kvasir | ClinicDB | ColonDB | ETIS | CVC-300 | zero-shot |
|---|---|---|---|---|---|---|
| U-Net | .7633 | .8003 | .6126 | .4066 | .7186 | .5793 |
| UNet++ | .7723 | .8204 | .5273 | .3938 | .6186 | .5132 |
| PraNet | .9004 | .9214 | .7159 | .6553 | .8977 | .7563 |
| SANet | .9081 | .9128 | .7569 | .7779 | .8816 | .8055 |
| Polyp-PVT | .9061 | .9239 | .7976 | .7499 | .8882 | .8119 |
| **EdgeMambaFormer** | .9168 | .9182 | .7975 | .7979 | .8644 | .8200 |

The top three are within seed variance of one another on the zero-shot mean.

## Install

```bash
git clone https://github.com/Indroneel-roy/EdgemambaFormer.git
cd EdgemambaFormer
pip install -r requirements.txt
```


## Data

Download `TrainDataset` and `TestDataset` from [PraNet](https://github.com/DengPingFan/PraNet):

```
data/
  TrainDataset/{images,masks}/        1,450 images (Kvasir-SEG + CVC-ClinicDB)
  TestDataset/{Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300}/{images,masks}/
```

145 validation images are taken from `TrainDataset` with a fixed seed. Test
images are never used for model selection.

## Train

```bash
python train.py --data_root data/ --model full --seed 42      # one run
bash scripts/run_all.sh data/                                 # all 39 runs, ~51 GPU-hours
```

`--model` can be any of the 13 models:

| Model | What it is |
|---|---|
| `full` | EdgeMambaFormer |
| `learned_conv`, `random_filter`, `no_decomp`, `no_aux` | ablations A–D |
| `no_eag`, `no_csmm`, `no_dbtd` | ablations X1–X3 |
| `unet`, `unetpp`, `pranet`, `sanet`, `polyppvt` | baselines, each implemented from its paper |

Each model trains with the exact recipe used in the paper (see `RECIPES` in
`train.py`). Results go to `results/my_runs.csv`; the paper's results are never
overwritten.




## Layout

```
models/      EdgeMambaFormer, the five baselines, losses
data/        splits, leakage check, training pipelines
eval/        metrics (the only place any metric is computed)
scripts/     tables, figure, pipeline checks, run_all.sh
train.py     trains any model
results/     all_runs.csv — every number in the paper
figures/     generated figures
```

## Limitations

One backbone (PVTv2-B2); three seeds, so effects below ~0.5 mDice cannot be
resolved; the auxiliary head is supervised with masks, not boundaries;


## Citation

```bibtex
@article{roy2026boundary,
  title  = {Do Boundary-Attention Modules Improve Polyp Segmentation? A Controlled
            Ablation of Wavelet, State-Space and Dual-Branch Components},
  author = {Roy, Indroneel and Prince, Mohammad Kamruzzaman Khan},
  year   = {2026}
}
```

MIT License.
