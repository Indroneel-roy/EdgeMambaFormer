#!/usr/bin/env bash
# Reproduce every run in the paper: 13 models x 3 seeds = 39 runs.
#
#   bash scripts/run_all.sh data/
#
# Runs already in results/my_runs.csv are skipped, and an interrupted run
# resumes from its last epoch, so this can be stopped and restarted safely.
# About 51 GPU-hours on one RTX 4090, most of it the five EdgeMambaFormer
# configurations that keep both the Mamba module and the full decoder.
set -e
DATA=${1:?usage: bash scripts/run_all.sh <data_root>}

MODELS="full learned_conv random_filter no_decomp no_aux no_eag no_csmm no_dbtd \
        unet unetpp pranet sanet polyppvt"

for seed in 42 43 44; do
  for m in $MODELS; do
    python train.py --data_root "$DATA" --model "$m" --seed "$seed"
  done
done

python scripts/make_tables.py --csv results/my_runs.csv
