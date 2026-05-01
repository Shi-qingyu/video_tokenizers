#!/usr/bin/env bash

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_video_linear_probe.py \
  --model-type jepa \
  --train-annotation-path data/train_t_polished_v3.json \
  --train-data-root ./data \
  --eval-annotation-path data/eval_polished_v3.json \
  --eval-data-root ./data \
  --jepa-checkpoint ./facebook/vjepa2-vitl-fpc64-256 \
  --output-dir outputs/linear_probe_jepa_ddp \
  --log-file outputs/linear_probe_jepa_ddp/run.log \
  --feature-batch-size 32 \
  --feature-num-workers 8 \
  --epochs 1
