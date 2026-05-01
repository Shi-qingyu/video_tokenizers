#!/usr/bin/env bash

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_video_linear_probe.py \
  --model-type qwenvit \
  --train-annotation-path data/train_t_polished_v3.json \
  --train-data-root ./data \
  --eval-annotation-path data/eval_polished_v3.json \
  --eval-data-root ./data \
  --qwen-model-path ./Qwen/Qwen3-VL-2B-Instruct \
  --output-dir outputs/linear_probe_qwenvit_ddp \
  --log-file outputs/linear_probe_qwenvit_ddp/run.log \
  --feature-batch-size 32 \
  --feature-num-workers 8 \
  --epochs 1
