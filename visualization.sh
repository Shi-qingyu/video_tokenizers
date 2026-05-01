python3 visualize_tokenizer_temporal_dynamics.py \
  --annotation-path data/train_t_polished_v3.json \
  --data-root ./data \
  --sample-index 0 \
  --output-dir outputs/temporal_viz \
  --jepa-checkpoint /path/to/vjepa_checkpoint.pt \
  --dino-weights /path/to/dinov3_checkpoint.pth \
  --qwen-model-path /path/to/Qwen3-VL-2B-Instruct