python3 visualize_tokenizer_temporal_dynamics.py \
  --annotation-path data/train_t_polished_v3.json \
  --data-root ./data \
  --sample-index 0 \
  --output-dir outputs/temporal_viz \
  --log-file outputs/temporal_viz/run.log \
  --jepa-checkpoint ./facebook/vjepa2-vitl-fpc64-256 \
  --dino-weights ./facebook/dinov3-vitl16-pretrain-lvd1689m \
  --qwen-model-path ./Qwen/Qwen3-VL-2B-Instruct \
  --models qwenvit
