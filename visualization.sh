MODELS="dino"

python3 visualize_tokenizer_temporal_dynamics.py \
  --annotation-path data/train_t_polished_v3.json \
  --data-root ./data \
  --sample-index 1 \
  --output-dir outputs/temporal_viz/${MODELS} \
  --log-file outputs/temporal_viz/run.log \
  --jepa-checkpoint ./facebook/vjepa2-vitl-fpc64-256 \
  --dino-weights /mnt/bn/xiangtai-training-data-video/sqy/projects/finish/RecTok/offline_models/dinov3_vit_large_patch16 \
  --qwen-model-path ./Qwen/Qwen3-VL-2B-Instruct \
  --models ${MODELS}

# V-JEPA 2.1 raw checkpoint example:
# python3 visualize_tokenizer_temporal_dynamics.py \
#   --annotation-path data/train_t_polished_v3.json \
#   --data-root ./data \
#   --sample-index 1 \
#   --output-dir outputs/temporal_viz_jepa21 \
#   --log-file outputs/temporal_viz_jepa21/run.log \
#   --jepa-checkpoint /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
#   --models jepa
