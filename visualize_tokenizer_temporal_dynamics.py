#!/usr/bin/env python3
"""
Visualize how different visual tokenizers encode temporal information in video.

The script extracts time-aligned embeddings from:
- V-JEPA / JEPA style video encoders
- DINO image tokenizers (frame-by-frame)
- Qwen-VL visual encoder

For each tokenizer it generates:
1. A temporal self-similarity matrix
2. A frame-aligned PCA visualization of spatial tokens
3. Saved raw reference frames
4. Spatial token change heatmaps between adjacent time steps

Example:
    python3 visualize_tokenizer_temporal_dynamics.py \
        --video-path /path/to/video.mp4 \
        --output-dir outputs/temporal_viz \
        --jepa-checkpoint /path/to/vjepa.pt \
        --dino-weights /path/to/dinov3.pth \
        --qwen-model-path /path/to/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
DINOV3_ROOT = REPO_ROOT / "dinov3"
VJEPA2_ROOT = REPO_ROOT / "vjepa2"
QWEN_FINETUNE_ROOT = REPO_ROOT / "Qwen3-VL" / "qwen-vl-finetune"
LOCAL_FACEBOOK_ROOT = REPO_ROOT / "facebook"
LOCAL_QWEN_ROOT = REPO_ROOT / "Qwen"

for extra_path in (DINOV3_ROOT, VJEPA2_ROOT):
    extra_path_str = str(extra_path)
    if extra_path.exists() and extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_MODELS = ("jepa", "dino", "qwenvit")
JEPA_FAMILIES = ("auto", "vjepa2", "vjepa2_1")


def _default_existing_dir(path: Path) -> Optional[str]:
    return str(path.resolve()) if path.exists() else None


DEFAULT_DINO_REF = _default_existing_dir(LOCAL_FACEBOOK_ROOT / "dinov3-vitl16-pretrain-lvd1689m")
DEFAULT_JEPA_REF = _default_existing_dir(LOCAL_FACEBOOK_ROOT / "vjepa2-vitl-fpc64-256")
DEFAULT_QWEN_REF = _default_existing_dir(LOCAL_QWEN_ROOT / "Qwen3-VL-2B-Instruct")


def get_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save temporal visualization figures.") from exc
    return plt


def get_torchvision_functional():
    try:
        from torchvision.transforms import functional as TF
    except ImportError as exc:
        raise ImportError("torchvision is required for frame and video preprocessing.") from exc
    return TF


def is_local_dir_model_ref(model_ref: Optional[str]) -> bool:
    if not model_ref:
        return False
    path = Path(model_ref).expanduser()
    return path.exists() and path.is_dir()


def infer_jepa_family(args: argparse.Namespace) -> str:
    if args.jepa_family != "auto":
        return args.jepa_family
    if not args.jepa_checkpoint:
        return "vjepa2"
    checkpoint_name = Path(args.jepa_checkpoint).name
    return "vjepa2_1" if "vjepa2_1" in checkpoint_name else "vjepa2"


def infer_jepa_arch(checkpoint_name: str, family: str) -> str:
    if family == "vjepa2_1":
        if "_vitG_" in checkpoint_name:
            return "vit_gigantic_xformers"
        if "_vitg_" in checkpoint_name:
            return "vit_giant_xformers"
        if "_vitl_" in checkpoint_name:
            return "vit_large"
        if "_vitb_" in checkpoint_name:
            return "vit_base"
        return "vit_large"

    checkpoint_name = checkpoint_name.lower()
    if "vitg" in checkpoint_name:
        return "vit_giant_xformers_rope"
    if "vith" in checkpoint_name:
        return "vit_huge_rope"
    return "vit_large_rope"


def default_jepa_input_size(family: str) -> int:
    return 384 if family == "vjepa2_1" else 256


def default_jepa_short_side(input_size: int) -> int:
    return int(256.0 / 224.0 * input_size)


def resolve_jepa_config(args: argparse.Namespace) -> Dict[str, Any]:
    family = infer_jepa_family(args)
    checkpoint_name = Path(args.jepa_checkpoint).name if args.jepa_checkpoint else ""
    arch = args.jepa_arch or infer_jepa_arch(checkpoint_name, family)
    checkpoint_key = args.jepa_checkpoint_key or ("ema_encoder" if family == "vjepa2_1" else "target_encoder")
    input_size = args.jepa_input_size or default_jepa_input_size(family)
    short_side = args.jepa_short_side or default_jepa_short_side(input_size)
    return {
        "family": family,
        "arch": arch,
        "checkpoint_key": checkpoint_key,
        "input_size": input_size,
        "short_side": short_side,
    }


@dataclass
class VideoSample:
    video_path: Path
    frames: np.ndarray
    timestamps: np.ndarray
    fps: float
    total_frames: int


@dataclass
class TemporalFeatures:
    name: str
    time_embeddings: np.ndarray
    token_maps: np.ndarray
    timestamps: np.ndarray
    summary: Dict[str, float]
    metadata: Dict[str, Any]


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=str, default=None, help="Path to a video file.")
    parser.add_argument(
        "--annotation-path",
        type=str,
        default=None,
        help="Optional json/jsonl annotation file. Used when --video-path is not provided.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional data root used to resolve relative video paths from the annotation.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which sample to read from the annotation file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help="Tokenizers to visualize.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save plots and features.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional file to capture stdout/stderr.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--display-frames",
        type=int,
        default=8,
        help="How many uniformly sampled reference frames to render on the top row.",
    )
    parser.add_argument(
        "--max-heatmaps",
        type=int,
        default=6,
        help="Maximum number of adjacent time-step heatmaps to show per tokenizer.",
    )

    parser.add_argument("--dino-arch", type=str, default="dinov3_vitb16")
    parser.add_argument(
        "--dino-weights",
        type=str,
        default=DEFAULT_DINO_REF,
        help="Path to a DINO .pth checkpoint or a local HF DINO model directory.",
    )
    parser.add_argument("--dino-input-size", type=int, default=224)
    parser.add_argument("--dino-short-side", type=int, default=256)
    parser.add_argument("--dino-num-frames", type=int, default=16)

    parser.add_argument(
        "--jepa-family",
        type=str,
        default="auto",
        choices=list(JEPA_FAMILIES),
        help="Which JEPA implementation to use for raw .pt checkpoints. 'auto' infers from the checkpoint path.",
    )
    parser.add_argument(
        "--jepa-arch",
        type=str,
        default=None,
        help="Optional raw PyTorch JEPA architecture override. If omitted, the script infers a matching arch.",
    )
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=DEFAULT_JEPA_REF,
        help="Path to a JEPA .pt checkpoint or a local HF V-JEPA model directory.",
    )
    parser.add_argument(
        "--jepa-checkpoint-key",
        type=str,
        default=None,
        help="Optional raw checkpoint key override, e.g. target_encoder or ema_encoder.",
    )
    parser.add_argument("--jepa-input-size", type=int, default=None)
    parser.add_argument("--jepa-short-side", type=int, default=None)
    parser.add_argument("--jepa-num-frames", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)

    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default=DEFAULT_QWEN_REF,
        help="Path or HF id of a Qwen-VL model directory.",
    )
    parser.add_argument(
        "--qwen-force-frames",
        type=int,
        default=None,
        help="Override Qwen video_processor min/max_frames with a fixed frame count.",
    )
    return parser.parse_args()


def choose_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    handle = log_file.open("a", encoding="utf-8", buffering=1)
    stdout = TeeStream(sys.__stdout__, handle)
    stderr = TeeStream(sys.__stderr__, handle)
    sys.stdout = stdout
    sys.stderr = stderr
    atexit.register(handle.close)
    print(f"[logging] writing combined stdout/stderr to {log_file}")
    print(f"[logging] started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"[logging] command: {' '.join(sys.argv)}")


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotation structure in {path}.")


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video_path:
        return Path(args.video_path).expanduser().resolve()

    if not args.annotation_path:
        raise ValueError("Provide either --video-path or --annotation-path.")

    annotation_path = Path(args.annotation_path).expanduser().resolve()
    samples = read_json_or_jsonl(annotation_path)
    if not samples:
        raise ValueError(f"No samples found in {annotation_path}.")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"--sample-index {args.sample_index} is out of range for {annotation_path}.")

    sample = samples[args.sample_index]
    if isinstance(sample, list):
        if not sample:
            raise ValueError(f"Sample {args.sample_index} is empty.")
        sample = sample[0]
    videos = sample.get("videos") or sample.get("video")
    if not videos:
        raise ValueError(f"Sample {args.sample_index} does not contain a video field.")
    if isinstance(videos, str):
        video_rel_path = videos
    else:
        video_rel_path = videos[0]

    if args.data_root:
        base_root = Path(args.data_root).expanduser().resolve()
    else:
        base_root = annotation_path.parent
    return (base_root / video_rel_path).resolve()


def _read_video_with_decord(video_path: Path) -> Tuple[np.ndarray, float]:
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    return frames, fps


def _read_video_with_torchvision(video_path: Path) -> Tuple[np.ndarray, float]:
    from torchvision.io import read_video

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    fps = float(info.get("video_fps", 0.0))
    return frames.numpy(), fps


def _read_video_with_imageio(video_path: Path) -> Tuple[np.ndarray, float]:
    import imageio.v3 as iio

    frames = iio.imread(str(video_path))
    metadata = iio.immeta(str(video_path))
    fps = float(metadata.get("fps", 0.0))
    return frames, fps


def load_video(video_path: Path) -> Tuple[np.ndarray, float]:
    readers = [_read_video_with_decord, _read_video_with_torchvision, _read_video_with_imageio]
    errors: List[str] = []
    for reader in readers:
        try:
            frames, fps = reader(video_path)
            if frames.ndim != 4:
                raise ValueError(f"Expected frames to have shape [T, H, W, C], got {frames.shape}.")
            if frames.shape[-1] != 3:
                raise ValueError(f"Expected RGB frames, got last dim = {frames.shape[-1]}.")
            return frames, fps
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to read video {video_path} with available backends:\n{joined}")


def sample_video_uniform(video_path: Path, num_frames: int) -> VideoSample:
    frames, fps = load_video(video_path)
    total_frames = int(frames.shape[0])
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames.")

    target_frames = max(2, min(num_frames, total_frames))
    frame_indices = np.unique(np.linspace(0, total_frames - 1, target_frames, dtype=int))
    sampled = frames[frame_indices]
    effective_fps = fps if fps > 0 else float(target_frames)
    timestamps = frame_indices.astype(np.float32) / max(effective_fps, 1e-6)
    return VideoSample(
        video_path=video_path,
        frames=sampled,
        timestamps=timestamps,
        fps=effective_fps,
        total_frames=total_frames,
    )


def to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.clip(frame, 0, 255)
    return frame.astype(np.uint8)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(to_uint8_frame(frame)).permute(2, 0, 1).float() / 255.0
    return tensor


def resize_short_side(frame: torch.Tensor, short_side: int) -> torch.Tensor:
    TF = get_torchvision_functional()
    _, h, w = frame.shape
    if min(h, w) == short_side:
        return frame
    scale = short_side / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return TF.resize(frame, [new_h, new_w], antialias=True)


def center_crop_square(frame: torch.Tensor, size: int) -> torch.Tensor:
    TF = get_torchvision_functional()
    return TF.center_crop(frame, [size, size])


def preprocess_image_frames(
    frames: np.ndarray,
    short_side: int,
    crop_size: int,
    mean: torch.Tensor = IMAGENET_MEAN,
    std: torch.Tensor = IMAGENET_STD,
) -> torch.Tensor:
    processed = []
    for frame in frames:
        tensor = frame_to_tensor(frame)
        tensor = resize_short_side(tensor, short_side)
        tensor = center_crop_square(tensor, crop_size)
        tensor = (tensor - mean) / std
        processed.append(tensor)
    return torch.stack(processed, dim=0)


def preprocess_video_clip(
    frames: np.ndarray,
    short_side: int,
    crop_size: int,
    mean: torch.Tensor = IMAGENET_MEAN,
    std: torch.Tensor = IMAGENET_STD,
) -> torch.Tensor:
    frame_tensor = preprocess_image_frames(frames, short_side=short_side, crop_size=crop_size, mean=mean, std=std)
    return frame_tensor.permute(1, 0, 2, 3)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    feats = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    return feats @ feats.T


def adjacent_cosine_deltas(features: np.ndarray) -> np.ndarray:
    lhs = features[:-1]
    rhs = features[1:]
    lhs = lhs / np.linalg.norm(lhs, axis=1, keepdims=True).clip(min=1e-8)
    rhs = rhs / np.linalg.norm(rhs, axis=1, keepdims=True).clip(min=1e-8)
    cosine = np.sum(lhs * rhs, axis=1)
    return 1.0 - cosine


def distance_from_first(features: np.ndarray) -> np.ndarray:
    first = features[:1]
    lhs = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-8)
    rhs = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    cosine = np.sum(lhs * rhs, axis=1)
    return 1.0 - cosine


def compute_spatial_change_maps(token_maps: np.ndarray) -> np.ndarray:
    diffs = token_maps[1:] - token_maps[:-1]
    return np.linalg.norm(diffs, axis=-1)


def summarize_features(features: np.ndarray) -> Dict[str, float]:
    adjacent = adjacent_cosine_deltas(features)
    from_first = distance_from_first(features)
    return {
        "num_steps": float(features.shape[0]),
        "mean_adjacent_delta": float(adjacent.mean()) if len(adjacent) else 0.0,
        "max_adjacent_delta": float(adjacent.max()) if len(adjacent) else 0.0,
        "mean_distance_from_first": float(from_first.mean()),
        "max_distance_from_first": float(from_first.max()),
    }


def _load_jepa_state_dict(
    checkpoint_path: Path,
    preferred_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}.")

    candidate_keys: List[str] = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["target_encoder", "ema_encoder", "encoder"])

    state_dict = checkpoint
    for key in candidate_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _build_raw_jepa_model(
    jepa_cfg: Dict[str, Any],
    num_frames: int,
    tubelet_size: int,
):
    if jepa_cfg["family"] == "vjepa2_1":
        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        model_kwargs = dict(
            img_size=(jepa_cfg["input_size"], jepa_cfg["input_size"]),
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            patch_size=16,
            uniform_power=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        if not jepa_cfg["arch"].endswith("_rope"):
            model_kwargs["use_rope"] = True
    else:
        from src.models import vision_transformer as vit_encoder

        model_kwargs = dict(
            img_size=(jepa_cfg["input_size"], jepa_cfg["input_size"]),
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )

    if not hasattr(vit_encoder, jepa_cfg["arch"]):
        raise ValueError(f"Unknown JEPA architecture: {jepa_cfg['arch']}")

    model = getattr(vit_encoder, jepa_cfg["arch"])(**model_kwargs)
    return model


def extract_jepa_features(args: argparse.Namespace, device: torch.device, video_path: Path) -> TemporalFeatures:
    if is_local_dir_model_ref(args.jepa_checkpoint):
        from transformers import AutoModel, AutoVideoProcessor

        sample = sample_video_uniform(video_path, args.jepa_num_frames)
        model_ref = str(Path(args.jepa_checkpoint).expanduser().resolve())
        print(f"[JEPA] loading local HF model from {model_ref}")
        processor = AutoVideoProcessor.from_pretrained(model_ref)
        model = AutoModel.from_pretrained(model_ref).to(device).eval()
        video = torch.from_numpy(sample.frames).permute(0, 3, 1, 2)
        inputs = processor(video, return_tensors="pt")
        pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
        if pixel_values is None:
            raise KeyError("AutoVideoProcessor did not return pixel_values_videos or pixel_values")

        with torch.inference_mode():
            tokens = model.get_vision_features(pixel_values.to(device))

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[0].detach().cpu()
        patch_size = int(getattr(model.config, "patch_size", 16))
        tubelet_size = int(getattr(model.config, "tubelet_size", 2))
        sampled_frames = int(sample.frames.shape[0])
        t_steps = max(sampled_frames // max(tubelet_size, 1), 1)

        # AutoVideoProcessor layouts can vary. The last two dims are spatial.
        h_pixels = int(pixel_values.shape[-2])
        w_pixels = int(pixel_values.shape[-1])
        h_tokens = max(h_pixels // max(patch_size, 1), 1)
        w_tokens = max(w_pixels // max(patch_size, 1), 1)

        expected_tokens = t_steps * h_tokens * w_tokens
        if tokens.shape[0] != expected_tokens:
            # Fall back to inferring only the spatial grid from the token count.
            spatial_tokens = max(tokens.shape[0] // max(t_steps, 1), 1)
            side = int(round(math.sqrt(spatial_tokens)))
            if side * side == spatial_tokens:
                h_tokens = side
                w_tokens = side
            else:
                h_tokens = spatial_tokens
                w_tokens = 1
            expected_tokens = t_steps * h_tokens * w_tokens

        if tokens.shape[0] != expected_tokens:
            raise ValueError(
                f"JEPA token reshape mismatch: tokens={tokens.shape[0]}, "
                f"t_steps={t_steps}, h_tokens={h_tokens}, w_tokens={w_tokens}, "
                f"pixel_values_shape={tuple(pixel_values.shape)}"
            )
        tokens = tokens.view(t_steps, h_tokens, w_tokens, -1)
        time_embeddings = tokens.mean(dim=(1, 2)).numpy()
        token_maps = tokens.numpy()

        timestamps = sample.timestamps
        if len(timestamps) != t_steps:
            start_time = float(timestamps[0]) if len(timestamps) else 0.0
            end_time = float(timestamps[-1]) if len(timestamps) else float(t_steps - 1)
            timestamps = np.linspace(start_time, end_time, t_steps, dtype=np.float32)

        return TemporalFeatures(
            name="JEPA",
            time_embeddings=time_embeddings,
            token_maps=token_maps,
            timestamps=timestamps.astype(np.float32),
            summary=summarize_features(time_embeddings),
            metadata={
                "source": "hf_local_dir",
                "model_path": str(Path(args.jepa_checkpoint).expanduser().resolve()),
                "num_sampled_frames": int(sample.frames.shape[0]),
            },
        )

    jepa_cfg = resolve_jepa_config(args)
    sample = sample_video_uniform(video_path, args.jepa_num_frames)
    if sample.frames.shape[0] % args.jepa_tubelet_size != 0:
        target = (sample.frames.shape[0] // args.jepa_tubelet_size) * args.jepa_tubelet_size
        if target < args.jepa_tubelet_size:
            raise ValueError("Not enough sampled frames for the chosen JEPA tubelet size.")
        sample.frames = sample.frames[:target]
        sample.timestamps = sample.timestamps[:target]

    clip = preprocess_video_clip(
        sample.frames,
        short_side=jepa_cfg["short_side"],
        crop_size=jepa_cfg["input_size"],
    ).unsqueeze(0)

    print(
        "[JEPA] raw checkpoint config: "
        f"family={jepa_cfg['family']} arch={jepa_cfg['arch']} "
        f"checkpoint_key={jepa_cfg['checkpoint_key']} "
        f"input_size={jepa_cfg['input_size']} short_side={jepa_cfg['short_side']}"
    )
    model = _build_raw_jepa_model(
        jepa_cfg=jepa_cfg,
        num_frames=int(sample.frames.shape[0]),
        tubelet_size=args.jepa_tubelet_size,
    )
    if args.jepa_checkpoint:
        state_dict = _load_jepa_state_dict(
            Path(args.jepa_checkpoint).expanduser().resolve(),
            preferred_key=jepa_cfg["checkpoint_key"],
        )
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[JEPA] Loaded checkpoint with message: {msg}")
    else:
        print("[JEPA] No checkpoint provided, using randomly initialized weights.")

    model = model.to(device).eval()
    with torch.inference_mode():
        tokens = model(clip.to(device))

    tokens = tokens[0].detach().cpu()
    t_steps = sample.frames.shape[0] // args.jepa_tubelet_size
    h_tokens = jepa_cfg["input_size"] // model.patch_size
    w_tokens = jepa_cfg["input_size"] // model.patch_size
    tokens = tokens.view(t_steps, h_tokens, w_tokens, -1)
    time_embeddings = tokens.mean(dim=(1, 2)).numpy()
    token_maps = tokens.numpy()
    time_timestamps = sample.timestamps.reshape(t_steps, args.jepa_tubelet_size).mean(axis=1)

    return TemporalFeatures(
        name="JEPA",
        time_embeddings=time_embeddings,
        token_maps=token_maps,
        timestamps=time_timestamps,
        summary=summarize_features(time_embeddings),
        metadata={
            "family": jepa_cfg["family"],
            "arch": jepa_cfg["arch"],
            "checkpoint_key": jepa_cfg["checkpoint_key"],
            "input_size": jepa_cfg["input_size"],
            "short_side": jepa_cfg["short_side"],
            "tubelet_size": args.jepa_tubelet_size,
            "num_sampled_frames": int(sample.frames.shape[0]),
        },
    )


def extract_dino_features(args: argparse.Namespace, device: torch.device, video_path: Path) -> TemporalFeatures:
    if is_local_dir_model_ref(args.dino_weights):
        from transformers import AutoImageProcessor, AutoModel

        sample = sample_video_uniform(video_path, args.dino_num_frames)
        model_ref = str(Path(args.dino_weights).expanduser().resolve())
        print(f"[DINO] loading local HF model from {model_ref}")
        processor = AutoImageProcessor.from_pretrained(model_ref)
        model = AutoModel.from_pretrained(model_ref).to(device).eval()
        inputs = processor(images=[to_uint8_frame(frame) for frame in sample.frames], return_tensors="pt")

        with torch.inference_mode():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            patch_tokens = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        pixel_values = inputs["pixel_values"]
        patch_size = int(getattr(model.config, "patch_size", 16))
        h_tokens = max(int(pixel_values.shape[-2] // max(patch_size, 1)), 1)
        w_tokens = max(int(pixel_values.shape[-1] // max(patch_size, 1)), 1)
        n_patch_tokens = h_tokens * w_tokens
        prefix_tokens = max(int(patch_tokens.shape[1] - n_patch_tokens), 0)
        patch_tokens = patch_tokens[:, prefix_tokens:, :].detach().cpu()
        patch_tokens = patch_tokens.view(patch_tokens.shape[0], h_tokens, w_tokens, -1)
        time_embeddings = patch_tokens.mean(dim=(1, 2)).numpy()
        token_maps = patch_tokens.numpy()

        return TemporalFeatures(
            name="DINO",
            time_embeddings=time_embeddings,
            token_maps=token_maps,
            timestamps=sample.timestamps,
            summary=summarize_features(time_embeddings),
            metadata={
                "source": "hf_local_dir",
                "model_path": str(Path(args.dino_weights).expanduser().resolve()),
                "num_sampled_frames": int(sample.frames.shape[0]),
            },
        )

    from dinov3.hub import backbones as dino_backbones

    if not hasattr(dino_backbones, args.dino_arch):
        raise ValueError(f"Unknown DINO architecture: {args.dino_arch}")

    sample = sample_video_uniform(video_path, args.dino_num_frames)
    images = preprocess_image_frames(
        sample.frames,
        short_side=args.dino_short_side,
        crop_size=args.dino_input_size,
    )

    model_kwargs: Dict[str, Any] = {"pretrained": args.dino_weights is None}
    if args.dino_weights is not None:
        model_kwargs["pretrained"] = True
        model_kwargs["weights"] = str(Path(args.dino_weights).expanduser().resolve())
    model = getattr(dino_backbones, args.dino_arch)(**model_kwargs)
    model = model.to(device).eval()

    with torch.inference_mode():
        outputs = model.forward_features(images.to(device))
        patch_tokens = outputs["x_norm_patchtokens"].detach().cpu()

    h_tokens = args.dino_input_size // model.patch_size
    w_tokens = args.dino_input_size // model.patch_size
    patch_tokens = patch_tokens.view(patch_tokens.shape[0], h_tokens, w_tokens, -1)
    time_embeddings = patch_tokens.mean(dim=(1, 2)).numpy()
    token_maps = patch_tokens.numpy()

    return TemporalFeatures(
        name="DINO",
        time_embeddings=time_embeddings,
        token_maps=token_maps,
        timestamps=sample.timestamps,
        summary=summarize_features(time_embeddings),
        metadata={
            "input_size": args.dino_input_size,
            "num_sampled_frames": int(sample.frames.shape[0]),
        },
    )


def _load_qwen_model_and_processor(model_path: str, device: torch.device) -> Tuple[Any, Any]:
    try:
        from transformers import (
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
    except ImportError as exc:
        raise ImportError("transformers is required for Qwen-ViT visualization.") from exc

    lower_name = model_path.lower()
    if "qwen3" in lower_name and "a" in Path(model_path.rstrip("/")).name.lower():
        model_cls = Qwen3VLMoeForConditionalGeneration
    elif "qwen3" in lower_name:
        model_cls = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in lower_name:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = model_cls.from_pretrained(model_path, dtype=dtype)
    except TypeError:
        model = model_cls.from_pretrained(model_path, torch_dtype=dtype)
    return model.to(device).eval(), processor


def _call_qwen_visual(model: Any, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
    call_variants = [
        lambda: model.visual(pixel_values_videos, grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values_videos, video_grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values=pixel_values_videos, grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values=pixel_values_videos, video_grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values_videos),
    ]
    last_error: Optional[Exception] = None
    for fn in call_variants:
        try:
            out = fn()
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Unable to call Qwen visual encoder: {last_error}")

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)) and out:
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(out, dict):
        for key in ("last_hidden_state", "hidden_states", "image_embeds", "video_embeds"):
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and value and isinstance(value[-1], torch.Tensor):
                return value[-1]
    raise TypeError(f"Unsupported Qwen visual output type: {type(out)}")


def _infer_qwen_token_map_shape(
    tokens: torch.Tensor,
    video_grid_thw: torch.Tensor,
    merge_size: int,
) -> Tuple[int, int, int]:
    t, h, w = [int(x) for x in video_grid_thw[0].tolist()]
    token_count = int(tokens.shape[1])
    if token_count == t * h * w:
        return t, h, w

    merged_h = max(h // max(merge_size, 1), 1)
    merged_w = max(w // max(merge_size, 1), 1)
    if token_count == t * merged_h * merged_w:
        return t, merged_h, merged_w

    if token_count % t != 0:
        raise ValueError(
            f"Cannot reshape Qwen tokens: token_count={token_count}, time={t}, grid={tuple(video_grid_thw[0].tolist())}"
        )

    spatial_tokens = token_count // t
    side = int(round(math.sqrt(spatial_tokens)))
    if side * side == spatial_tokens:
        return t, side, side
    return t, spatial_tokens, 1


def extract_qwenvit_features(args: argparse.Namespace, device: torch.device, video_path: Path) -> TemporalFeatures:
    if not args.qwen_model_path:
        raise ValueError("--qwen-model-path is required when qwenvit is enabled.")

    if QWEN_FINETUNE_ROOT.exists():
        qwen_path = str(QWEN_FINETUNE_ROOT)
        if qwen_path not in sys.path:
            sys.path.insert(0, qwen_path)

    model, processor = _load_qwen_model_and_processor(args.qwen_model_path, device)
    if args.qwen_force_frames is not None and hasattr(processor, "video_processor"):
        if hasattr(processor.video_processor, "min_frames"):
            processor.video_processor.min_frames = args.qwen_force_frames
        if hasattr(processor.video_processor, "max_frames"):
            processor.video_processor.max_frames = args.qwen_force_frames

    vp_output = processor.video_processor(
        videos=[str(video_path)],
        return_tensors="pt",
        return_metadata=True,
    )
    pixel_values_videos = vp_output.pixel_values_videos
    video_grid_thw = vp_output.video_grid_thw
    video_metadata = vp_output.video_metadata[0]

    with torch.inference_mode():
        tokens = _call_qwen_visual(
            model,
            pixel_values_videos.to(device),
            video_grid_thw.to(device),
        )

    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    tokens = tokens.detach().cpu().float()
    merge_size = getattr(getattr(processor, "image_processor", processor), "merge_size", 2)
    t_steps, h_tokens, w_tokens = _infer_qwen_token_map_shape(tokens, video_grid_thw, merge_size)
    token_maps = tokens[0].view(t_steps, h_tokens, w_tokens, -1).numpy()
    time_embeddings = token_maps.mean(axis=(1, 2))

    if "frames_indices" in video_metadata and "fps" in video_metadata and video_metadata["fps"]:
        timestamps = np.asarray(video_metadata["frames_indices"], dtype=np.float32) / float(video_metadata["fps"])
    else:
        duration = float(video_metadata.get("duration", t_steps))
        timestamps = np.linspace(0.0, duration, t_steps, endpoint=False, dtype=np.float32)

    if len(timestamps) != t_steps:
        start_time = float(timestamps[0]) if len(timestamps) else 0.0
        end_time = float(timestamps[-1]) if len(timestamps) else float(t_steps - 1)
        timestamps = np.linspace(start_time, end_time, t_steps, dtype=np.float32)

    return TemporalFeatures(
        name="QwenViT",
        time_embeddings=time_embeddings,
        token_maps=token_maps,
        timestamps=timestamps.astype(np.float32),
        summary=summarize_features(time_embeddings),
        metadata={
            "video_grid_thw": [int(x) for x in video_grid_thw[0].tolist()],
            "merge_size": int(merge_size),
            "duration": float(video_metadata.get("duration", 0.0)),
        },
    )


def add_reference_frames_row(fig: Any, axes: Sequence[Any], sample: VideoSample) -> None:
    n_axes = len(axes)
    frame_indices = np.unique(np.linspace(0, len(sample.frames) - 1, n_axes, dtype=int))
    for ax, idx in zip(axes, frame_indices):
        ax.imshow(sample.frames[idx])
        ax.set_title(f"{sample.timestamps[idx]:.2f}s", fontsize=9)
        ax.axis("off")
    if len(axes) > 0:
        axes[0].set_ylabel("Original\nframes", fontsize=10)


def save_reference_frames(output_dir: Path, sample: VideoSample) -> None:
    plt = get_matplotlib_pyplot()
    ensure_dir(output_dir)
    for idx, frame in enumerate(sample.frames):
        timestamp = float(sample.timestamps[idx]) if idx < len(sample.timestamps) else float(idx)
        frame_name = f"frame_{idx:03d}_{timestamp:07.3f}s.png"
        plt.imsave(output_dir / frame_name, to_uint8_frame(frame))


def project_token_maps_to_rgb(token_maps: np.ndarray) -> np.ndarray:
    flat = token_maps.reshape(-1, token_maps.shape[-1]).astype(np.float32)
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] == 1:
        proj = np.zeros((1, 3), dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        comp = min(3, vt.shape[0])
        proj = flat @ vt[:comp].T
        if comp < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - comp)))

    proj = proj.reshape(*token_maps.shape[:3], 3)
    lower = np.percentile(proj, 1.0, axis=(0, 1, 2), keepdims=True)
    upper = np.percentile(proj, 99.0, axis=(0, 1, 2), keepdims=True)
    scale = np.where(upper > lower, upper - lower, 1.0)
    proj = np.clip((proj - lower) / scale, 0.0, 1.0)
    return proj.astype(np.float32)


def resize_pca_map_to_frame(pca_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if pca_map.shape[0] == target_h and pca_map.shape[1] == target_w:
        return pca_map

    tensor = torch.from_numpy(pca_map).permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0].permute(1, 2, 0).numpy()


def resize_scalar_map_to_frame(change_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if change_map.shape[0] == target_h and change_map.shape[1] == target_w:
        return change_map

    tensor = torch.from_numpy(change_map).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def align_timestamps(reference_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if len(target_timestamps) == 0:
        return np.zeros(len(reference_timestamps), dtype=int)
    deltas = np.abs(reference_timestamps[:, None] - target_timestamps[None, :])
    return deltas.argmin(axis=1).astype(int)


def align_change_map_timestamps(reference_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if len(target_timestamps) < 2:
        return np.zeros(len(reference_timestamps), dtype=int)
    midpoints = 0.5 * (target_timestamps[:-1] + target_timestamps[1:])
    deltas = np.abs(reference_timestamps[:, None] - midpoints[None, :])
    return deltas.argmin(axis=1).astype(int)


def plot_frame_pca_heatmap_overview(
    output_path: Path,
    reference_sample: VideoSample,
    item: TemporalFeatures,
) -> None:
    plt = get_matplotlib_pyplot()
    cols = len(reference_sample.frames)
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 8.2), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(rows, 1)

    add_reference_frames_row(fig, axes[0], reference_sample)

    pca_maps = project_token_maps_to_rgb(item.token_maps)
    pca_indices = align_timestamps(reference_sample.timestamps, item.timestamps)
    change_maps = compute_spatial_change_maps(item.token_maps)
    change_indices = align_change_map_timestamps(reference_sample.timestamps, item.timestamps)
    vmax = float(change_maps.max()) if change_maps.size else 1.0
    last_im = None

    for col_idx in range(cols):
        target_frame = reference_sample.frames[col_idx]
        target_hw = (int(target_frame.shape[0]), int(target_frame.shape[1]))

        pca_token_idx = int(pca_indices[col_idx])
        pca_ax = axes[1, col_idx]
        resized_pca_map = resize_pca_map_to_frame(pca_maps[pca_token_idx], target_hw)
        pca_ax.imshow(resized_pca_map)
        pca_ax.set_xticks([])
        pca_ax.set_yticks([])
        pca_ax.set_xlabel(f"{item.timestamps[pca_token_idx]:.2f}s", fontsize=9)
        if col_idx == 0:
            pca_ax.set_ylabel(f"{item.name}\nPCA", fontsize=10)

        heatmap_ax = axes[2, col_idx]
        if len(change_maps) == 0:
            heatmap_ax.axis("off")
            continue
        change_idx = int(change_indices[col_idx])
        resized_change_map = resize_scalar_map_to_frame(change_maps[change_idx], target_hw)
        last_im = heatmap_ax.imshow(
            resized_change_map,
            cmap="viridis",
            vmin=0.0,
            vmax=max(vmax, 1e-6),
        )
        start_t = float(item.timestamps[change_idx])
        end_t = float(item.timestamps[min(change_idx + 1, len(item.timestamps) - 1)])
        heatmap_ax.set_xticks([])
        heatmap_ax.set_yticks([])
        heatmap_ax.set_xlabel(f"{start_t:.2f}s -> {end_t:.2f}s", fontsize=8)
        if col_idx == 0:
            heatmap_ax.set_ylabel(f"{item.name}\nHeatmap", fontsize=10)

    fig.suptitle(
        f"{item.name}: original frames, PCA token maps, and spatial change heatmaps\nVideo: {reference_sample.video_path.name}",
        fontsize=16,
    )
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[2].tolist(), fraction=0.02, pad=0.01)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_temporal_similarity(
    output_path: Path,
    features_list: Sequence[TemporalFeatures],
) -> None:
    plt = get_matplotlib_pyplot()
    rows = len(features_list)
    fig, axes = plt.subplots(rows, 1, figsize=(6.0, 5.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for ax, item in zip(axes, features_list):
        similarity = cosine_similarity_matrix(item.time_embeddings)
        im = ax.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="magma")
        ax.set_title(f"{item.name}: temporal similarity")
        ax.set_xlabel("time index")
        ax.set_ylabel("time index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Temporal self-similarity matrices", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_spatial_heatmaps(
    output_path: Path,
    features_list: Sequence[TemporalFeatures],
    max_heatmaps: int,
) -> None:
    plt = get_matplotlib_pyplot()
    cols = max_heatmaps
    rows = len(features_list)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    for row_idx, item in enumerate(features_list):
        change_maps = compute_spatial_change_maps(item.token_maps)
        if len(change_maps) == 0:
            for ax in axes[row_idx]:
                ax.axis("off")
            continue
        selected = np.unique(np.linspace(0, len(change_maps) - 1, min(cols, len(change_maps)), dtype=int))
        vmax = float(change_maps.max()) if change_maps.size else 1.0

        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx >= len(selected):
                ax.axis("off")
                continue
            change_idx = int(selected[col_idx])
            im = ax.imshow(change_maps[change_idx], cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-6))
            start_t = float(item.timestamps[change_idx])
            end_t = float(item.timestamps[min(change_idx + 1, len(item.timestamps) - 1)])
            ax.set_title(f"{start_t:.2f}s -> {end_t:.2f}s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(item.name, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Spatial token change heatmaps between adjacent time steps", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_features_npz(output_path: Path, features_list: Sequence[TemporalFeatures]) -> None:
    arrays: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for item in features_list:
        prefix = item.name.lower()
        arrays[f"{prefix}_time_embeddings"] = item.time_embeddings
        arrays[f"{prefix}_token_maps"] = item.token_maps
        arrays[f"{prefix}_timestamps"] = item.timestamps
        metadata[prefix] = {"summary": item.summary, "metadata": item.metadata}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))
    np.savez_compressed(output_path, **arrays)


def print_summary(features_list: Sequence[TemporalFeatures], output_dir: Path) -> None:
    print(f"Saved outputs to: {output_dir}")
    for item in features_list:
        print(
            f"[{item.name}] steps={int(item.summary['num_steps'])} "
            f"mean_adjacent_delta={item.summary['mean_adjacent_delta']:.4f} "
            f"max_adjacent_delta={item.summary['max_adjacent_delta']:.4f} "
            f"mean_drift={item.summary['mean_distance_from_first']:.4f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "visualize_tokenizer_temporal_dynamics.log"
    setup_logging(log_file)

    device = choose_device(args.device)
    video_path = resolve_video_path(args)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reference_sample = sample_video_uniform(video_path, args.display_frames)
    features_list: List[TemporalFeatures] = []

    if "jepa" in args.models:
        features_list.append(extract_jepa_features(args, device, video_path))
    if "dino" in args.models:
        features_list.append(extract_dino_features(args, device, video_path))
    if "qwenvit" in args.models:
        features_list.append(extract_qwenvit_features(args, device, video_path))

    if not features_list:
        raise ValueError("No models selected.")

    save_reference_frames(output_dir / "reference_frames", reference_sample)
    plot_temporal_similarity(output_dir / "temporal_similarity.png", features_list)
    for item in features_list:
        prefix = item.name.lower()
        plot_frame_pca_heatmap_overview(output_dir / f"{prefix}_frame_pca_heatmap.png", reference_sample, item)
    save_features_npz(output_dir / "temporal_features.npz", features_list)
    print_summary(features_list, output_dir)


if __name__ == "__main__":
    main()
