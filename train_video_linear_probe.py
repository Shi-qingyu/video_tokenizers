#!/usr/bin/env python3
"""
Train a frozen-backbone linear probe for 7 binary video quality dimensions.

The script reads the RL-style video annotations, parses the assistant answer
into 7 Yes/No labels, freezes a selected visual tokenizer, precomputes one
pooled feature vector per video, trains a single linear layer for one epoch,
and evaluates accuracy on the eval split.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = Path(__file__).resolve().parent
DINOV3_ROOT = REPO_ROOT / "dinov3"
VJEPA2_ROOT = REPO_ROOT / "vjepa2"
QWEN_FINETUNE_ROOT = REPO_ROOT / "Qwen3-VL" / "qwen-vl-finetune"
LOCAL_FACEBOOK_ROOT = REPO_ROOT / "facebook"

for extra_path in (DINOV3_ROOT, VJEPA2_ROOT):
    extra_path_str = str(extra_path)
    if extra_path.exists() and extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)


LABEL_NAMES = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
]
LABEL_KEY_MAP = {name.lower(): idx for idx, name in enumerate(LABEL_NAMES)}
MODEL_TYPES = ("jepa", "dino", "qwenvit", "all")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _default_existing_dir(path: Path) -> Optional[str]:
    return str(path.resolve()) if path.exists() else None


DEFAULT_DINO_REF = _default_existing_dir(LOCAL_FACEBOOK_ROOT / "dinov3-vitl16-pretrain-lvd1689m")
DEFAULT_JEPA_REF = _default_existing_dir(LOCAL_FACEBOOK_ROOT / "vjepa2-vitl-fpc64-256")


@dataclass
class ProbeSample:
    video_path: Path
    labels: torch.Tensor
    raw_answer: str


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
    parser.add_argument("--model-type", type=str, default="all", choices=list(MODEL_TYPES))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--train-dataset-use", type=str, default="videoreward")
    parser.add_argument("--eval-dataset-use", type=str, default="videoreward_eval")
    parser.add_argument("--train-annotation-path", type=str, default=None)
    parser.add_argument("--train-data-root", type=str, default=None)
    parser.add_argument("--eval-annotation-path", type=str, default=None)
    parser.add_argument("--eval-data-root", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--feature-num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

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

    parser.add_argument("--jepa-arch", type=str, default="vit_large_rope")
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=DEFAULT_JEPA_REF,
        help="Path to a JEPA .pt checkpoint or a local HF V-JEPA model directory.",
    )
    parser.add_argument("--jepa-input-size", type=int, default=256)
    parser.add_argument("--jepa-short-side", type=int, default=292)
    parser.add_argument("--jepa-num-frames", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)

    parser.add_argument("--qwen-model-path", type=str, default=None)
    parser.add_argument("--qwen-force-frames", type=int, default=None)
    return parser.parse_args()


def choose_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_distributed_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(device_arg: Optional[str]) -> torch.device:
    if not is_distributed_run():
        return choose_device(device_arg)

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_ranked_log_file(log_file: Path) -> Path:
    world_size = get_world_size()
    rank = get_rank()
    if world_size <= 1:
        return log_file
    return log_file.with_name(f"{log_file.stem}.rank{rank}{log_file.suffix}")


def setup_logging(log_file: Path) -> None:
    log_file = resolve_ranked_log_file(log_file)
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
        data: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        return loaded
    raise ValueError(f"Unsupported annotation format in {path}")


def flatten_annotation_items(items: Sequence[Any], base_path: Path) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(flatten_annotation_items(item, base_path))
            continue
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        copied["data_path"] = str(base_path)
        flattened.append(copied)
    return flattened


def load_items_from_dataset_use(dataset_use: str) -> List[Dict[str, Any]]:
    from dataset import data_list

    dataset_names = [name.strip() for name in dataset_use.split(",") if name.strip()]
    configs = data_list(dataset_names)
    all_items: List[Dict[str, Any]] = []
    for config in configs:
        annotation_path = (REPO_ROOT / config["annotation_path"]).resolve()
        data_root = (REPO_ROOT / config["data_path"]).resolve()
        items = read_json_or_jsonl(annotation_path)
        all_items.extend(flatten_annotation_items(items, data_root))
    return all_items


def load_items_from_annotation(annotation_path: str, data_root: str) -> List[Dict[str, Any]]:
    anno_path = Path(annotation_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    items = read_json_or_jsonl(anno_path)
    return flatten_annotation_items(items, root)


def parse_answer_labels(answer_text: str) -> torch.Tensor:
    match = re.search(r"<answer>(.*?)</answer>", answer_text, flags=re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else answer_text
    labels = torch.full((len(LABEL_NAMES),), -1.0, dtype=torch.float32)

    # Matches patterns like "Video Quality: Yes."
    pattern = re.compile(r"([A-Za-z\- ]+?)\s*:\s*(Yes|No)\b", flags=re.IGNORECASE)
    for key, value in pattern.findall(content):
        normalized_key = " ".join(key.strip().lower().split())
        if normalized_key in LABEL_KEY_MAP:
            labels[LABEL_KEY_MAP[normalized_key]] = 1.0 if value.lower() == "yes" else 0.0

    if (labels < 0).any():
        missing = [LABEL_NAMES[idx] for idx in torch.where(labels < 0)[0].tolist()]
        raise ValueError(f"Failed to parse labels for dimensions: {missing}")
    return labels


def build_probe_samples(items: Sequence[Dict[str, Any]], max_samples: Optional[int] = None) -> List[ProbeSample]:
    samples: List[ProbeSample] = []
    for item in items:
        videos = item.get("videos") or item.get("video")
        if not videos:
            continue
        video_rel = videos[0] if isinstance(videos, list) else videos
        conversations = item.get("conversations", [])
        assistant_turns = [turn for turn in conversations if turn.get("from") == "gpt"]
        if not assistant_turns:
            continue

        answer_text = assistant_turns[-1]["value"]
        try:
            labels = parse_answer_labels(answer_text)
        except Exception:
            continue

        data_path = Path(item.get("data_path", "")).expanduser().resolve()
        video_path = (data_path / video_rel).resolve()
        if not video_path.exists():
            continue

        samples.append(ProbeSample(video_path=video_path, labels=labels, raw_answer=answer_text))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


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
    readers = (_read_video_with_decord, _read_video_with_torchvision, _read_video_with_imageio)
    errors: List[str] = []
    for reader in readers:
        try:
            frames, fps = reader(video_path)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"Expected [T, H, W, 3], got {frames.shape}")
            return frames, fps
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
    raise RuntimeError(f"Failed to read video {video_path}:\n" + "\n".join(errors))


def sample_video_uniform(video_path: Path, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    frames, fps = load_video(video_path)
    total_frames = int(frames.shape[0])
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    target_frames = max(1, min(num_frames, total_frames))
    frame_indices = np.unique(np.linspace(0, total_frames - 1, target_frames, dtype=int))
    sampled_frames = frames[frame_indices]
    if fps <= 0:
        fps = float(target_frames)
    timestamps = frame_indices.astype(np.float32) / max(fps, 1e-6)
    return sampled_frames, timestamps


def _sample_video_frames_worker(payload: Tuple[str, int]) -> np.ndarray:
    video_path_str, num_frames = payload
    frames, _ = sample_video_uniform(Path(video_path_str), num_frames)
    return frames


def load_sampled_frames_batch(
    video_paths: Sequence[Path],
    num_frames: int,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[np.ndarray]:
    if executor is None:
        return [sample_video_uniform(video_path, num_frames)[0] for video_path in video_paths]
    payloads = [(str(video_path), num_frames) for video_path in video_paths]
    return list(executor.map(_sample_video_frames_worker, payloads))


def pad_frames_to_length(frames: np.ndarray, target_frames: int) -> np.ndarray:
    if frames.shape[0] == target_frames:
        return frames
    if frames.shape[0] > target_frames:
        return frames[:target_frames]
    pad_count = target_frames - frames.shape[0]
    pad_frames = np.repeat(frames[-1:], pad_count, axis=0)
    return np.concatenate([frames, pad_frames], axis=0)


def get_torchvision_functional():
    try:
        from torchvision.transforms import functional as TF
    except ImportError as exc:
        raise ImportError("torchvision is required for video frame preprocessing.") from exc
    return TF


def is_local_dir_model_ref(model_ref: Optional[str]) -> bool:
    if not model_ref:
        return False
    path = Path(model_ref).expanduser()
    return path.exists() and path.is_dir()


def to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(to_uint8_frame(frame)).permute(2, 0, 1).float() / 255.0


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


def preprocess_image_frames(frames: np.ndarray, short_side: int, crop_size: int) -> torch.Tensor:
    processed = []
    for frame in frames:
        tensor = frame_to_tensor(frame)
        tensor = resize_short_side(tensor, short_side)
        tensor = center_crop_square(tensor, crop_size)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(tensor)
    return torch.stack(processed, dim=0)


def preprocess_video_clip(frames: np.ndarray, short_side: int, crop_size: int) -> torch.Tensor:
    return preprocess_image_frames(frames, short_side, crop_size).permute(1, 0, 2, 3)


class FeatureExtractor:
    feature_dim: int
    name: str

    def extract(self, video_path: Path) -> torch.Tensor:
        raise NotImplementedError

    def extract_batch(
        self,
        video_paths: Sequence[Path],
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[torch.Tensor]:
        return [self.extract(video_path) for video_path in video_paths]


class JEPAHFVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_vision_features(pixel_values)


class JEPALocalVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.model(clip)


class DINOHFVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]


class DINOLocalVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_features(images)
        return outputs["x_norm_patchtokens"]


class QwenVisualWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
        return _call_qwen_visual(self.model, pixel_values_videos, video_grid_thw)


def maybe_wrap_dataparallel(module: nn.Module, device: torch.device) -> nn.Module:
    if dist.is_available() and dist.is_initialized():
        return module
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[multi-gpu] enabling DataParallel across {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(module)
    return module


def _load_jepa_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    elif "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    elif "ema_encoder" in checkpoint:
        state_dict = checkpoint["ema_encoder"]
    else:
        state_dict = checkpoint

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


class JEPAFeatureExtractor(FeatureExtractor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.name = "jepa"
        self.use_hf_local_dir = is_local_dir_model_ref(args.jepa_checkpoint)

        if self.use_hf_local_dir:
            from transformers import AutoModel, AutoVideoProcessor

            model_ref = str(Path(args.jepa_checkpoint).expanduser().resolve())
            print(f"[JEPA] loading local HF model from {model_ref}")
            self.processor = AutoVideoProcessor.from_pretrained(model_ref)
            self.model = AutoModel.from_pretrained(model_ref).to(device).eval()
            self.feature_dim = int(getattr(self.model.config, "hidden_size", 0))
            self.forward_model = maybe_wrap_dataparallel(JEPAHFVisionWrapper(self.model).to(device).eval(), device)
        else:
            from src.models import vision_transformer as vit_encoder

            if not hasattr(vit_encoder, args.jepa_arch):
                raise ValueError(f"Unknown JEPA architecture: {args.jepa_arch}")

            model = getattr(vit_encoder, args.jepa_arch)(
                img_size=(args.jepa_input_size, args.jepa_input_size),
                num_frames=args.jepa_num_frames,
                tubelet_size=args.jepa_tubelet_size,
            )
            if args.jepa_checkpoint:
                state_dict = _load_jepa_state_dict(Path(args.jepa_checkpoint).expanduser().resolve())
                msg = model.load_state_dict(state_dict, strict=False)
                print(f"[JEPA] checkpoint load message: {msg}")
            else:
                print("[JEPA] no checkpoint provided, using current initialization")

            self.model = model.to(device).eval()
            self.feature_dim = int(self.model.embed_dim)
            self.forward_model = maybe_wrap_dataparallel(JEPALocalVisionWrapper(self.model).to(device).eval(), device)

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def extract(self, video_path: Path) -> torch.Tensor:
        if self.use_hf_local_dir:
            frames, _ = sample_video_uniform(video_path, self.args.jepa_num_frames)
            video = torch.from_numpy(frames).permute(0, 3, 1, 2)
            inputs = self.processor(video, return_tensors="pt")
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                raise KeyError("AutoVideoProcessor did not return pixel_values_videos or pixel_values")
            pixel_values = pixel_values.to(self.device)
            tokens = self.forward_model(pixel_values)
            if tokens.ndim == 3:
                tokens = tokens[0]
            pooled = tokens.mean(dim=0)
            if self.feature_dim <= 0:
                self.feature_dim = int(pooled.numel())
            return pooled.float().cpu()

        frames, _ = sample_video_uniform(video_path, self.args.jepa_num_frames)
        if frames.shape[0] < self.args.jepa_tubelet_size:
            pad_count = self.args.jepa_tubelet_size - frames.shape[0]
            pad_frames = np.repeat(frames[-1:], pad_count, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
        if frames.shape[0] % self.args.jepa_tubelet_size != 0:
            usable = (frames.shape[0] // self.args.jepa_tubelet_size) * self.args.jepa_tubelet_size
            frames = frames[:usable]
        clip = preprocess_video_clip(
            frames,
            short_side=self.args.jepa_short_side,
            crop_size=self.args.jepa_input_size,
        ).unsqueeze(0).to(self.device)
        tokens = self.forward_model(clip)[0]
        return tokens.mean(dim=0).float().cpu()

    @torch.inference_mode()
    def extract_batch(
        self,
        video_paths: Sequence[Path],
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[torch.Tensor]:
        if self.use_hf_local_dir:
            features: List[torch.Tensor] = []
            for video_path in video_paths:
                features.append(self.extract(video_path))
            return features

        frames_list = load_sampled_frames_batch(video_paths, self.args.jepa_num_frames, executor=executor)
        clips = []
        for frames in frames_list:
            if frames.shape[0] < self.args.jepa_tubelet_size:
                pad_count = self.args.jepa_tubelet_size - frames.shape[0]
                pad_frames = np.repeat(frames[-1:], pad_count, axis=0)
                frames = np.concatenate([frames, pad_frames], axis=0)
            frames = pad_frames_to_length(frames, self.args.jepa_num_frames)
            if frames.shape[0] % self.args.jepa_tubelet_size != 0:
                usable = (frames.shape[0] // self.args.jepa_tubelet_size) * self.args.jepa_tubelet_size
                frames = frames[:usable]
            clips.append(
                preprocess_video_clip(
                    frames,
                    short_side=self.args.jepa_short_side,
                    crop_size=self.args.jepa_input_size,
                )
            )

        batch = torch.stack(clips, dim=0).to(self.device)
        tokens = self.forward_model(batch)
        pooled = tokens.mean(dim=1).float().cpu()
        return [pooled[i] for i in range(pooled.shape[0])]


class DINOFeatureExtractor(FeatureExtractor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.name = "dino"
        self.use_hf_local_dir = is_local_dir_model_ref(args.dino_weights)

        if self.use_hf_local_dir:
            from transformers import AutoImageProcessor, AutoModel

            model_ref = str(Path(args.dino_weights).expanduser().resolve())
            print(f"[DINO] loading local HF model from {model_ref}")
            self.processor = AutoImageProcessor.from_pretrained(model_ref)
            self.model = AutoModel.from_pretrained(model_ref).to(device).eval()
            self.feature_dim = int(getattr(self.model.config, "hidden_size", 0))
            self.forward_model = maybe_wrap_dataparallel(DINOHFVisionWrapper(self.model).to(device).eval(), device)
        else:
            from dinov3.hub import backbones as dino_backbones

            if not hasattr(dino_backbones, args.dino_arch):
                raise ValueError(f"Unknown DINO architecture: {args.dino_arch}")

            model_kwargs: Dict[str, Any] = {"pretrained": args.dino_weights is None}
            if args.dino_weights is not None:
                model_kwargs["pretrained"] = True
                model_kwargs["weights"] = str(Path(args.dino_weights).expanduser().resolve())
            self.model = getattr(dino_backbones, args.dino_arch)(**model_kwargs).to(device).eval()
            self.feature_dim = int(self.model.embed_dim)
            self.forward_model = maybe_wrap_dataparallel(DINOLocalVisionWrapper(self.model).to(device).eval(), device)

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def extract(self, video_path: Path) -> torch.Tensor:
        frames, _ = sample_video_uniform(video_path, self.args.dino_num_frames)
        if self.use_hf_local_dir:
            inputs = self.processor(images=[to_uint8_frame(frame) for frame in frames], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            last_hidden = self.forward_model(inputs["pixel_values"])
            pixel_values = inputs["pixel_values"]
            patch_size = int(getattr(self.model.config, "patch_size", 16))
            h_tokens = int(pixel_values.shape[-2] // patch_size)
            w_tokens = int(pixel_values.shape[-1] // patch_size)
            n_patch_tokens = h_tokens * w_tokens
            prefix_tokens = max(int(last_hidden.shape[1] - n_patch_tokens), 0)
            patch_tokens = last_hidden[:, prefix_tokens:, :]
            frame_features = patch_tokens.mean(dim=1)
            pooled = frame_features.mean(dim=0)
            if self.feature_dim <= 0:
                self.feature_dim = int(pooled.numel())
            return pooled.float().cpu()

        images = preprocess_image_frames(
            frames,
            short_side=self.args.dino_short_side,
            crop_size=self.args.dino_input_size,
        ).to(self.device)
        patch_tokens = self.forward_model(images)
        frame_features = patch_tokens.mean(dim=1)
        return frame_features.mean(dim=0).float().cpu()

    @torch.inference_mode()
    def extract_batch(
        self,
        video_paths: Sequence[Path],
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[torch.Tensor]:
        frames_list = load_sampled_frames_batch(video_paths, self.args.dino_num_frames, executor=executor)

        if self.use_hf_local_dir:
            images = []
            for frames in frames_list:
                frames = pad_frames_to_length(frames, self.args.dino_num_frames)
                images.extend([to_uint8_frame(frame) for frame in frames])
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            last_hidden = self.forward_model(inputs["pixel_values"])
            pixel_values = inputs["pixel_values"]
            patch_size = int(getattr(self.model.config, "patch_size", 16))
            h_tokens = int(pixel_values.shape[-2] // patch_size)
            w_tokens = int(pixel_values.shape[-1] // patch_size)
            n_patch_tokens = h_tokens * w_tokens
            prefix_tokens = max(int(last_hidden.shape[1] - n_patch_tokens), 0)
            patch_tokens = last_hidden[:, prefix_tokens:, :]
            patch_tokens = patch_tokens.view(len(video_paths), self.args.dino_num_frames, n_patch_tokens, -1)
            frame_features = patch_tokens.mean(dim=2)
            pooled = frame_features.mean(dim=1).float().cpu()
            if self.feature_dim <= 0:
                self.feature_dim = int(pooled.shape[-1])
            return [pooled[i] for i in range(pooled.shape[0])]

        image_batches = []
        for frames in frames_list:
            frames = pad_frames_to_length(frames, self.args.dino_num_frames)
            image_batches.append(
                preprocess_image_frames(
                    frames,
                    short_side=self.args.dino_short_side,
                    crop_size=self.args.dino_input_size,
                )
            )
        images = torch.stack(image_batches, dim=0).to(self.device)
        batch_size, num_frames = images.shape[:2]
        flat_images = images.view(batch_size * num_frames, *images.shape[2:])
        patch_tokens = self.forward_model(flat_images)
        patch_tokens = patch_tokens.view(batch_size, num_frames, patch_tokens.shape[1], patch_tokens.shape[2])
        frame_features = patch_tokens.mean(dim=2)
        pooled = frame_features.mean(dim=1).float().cpu()
        return [pooled[i] for i in range(pooled.shape[0])]


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
        raise ImportError("transformers is required for Qwen-ViT probing") from exc

    lower_name = model_path.lower()
    model_name = Path(model_path.rstrip("/")).name.lower()
    if "qwen3" in lower_name and "a" in model_name:
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
    attempts = [
        lambda: model.visual(pixel_values_videos, grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values_videos, video_grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values=pixel_values_videos, grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values=pixel_values_videos, video_grid_thw=video_grid_thw),
        lambda: model.visual(pixel_values_videos),
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            out = attempt()
            break
        except TypeError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Unable to call Qwen visual encoder: {last_error}")

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(out, dict):
        for key in ("last_hidden_state", "image_embeds", "video_embeds", "hidden_states"):
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and value and isinstance(value[-1], torch.Tensor):
                return value[-1]
    raise TypeError(f"Unsupported Qwen visual output type: {type(out)}")


class QwenViTFeatureExtractor(FeatureExtractor):
    def __init__(self, args: argparse.Namespace, device: torch.device):
        if not args.qwen_model_path:
            raise ValueError("--qwen-model-path is required for qwenvit")
        if QWEN_FINETUNE_ROOT.exists():
            qwen_path = str(QWEN_FINETUNE_ROOT)
            if qwen_path not in sys.path:
                sys.path.insert(0, qwen_path)

        self.args = args
        self.device = device
        self.name = "qwenvit"
        self.model, self.processor = _load_qwen_model_and_processor(args.qwen_model_path, device)
        self.forward_model = maybe_wrap_dataparallel(QwenVisualWrapper(self.model).to(device).eval(), device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = 0

        if args.qwen_force_frames is not None and hasattr(self.processor, "video_processor"):
            if hasattr(self.processor.video_processor, "min_frames"):
                self.processor.video_processor.min_frames = args.qwen_force_frames
            if hasattr(self.processor.video_processor, "max_frames"):
                self.processor.video_processor.max_frames = args.qwen_force_frames

    @torch.inference_mode()
    def extract(self, video_path: Path) -> torch.Tensor:
        vp_output = self.processor.video_processor(
            videos=[str(video_path)],
            return_tensors="pt",
            return_metadata=True,
        )
        pixel_values_videos = vp_output.pixel_values_videos.to(self.device)
        video_grid_thw = vp_output.video_grid_thw.to(self.device)
        tokens = self.forward_model(pixel_values_videos, video_grid_thw)
        if tokens.ndim == 2:
            pooled = tokens.mean(dim=0)
        elif tokens.ndim == 3:
            pooled = tokens[0].mean(dim=0)
        else:
            raise ValueError(f"Unexpected Qwen token shape: {tuple(tokens.shape)}")
        pooled = pooled.float().cpu()
        if self.feature_dim <= 0:
            self.feature_dim = int(pooled.numel())
        return pooled

    @torch.inference_mode()
    def extract_batch(
        self,
        video_paths: Sequence[Path],
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[torch.Tensor]:
        vp_output = self.processor.video_processor(
            videos=[str(video_path) for video_path in video_paths],
            return_tensors="pt",
            return_metadata=True,
        )
        pixel_values_videos = vp_output.pixel_values_videos.to(self.device)
        video_grid_thw = vp_output.video_grid_thw.to(self.device)
        tokens = self.forward_model(pixel_values_videos, video_grid_thw)

        if tokens.ndim == 3 and tokens.shape[0] == len(video_paths):
            pooled = tokens.mean(dim=1).float().cpu()
            if self.feature_dim <= 0:
                self.feature_dim = int(pooled.shape[-1])
            return [pooled[i] for i in range(pooled.shape[0])]

        if tokens.ndim != 2:
            raise ValueError(f"Unexpected batched Qwen token shape: {tuple(tokens.shape)}")

        merge_size = int(getattr(getattr(self.processor, "image_processor", self.processor), "merge_size", 2))
        merge_area = max(merge_size * merge_size, 1)
        split_sizes = []
        for t, h, w in video_grid_thw.detach().cpu().tolist():
            token_count = max(int((int(t) * int(h) * int(w)) // merge_area), 1)
            split_sizes.append(token_count)

        chunks = list(torch.split(tokens, split_sizes, dim=0))
        pooled_features = [chunk.mean(dim=0).float().cpu() for chunk in chunks]
        if self.feature_dim <= 0 and pooled_features:
            self.feature_dim = int(pooled_features[0].numel())
        return pooled_features


def build_feature_extractor(model_type: str, args: argparse.Namespace, device: torch.device) -> FeatureExtractor:
    if model_type == "jepa":
        return JEPAFeatureExtractor(args, device)
    if model_type == "dino":
        return DINOFeatureExtractor(args, device)
    if model_type == "qwenvit":
        return QwenViTFeatureExtractor(args, device)
    raise ValueError(f"Unsupported model type: {model_type}")


def precompute_features(
    extractor: FeatureExtractor,
    samples: Sequence[ProbeSample],
    feature_batch_size: int,
    feature_num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Path]]:
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    kept_paths: List[Path] = []
    kept_indices: List[int] = []

    rank = get_rank()
    world_size = get_world_size()
    indexed_samples = [(idx, sample) for idx, sample in enumerate(samples) if idx % world_size == rank]
    total = len(indexed_samples)
    max_workers = max(int(feature_num_workers), 0)
    executor: Optional[ProcessPoolExecutor] = None
    if max_workers > 1:
        executor = ProcessPoolExecutor(max_workers=max_workers)

    try:
        for start_idx in range(0, total, max(feature_batch_size, 1)):
            batch_with_indices = list(indexed_samples[start_idx : start_idx + max(feature_batch_size, 1)])
            batch_samples = [sample for _, sample in batch_with_indices]
            batch_paths = [sample.video_path for sample in batch_samples]
            try:
                batch_features = extractor.extract_batch(batch_paths, executor=executor)
            except Exception as exc:
                print(f"[{extractor.name}] rank={rank} batch failed at start={start_idx}: {exc}")
                for global_idx, sample in batch_with_indices:
                    try:
                        feature = extractor.extract(sample.video_path)
                    except Exception as inner_exc:
                        print(f"[{extractor.name}] skip {sample.video_path}: {inner_exc}")
                        continue
                    features.append(feature)
                    labels.append(sample.labels)
                    kept_paths.append(sample.video_path)
                    kept_indices.append(global_idx)
                print(f"[{extractor.name}] rank={rank} extracted up to {min(start_idx + len(batch_samples), total)}/{total}")
                continue

            if len(batch_features) != len(batch_samples):
                raise RuntimeError(
                    f"[{extractor.name}] batch feature count mismatch: "
                    f"{len(batch_features)} features for {len(batch_samples)} samples"
                )

            for (global_idx, sample), feature in zip(batch_with_indices, batch_features):
                features.append(feature)
                labels.append(sample.labels)
                kept_paths.append(sample.video_path)
                kept_indices.append(global_idx)

            print(f"[{extractor.name}] rank={rank} extracted {min(start_idx + len(batch_samples), total)}/{total}")
    finally:
        if executor is not None:
            executor.shutdown()

    if not features:
        raise RuntimeError(f"No usable features extracted for {extractor.name}")

    local_items = []
    for global_idx, feature, label, path in zip(kept_indices, features, labels, kept_paths):
        local_items.append((global_idx, feature.numpy(), label.numpy(), str(path)))

    if dist.is_available() and dist.is_initialized():
        gathered_items: List[List[Tuple[int, np.ndarray, np.ndarray, str]]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered_items, local_items)
        merged_items = [item for sublist in gathered_items for item in sublist]
        merged_items.sort(key=lambda x: x[0])
        features = [torch.from_numpy(item[1]).float() for item in merged_items]
        labels = [torch.from_numpy(item[2]).float() for item in merged_items]
        kept_paths = [Path(item[3]) for item in merged_items]

    return torch.stack(features, dim=0), torch.stack(labels, dim=0), kept_paths


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    feature_dim = int(train_features.shape[1])
    probe = nn.Linear(feature_dim, len(LABEL_NAMES)).to(device)
    if dist.is_available() and dist.is_initialized():
        probe = DDP(
            probe,
            device_ids=[device.index] if device.type == "cuda" and device.index is not None else None,
            output_device=device.index if device.type == "cuda" and device.index is not None else None,
        )
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        probe = nn.DataParallel(probe)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(train_features, train_labels)
    train_sampler = None
    shuffle = True
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    probe.train()
    for epoch_idx in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        running_loss = 0.0
        running_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(batch_features.size(0))
            running_count += int(batch_features.size(0))

        epoch_loss = running_loss / max(running_count, 1)
        print(f"epoch={epoch_idx + 1} train_loss={epoch_loss:.6f}")

    metrics = evaluate_linear_probe(probe, eval_features, eval_labels, device)
    return probe, metrics


@torch.inference_mode()
def evaluate_linear_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    probe.eval()
    rank = get_rank()
    world_size = get_world_size()
    local_features = features[rank::world_size].to(device)
    local_labels = labels[rank::world_size].to(device)
    logits = probe(local_features)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct_per_dim = (preds == local_labels).float().sum(dim=0)
    total_per_dim = torch.full_like(correct_per_dim, float(local_labels.shape[0]))
    exact_correct = (preds == local_labels).all(dim=1).float().sum()
    total_samples = torch.tensor(float(local_labels.shape[0]), device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct_per_dim, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_per_dim, op=dist.ReduceOp.SUM)
        dist.all_reduce(exact_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    per_dim_accuracy = (correct_per_dim / total_per_dim.clamp(min=1.0)).cpu()
    exact_match = (exact_correct / total_samples.clamp(min=1.0)).cpu()
    results = {
        "overall_mean_accuracy": float(per_dim_accuracy.mean().item()),
        "exact_match_accuracy": float(exact_match.item()),
        "per_dimension_accuracy": {
            label_name: float(per_dim_accuracy[idx].item()) for idx, label_name in enumerate(LABEL_NAMES)
        },
    }
    return results


def save_results(
    output_dir: Path,
    model_type: str,
    metrics: Dict[str, Any],
    train_count: int,
    eval_count: int,
) -> None:
    payload = {
        "model_type": model_type,
        "train_samples": train_count,
        "eval_samples": eval_count,
        **metrics,
    }
    with (output_dir / f"{model_type}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_metrics(model_type: str, metrics: Dict[str, Any]) -> None:
    print(f"\n[{model_type}] eval results")
    print(f"overall_mean_accuracy={metrics['overall_mean_accuracy']:.4f}")
    print(f"exact_match_accuracy={metrics['exact_match_accuracy']:.4f}")
    for name, value in metrics["per_dimension_accuracy"].items():
        print(f"{name}: {value:.4f}")


def load_split(
    dataset_use: str,
    annotation_path: Optional[str],
    data_root: Optional[str],
    max_samples: Optional[int],
) -> List[ProbeSample]:
    if annotation_path:
        if not data_root:
            raise ValueError("When annotation_path is set, data_root must also be provided.")
        items = load_items_from_annotation(annotation_path, data_root)
    else:
        items = load_items_from_dataset_use(dataset_use)
    return build_probe_samples(items, max_samples=max_samples)


def run_single_model(model_type: str, args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    print(f"\n=== Running linear probe for {model_type} (rank={get_rank()}/{get_world_size()}) ===")
    extractor = build_feature_extractor(model_type, args, device)

    train_samples = load_split(
        dataset_use=args.train_dataset_use,
        annotation_path=args.train_annotation_path,
        data_root=args.train_data_root,
        max_samples=args.max_train_samples,
    )
    eval_samples = load_split(
        dataset_use=args.eval_dataset_use,
        annotation_path=args.eval_annotation_path,
        data_root=args.eval_data_root,
        max_samples=args.max_eval_samples,
    )
    print(f"[{model_type}] train samples={len(train_samples)} eval samples={len(eval_samples)}")

    train_features, train_labels, _ = precompute_features(
        extractor,
        train_samples,
        feature_batch_size=args.feature_batch_size,
        feature_num_workers=args.feature_num_workers,
    )
    eval_features, eval_labels, _ = precompute_features(
        extractor,
        eval_samples,
        feature_batch_size=args.feature_batch_size,
        feature_num_workers=args.feature_num_workers,
    )

    probe, metrics = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        eval_features=eval_features,
        eval_labels=eval_labels,
        args=args,
        device=device,
    )

    if is_main_process():
        state_dict = probe.module.state_dict() if isinstance(probe, (nn.DataParallel, DDP)) else probe.state_dict()
        torch.save(state_dict, output_dir / f"{model_type}_linear_probe.pt")
        save_results(output_dir, model_type, metrics, len(train_features), len(eval_features))
        print_metrics(model_type, metrics)


def main() -> None:
    args = parse_args()
    device = setup_distributed(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "train_video_linear_probe.log"
    setup_logging(log_file)
    set_seed(args.seed)

    if is_distributed_run():
        print(
            f"[ddp] initialized rank={get_rank()} world_size={get_world_size()} "
            f"local_rank={os.environ.get('LOCAL_RANK')} device={device}"
        )
    else:
        print(f"[device] using {device}")

    if args.model_type == "all":
        model_types = ["jepa", "dino", "qwenvit"]
    else:
        model_types = [args.model_type]

    try:
        for model_type in model_types:
            run_single_model(model_type, args, device, output_dir)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
