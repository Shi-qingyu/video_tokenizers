#!/usr/bin/env python3
"""
Download public DINOv3 and V-JEPA checkpoints used by this repo.

Examples:
    python3 download_backbone_checkpoints.py --family dino --models dinov3_vitb16
    python3 download_backbone_checkpoints.py --family jepa --models vjepa2_vit_large
    python3 download_backbone_checkpoints.py --family all --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


DINO_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"
JEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"


DINO_CHECKPOINTS: Dict[str, str] = {
    "dinov3_vits16": f"{DINO_BASE_URL}/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": f"{DINO_BASE_URL}/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": f"{DINO_BASE_URL}/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": f"{DINO_BASE_URL}/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16_sat493m": f"{DINO_BASE_URL}/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "dinov3_vitl16plus": f"{DINO_BASE_URL}/dinov3_vitl16plus/dinov3_vitl16plus_pretrain_lvd1689m-46503df0.pth",
    "dinov3_vith16plus": f"{DINO_BASE_URL}/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vit7b16": f"{DINO_BASE_URL}/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "dinov3_vit7b16_sat493m": f"{DINO_BASE_URL}/dinov3_vit7b16/dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
}


JEPA_CHECKPOINTS: Dict[str, str] = {
    "vjepa2_vit_large": f"{JEPA_BASE_URL}/vitl.pt",
    "vjepa2_vit_huge": f"{JEPA_BASE_URL}/vith.pt",
    "vjepa2_vit_giant": f"{JEPA_BASE_URL}/vitg.pt",
    "vjepa2_vit_giant_384": f"{JEPA_BASE_URL}/vitg-384.pt",
    "vjepa2_ac_vit_giant": f"{JEPA_BASE_URL}/vjepa2-ac-vitg.pt",
    "vjepa2_1_vit_base_384": f"{JEPA_BASE_URL}/vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2_1_vit_large_384": f"{JEPA_BASE_URL}/vjepa2_1_vitl_dist_vitG_384.pt",
    "vjepa2_1_vit_giant_384": f"{JEPA_BASE_URL}/vjepa2_1_vitg_384.pt",
    "vjepa2_1_vit_gigantic_384": f"{JEPA_BASE_URL}/vjepa2_1_vitG_384.pt",
}


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
    parser.add_argument("--family", choices=["dino", "jepa", "all"], default="all")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to download. Default: all models in the selected family.",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--list", action="store_true", help="List available model names and exit.")
    return parser.parse_args()


def build_registry(family: str) -> Dict[str, str]:
    if family == "dino":
        return dict(DINO_CHECKPOINTS)
    if family == "jepa":
        return dict(JEPA_CHECKPOINTS)
    registry = dict(DINO_CHECKPOINTS)
    registry.update(JEPA_CHECKPOINTS)
    return registry


def list_models() -> None:
    print("DINO models:")
    for name in DINO_CHECKPOINTS:
        print(f"  {name}")
    print("\nJEPA models:")
    for name in JEPA_CHECKPOINTS:
        print(f"  {name}")


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("a", encoding="utf-8", buffering=1)
    stdout = TeeStream(sys.__stdout__, handle)
    stderr = TeeStream(sys.__stderr__, handle)
    sys.stdout = stdout
    sys.stderr = stderr
    atexit.register(handle.close)
    print(f"[logging] writing combined stdout/stderr to {log_file}")
    print(f"[logging] started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"[logging] command: {' '.join(sys.argv)}")


def resolve_requests(family: str, selected_models: List[str] | None) -> List[Tuple[str, str]]:
    registry = build_registry(family)
    if not selected_models:
        if family == "dino":
            selected_models = list(DINO_CHECKPOINTS.keys())
        elif family == "jepa":
            selected_models = list(JEPA_CHECKPOINTS.keys())
        else:
            selected_models = list(registry.keys())

    missing = [name for name in selected_models if name not in registry]
    if missing:
        known = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown model(s): {missing}\nKnown models: {known}")

    return [(name, registry[name]) for name in selected_models]


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, dst_path: Path, force: bool) -> None:
    if dst_path.exists() and not force:
        print(f"[skip] {dst_path} already exists")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else 0
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes > 0:
                pct = downloaded / total_bytes * 100.0
                print(
                    f"\r  {pct:6.2f}%  {format_size(downloaded)} / {format_size(total_bytes)}",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  downloaded {format_size(downloaded)}", end="", flush=True)
    print()
    tmp_path.replace(dst_path)
    print(f"[saved] {dst_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else output_dir / "download_backbone_checkpoints.log"
    setup_logging(log_file)
    if args.list:
        list_models()
        return

    try:
        requests = resolve_requests(args.family, args.models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    for model_name, url in requests:
        family_dir = "dino" if model_name in DINO_CHECKPOINTS else "jepa"
        filename = os.path.basename(url)
        dst_path = output_dir / family_dir / filename
        download_file(url, dst_path, force=args.force)


if __name__ == "__main__":
    main()
