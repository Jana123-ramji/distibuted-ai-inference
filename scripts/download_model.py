"""Download the small ONNX model used by the CPU fallback inference backend.

Why ONNX + onnxruntime instead of PyTorch? The fallback backend must run real
inference on a normal student laptop with no GPU. `onnxruntime` is a 14 MB wheel
and the int8-quantised SmolLM2-135M-Instruct model is ~137 MB, versus ~250 MB
for a PyTorch install plus ~270 MB of fp32 weights. Same model, same real
inference, far less to download.

Usage:

    python scripts/download_model.py                 # default model
    python scripts/download_model.py --repo <hf-repo> --file onnx/model_int8.onnx

The files land in ./models/<name>/ which is git-ignored.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
# int8 quantisation keeps the model small and runs well on plain CPUs.
DEFAULT_ONNX_FILE = "onnx/model_int8.onnx"
# Small metadata files needed by the tokenizer and the decode loop.
SUPPORT_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def download(repo: str, onnx_file: str, target_root: Path) -> Path:
    target = target_root / repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    for name in SUPPORT_FILES:
        try:
            path = hf_hub_download(repo, name, local_dir=str(target))
            print(f"  ok  {name} ({os.path.getsize(path) / 1e6:.2f} MB)")
        except Exception as exc:  # optional files may not exist in every repo
            print(f"  --  {name} skipped ({type(exc).__name__})")

    print(f"  .. downloading {onnx_file} (this is the big one)")
    started = time.time()
    weights = hf_hub_download(repo, onnx_file, local_dir=str(target))
    size_mb = os.path.getsize(weights) / 1e6
    elapsed = time.time() - started
    print(f"  ok  {onnx_file} ({size_mb:.1f} MB in {elapsed:.0f}s)")
    return Path(weights)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("MODEL_ID", DEFAULT_REPO))
    parser.add_argument("--file", default=os.getenv("MODEL_ONNX_FILE", DEFAULT_ONNX_FILE))
    parser.add_argument("--dir", default=os.getenv("MODEL_DIR", "models"))
    args = parser.parse_args()

    print(f"[download] repo={args.repo} file={args.file} -> {args.dir}/")
    weights = download(args.repo, args.file, Path(args.dir))
    print(f"[download] done: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
