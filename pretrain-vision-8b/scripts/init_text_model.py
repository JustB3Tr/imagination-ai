#!/usr/bin/env python3
"""
Create a random-initialized LlamaForCausalLM from configs/text/config.json
and save a Hugging Face–style folder (for CPT from scratch or testing loaders).

Disk: ~8B bf16 is on the order of ~16GB in safetensors (plus overhead).

Usage:
  python init_text_model.py --out /path/to/out_dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs" / "text" / "config.json",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig.from_json_file(str(args.config))
    model = LlamaForCausalLM(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    print(f"Saved random-init LlamaForCausalLM to {args.out}", flush=True)
    print("Copy tokenizer files next to this folder before training.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Install: pip install torch transformers safetensors", file=sys.stderr)
        raise SystemExit(1) from e
