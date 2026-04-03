#!/usr/bin/env python3
"""Print parameter counts for configs under pretrain-vision-8b (no weights required)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--text-config",
        type=Path,
        default=_ROOT / "configs" / "text" / "config.json",
    )
    p.add_argument(
        "--vision-config",
        type=Path,
        default=_ROOT / "configs" / "vision" / "clip_vit_large_patch14_336_config.json",
    )
    p.add_argument("--vision", action="store_true", help="Also load CLIPVisionModel count.")
    args = p.parse_args()

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    text_cfg = LlamaConfig.from_json_file(str(args.text_config))
    with torch.device("meta"):
        text_m = LlamaForCausalLM(text_cfg)
    n_text = sum(p.numel() for p in text_m.parameters())
    print(f"Text (LlamaForCausalLM from {args.text_config}): {n_text:,} ({n_text / 1e9:.2f}B)")

    if args.vision:
        from transformers import CLIPVisionConfig, CLIPVisionModel

        vcfg = CLIPVisionConfig.from_json_file(str(args.vision_config))
        with torch.device("meta"):
            v_m = CLIPVisionModel(vcfg)
        n_v = sum(p.numel() for p in v_m.parameters())
        print(f"Vision (CLIPVisionModel from {args.vision_config}): {n_v:,} ({n_v / 1e6:.1f}M)")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Install: pip install torch transformers", file=sys.stderr)
        raise SystemExit(1) from e
