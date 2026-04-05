#!/usr/bin/env python3
"""
Colab-friendly pipeline: download up to ~250k real image–caption rows (dataset permitting),
train the CLIP projector for (rows × epochs) steps by default (attach_vision_multimodal.py),
then copy outputs to vision_cpt_out_real next to this repo.

Expected layout on Drive:
  imagination-v1.1.0/
    pretrain-vision-8b/          <- this script lives under scripts/
    vision_cpt_out_real/          <- default copy destination (created/overwritten)

One-liner (Colab, after cd to repo or using absolute paths):

  python /content/drive/MyDrive/imagination-v1.1.0/pretrain-vision-8b/scripts/colab_prepare_and_train_projector.py \\
    --llm_model /content/drive/MyDrive/imagination-v1.1.0

Requires: torch, transformers, datasets, pillow, accelerate (optional; not used by default).

Defaults target ~3 full passes over up to 250k rows (batch_size=1 ⇒ steps ≈ rows × epochs).
Food-101 train is ~75k rows, so you get 3× that many steps unless you point `--dataset` at a larger HF set.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download image–text pairs, train projector (~epochs × rows steps by default), copy to vision_cpt_out_real."
    )
    p.add_argument(
        "--llm_model",
        type=str,
        default=os.environ.get("IMAGINATION_LLM", "").strip() or None,
        help="HF folder or hub id for the Llama-class LM (or set IMAGINATION_LLM).",
    )
    p.add_argument(
        "--repo_root",
        type=Path,
        default=_repo_root(),
        help="Path to pretrain-vision-8b directory.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=250_000,
        help="Rows to download (capped by dataset size). Food-101 train ≈75k.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Used to set train_projector_steps = JSONL line count × epochs when --train_projector_steps omitted.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="ethz/food101",
        help="HF dataset id with `image` + `label` columns (default: Food-101).",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path(os.environ.get("IMAGINATION_VISION_DATA", "/content/data/imagination_vision_10k")),
        help="Local dir for images + train.jsonl (Colab: fast local disk).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("IMAGINATION_VISION_OUT", "/content/vision_cpt_out")),
        help="Local dir for projector.pt + meta JSON before copy to Drive.",
    )
    p.add_argument(
        "--copy_dir",
        type=Path,
        default=None,
        help="Destination for final bundle (default: <repo parent>/vision_cpt_out_real).",
    )
    p.add_argument(
        "--train_projector_steps",
        type=int,
        default=None,
        help="Optimizer steps (batch_size=1). Default: (JSONL rows) × --epochs. Override for quick tests.",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument(
        "--vision_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Reuse existing data_dir/train.jsonl if present (must have >= num_samples lines).",
    )
    return p.parse_args()


def _default_copy_dir(repo_root: Path) -> Path:
    return repo_root.parent / "vision_cpt_out_real"


def download_food101_jsonl(data_dir: Path, num_samples: int, dataset_id: str, seed: int) -> Path:
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / "train.jsonl"

    print(f"[colab-vision] Loading dataset {dataset_id!r} (streaming=False)...", flush=True)
    ds = load_dataset(dataset_id, split="train", trust_remote_code=True)
    n = min(int(num_samples), len(ds))
    if n < int(num_samples):
        print(f"[colab-vision] Dataset has only {len(ds)} rows; using n={n}.", flush=True)
    ds = ds.shuffle(seed=seed).select(range(n))

    image_key, label_key = "image", "label"
    if image_key not in ds.column_names:
        raise SystemExit(f"Dataset missing {image_key!r} column: {ds.column_names}")
    if label_key not in ds.column_names:
        raise SystemExit(f"Dataset missing {label_key!r} column: {ds.column_names}")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            img = row[image_key]
            label = str(row[label_key]).replace("_", " ")
            fname = f"{i:06d}.jpg"
            fpath = data_dir / fname
            img.convert("RGB").save(fpath, format="JPEG", quality=92)
            # attach_vision_multimodal requires the literal image token in text.
            text = (
                "User: <image>\n\nDescribe this photograph in one short phrase.\n"
                f"Assistant: {label}."
            )
            f.write(json.dumps({"image": fname, "text": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"[colab-vision] wrote {i + 1}/{n} rows", flush=True)

    print(f"[colab-vision] Wrote {n} samples to {data_dir} and {jsonl_path}", flush=True)
    return jsonl_path, n


def verify_jsonl(data_dir: Path) -> tuple[Path, int]:
    jsonl_path = data_dir / "train.jsonl"
    if not jsonl_path.is_file():
        raise SystemExit(f"Missing {jsonl_path}; run without --skip_download.")
    n = sum(1 for _ in jsonl_path.open(encoding="utf-8") if _.strip())
    if n < 1:
        raise SystemExit(f"{jsonl_path} is empty.")
    print(f"[colab-vision] Reusing {n} lines from {jsonl_path}", flush=True)
    return jsonl_path, n


def main() -> None:
    args = parse_args()
    if not args.llm_model:
        raise SystemExit("Pass --llm_model /path/to/hf_llm or set IMAGINATION_LLM.")

    repo_root = args.repo_root.resolve()
    attach_script = repo_root / "scripts" / "attach_vision_multimodal.py"
    if not attach_script.is_file():
        raise SystemExit(f"Missing {attach_script}")

    copy_dir = args.copy_dir.resolve() if args.copy_dir else _default_copy_dir(repo_root)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.skip_download:
        jsonl_path, n_lines = verify_jsonl(data_dir)
        if n_lines < args.num_samples:
            print(
                f"[colab-vision] Note: JSONL has {n_lines} rows (requested up to {args.num_samples}).",
                flush=True,
            )
    else:
        if args.dataset != "ethz/food101":
            print(
                "[colab-vision] Warning: only ethz/food101 is validated for auto column layout; "
                "other datasets may need a small edit to this script.",
                flush=True,
            )
        jsonl_path, n_lines = download_food101_jsonl(data_dir, args.num_samples, args.dataset, args.seed)

    if args.train_projector_steps is not None:
        train_steps = int(args.train_projector_steps)
    else:
        train_steps = int(n_lines) * int(args.epochs)
    print(
        f"[colab-vision] train_projector_steps={train_steps} "
        f"({n_lines} JSONL rows × {args.epochs} epoch(s); batch_size={args.batch_size}).",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(attach_script),
        "--llm_model",
        str(args.llm_model),
        "--train_file",
        str(jsonl_path),
        "--image_root",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--bf16",
        "--add_image_token",
        "--train_projector_steps",
        str(train_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--vision_model",
        str(args.vision_model),
        "--seed",
        str(args.seed),
    ]

    print("[colab-vision] Running:\n  " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
    if rc != 0:
        raise SystemExit(f"attach_vision_multimodal.py exited with {rc}")

    if not (output_dir / "projector.pt").is_file():
        raise SystemExit(f"No projector.pt under {output_dir}; training may have failed.")

    copy_dir.parent.mkdir(parents=True, exist_ok=True)
    if copy_dir.exists():
        shutil.rmtree(copy_dir)
    shutil.copytree(output_dir, copy_dir)
    print(f"[colab-vision] Copied bundle to {copy_dir}", flush=True)


if __name__ == "__main__":
    main()
