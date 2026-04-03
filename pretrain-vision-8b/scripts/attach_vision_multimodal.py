#!/usr/bin/env python3
"""
Attach a frozen CLIP vision tower + learned projector to a Llama-class causal LM and
process image+text JSONL data (LLaVA-style).

This script knows how to:
  - Load your **~3.2B Llama** (or any size) from a Hugging Face folder or hub id
    (`AutoModelForCausalLM`).
  - Load a **CLIP** vision encoder (default: ViT-L/14 @ 336, matching `configs/vision/`).
  - Build a small **2-layer MLP projector** from vision hidden size → LLM hidden size.
  - Read a **multimodal JSONL** dataset (local image paths + text), expand a single
    image placeholder into `image_seq_length` token ids, merge **projected patch
    embeddings** into `inputs_embeds`, and run the LM (dry-run or short projector-only
    training).

Dataset format (one JSON object per line):

  {"image": "relative/or/absolute/path.jpg", "text": "User: <image>\\n\\nWhat is this?"}

The substring `--image_token` (default `<image>`) must tokenize to **exactly one** token id
after you add it to the tokenizer (the script can `--add_image_token`).

Examples:

  # Sanity check: load models, run one forward from JSONL (no weight updates)
  python scripts/attach_vision_multimodal.py \\
    --llm_model /path/to/Meta-Llama-3.2-3B-Instruct \\
    --train_file ../data/train.multimodal.example.jsonl \\
    --image_root .. \\
    --add_image_token \\
    --dry_run

  # Train only the projector for a few steps (vision + LLM frozen)
  python scripts/attach_vision_multimodal.py \\
    --llm_model /path/to/Meta-Llama-3.2-3B-Instruct \\
    --train_file /path/to/train.jsonl \\
    --image_root /path/to/images \\
    --add_image_token \\
    --train_projector_steps 20 \\
    --output_dir ../outputs/projector_warmstart
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)
from transformers.activations import ACT2FN

_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLIP + projector + Llama multimodal JSONL pipeline (LLaVA-style)."
    )
    p.add_argument(
        "--llm_model",
        type=str,
        required=True,
        help="HF hub id or local path to Llama (e.g. Meta-Llama-3.2-3B / 3B-class checkpoint).",
    )
    p.add_argument(
        "--vision_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP vision checkpoint (ViT-L/14 @ 336 matches configs/vision/).",
    )
    p.add_argument(
        "--train_file",
        type=Path,
        required=True,
        help="JSONL with image paths + text (see module docstring).",
    )
    p.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help="If set, resolve relative image paths against this directory.",
    )
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument(
        "--image_token",
        type=str,
        default="<image>",
        help="Placeholder substring in text; must become a single tokenizer id.",
    )
    p.add_argument(
        "--add_image_token",
        action="store_true",
        help="Add --image_token as a special token and resize LM embeddings.",
    )
    p.add_argument(
        "--vision_feature_layer",
        type=int,
        default=-2,
        help="Which vision hidden layer to use (negative = from end). -2 matches llava_skeleton.json.",
    )
    p.add_argument(
        "--projector_act",
        type=str,
        default="gelu",
        help="Activation between projector Linear layers (e.g. gelu, silu).",
    )
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="Use bf16 on CUDA for LLM + vision.")
    p.add_argument("--freeze_vision", action="store_true", default=True)
    p.add_argument("--no_freeze_vision", action="store_false", dest="freeze_vision")
    p.add_argument("--freeze_llm", action="store_true", default=True)
    p.add_argument("--no_freeze_llm", action="store_false", dest="freeze_llm")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data, run one forward + backward, then exit (no checkpoint save).",
    )
    p.add_argument("--train_projector_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _vision_patch_count(cfg: Any) -> int:
    size = int(cfg.image_size)
    ps = int(cfg.patch_size)
    return (size // ps) ** 2


class MultimodalProjector(nn.Module):
    """LLaVA-style 2-layer MLP: vision_dim -> llm_dim -> llm_dim."""

    def __init__(
        self,
        vision_hidden: int,
        text_hidden: int,
        hidden_act: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=bias)
        self.act = ACT2FN[hidden_act]
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


@dataclass
class MultimodalBatch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class MultimodalJsonlDataset(Dataset):
    """Loads JSONL rows; reads images from disk on __getitem__."""

    def __init__(
        self,
        path: Path,
        image_root: Path | None,
        image_key: str,
        text_key: str,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.image_root = image_root
        self.image_key = image_key
        self.text_key = text_key

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, raw: str) -> Path:
        p = Path(raw)
        if not p.is_file() and self.image_root is not None:
            p = self.image_root / raw
        return p

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        img_path = self._resolve_image_path(str(row[self.image_key]))
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        text = str(row[self.text_key])
        return {"image": image, "text": text, "image_path": str(img_path)}


class MultimodalCollator:
    def __init__(
        self,
        tokenizer: Any,
        image_processor: CLIPImageProcessor,
        image_token_id: int,
        image_seq_length: int,
        max_length: int,
        image_token_str: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.max_length = max_length
        self.image_token_str = image_token_str

    def _expand_image_tokens(self, ids: list[int]) -> list[int]:
        out: list[int] = []
        for tid in ids:
            if tid == self.image_token_id:
                out.extend([self.image_token_id] * self.image_seq_length)
            else:
                out.append(tid)
        return out

    def __call__(self, batch: list[dict[str, Any]]) -> MultimodalBatch:
        pixel_values: list[torch.Tensor] = []
        all_ids: list[list[int]] = []
        for item in batch:
            enc = self.tokenizer(
                item["text"],
                add_special_tokens=False,
                truncation=False,
            )
            ids: list[int] = list(enc["input_ids"])
            if self.image_token_str not in item["text"]:
                raise ValueError(
                    f"Text must contain image placeholder {self.image_token_str!r}: {item['text'][:200]!r}..."
                )
            if self.image_token_id not in ids:
                raise ValueError(
                    f"Image token id {self.image_token_id} not in tokenized ids — "
                    "ensure --image_token is a single tokenizer id (use --add_image_token if needed)."
                )
            ids = self._expand_image_tokens(ids)
            ids = ids[: self.max_length]
            all_ids.append(ids)
            proc = self.image_processor(images=item["image"], return_tensors="pt")
            pixel_values.append(proc["pixel_values"])

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(x) for x in all_ids)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, ids in enumerate(all_ids):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(ids, dtype=torch.long)
            img_mask = torch.tensor(ids, dtype=torch.long) == self.image_token_id
            labels[i, :L][img_mask] = -100

        return MultimodalBatch(
            pixel_values=torch.cat(pixel_values, dim=0),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def vision_patch_embeddings(
    vision: CLIPVisionModel,
    pixel_values: torch.Tensor,
    feature_layer: int,
    patch_count: int,
) -> torch.Tensor:
    out = vision(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    idx = feature_layer if feature_layer >= 0 else len(hs) + feature_layer
    hidden = hs[idx]
    if hidden.shape[1] == patch_count + 1:
        hidden = hidden[:, 1:, :]
    elif hidden.shape[1] != patch_count:
        raise ValueError(
            f"Unexpected vision seq len {hidden.shape[1]} for patch_count={patch_count}"
        )
    return hidden


def merge_image_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    projected: torch.Tensor,
) -> torch.Tensor:
    """Replace every image-token row with matching projected patch vector (per batch row)."""
    out = inputs_embeds.clone()
    b, t, h = out.shape
    for i in range(b):
        mask = input_ids[i] == image_token_id
        n_img = int(mask.sum().item())
        if n_img != projected.shape[1]:
            raise ValueError(
                f"Batch {i}: {n_img} image tokens but projector seq={projected.shape[1]}"
            )
        out[i, mask] = projected[i].to(dtype=out.dtype)
    return out


def main() -> None:
    args = parse_args()
    if not args.train_file.is_file():
        raise SystemExit(f"--train_file not found: {args.train_file}")
    if args.train_projector_steps > 0 and args.output_dir is None:
        raise SystemExit("--output_dir is required when --train_projector_steps > 0")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_tok_piece = args.image_token
    if args.add_image_token:
        tid_probe = tokenizer.convert_tokens_to_ids(image_tok_piece)
        if tid_probe == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_tok_piece]})

    tok_ids = tokenizer.encode(image_tok_piece, add_special_tokens=False)
    if len(tok_ids) != 1:
        raise SystemExit(
            f"--image_token must encode to exactly 1 token id; got {len(tok_ids)}: {tok_ids!r}. "
            "Use --add_image_token or pick a single-token placeholder."
        )
    image_token_id = tok_ids[0]

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else None,
    )
    if args.add_image_token:
        llm.resize_token_embeddings(len(tokenizer))

    vision = CLIPVisionModel.from_pretrained(
        args.vision_model,
        torch_dtype=dtype if device.type == "cuda" else None,
    )
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    patch_count = _vision_patch_count(vision.config)
    llm_hidden = int(llm.config.hidden_size)
    vision_hidden = int(vision.config.hidden_size)

    projector = MultimodalProjector(
        vision_hidden=vision_hidden,
        text_hidden=llm_hidden,
        hidden_act=args.projector_act,
        bias=True,
    )

    if args.freeze_vision:
        vision.requires_grad_(False)
    if args.freeze_llm:
        llm.requires_grad_(False)
    projector.requires_grad_(True)

    vision.to(device=device, dtype=dtype)
    llm.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)
    projector.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)

    dataset = MultimodalJsonlDataset(
        args.train_file,
        args.image_root,
        args.image_column,
        args.text_column,
    )
    collator = MultimodalCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_id=image_token_id,
        image_seq_length=patch_count,
        max_length=args.max_length,
        image_token_str=args.image_token,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    if len(dataset) == 0:
        raise SystemExit("Dataset is empty (no JSONL rows).")

    opt = torch.optim.AdamW((p for p in projector.parameters() if p.requires_grad), lr=args.learning_rate)

    def forward_step(batch: MultimodalBatch) -> torch.Tensor:
        pv = batch.pixel_values.to(device=device, dtype=dtype)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        with torch.set_grad_enabled(not args.freeze_vision):
            img_hidden = vision_patch_embeddings(
                vision, pv, args.vision_feature_layer, patch_count
            )
        proj = projector(img_hidden)
        embed_layer = llm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        inputs_embeds = merge_image_embeddings(
            inputs_embeds, input_ids, image_token_id, proj
        )

        out = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return out.loss

    llm.train(not args.freeze_llm)
    vision.train(not args.freeze_vision)
    projector.train()

    if args.dry_run:
        batch = next(iter(loader))
        loss = forward_step(batch)
        print(
            f"[multimodal] rows={len(dataset)} vision={args.vision_model} "
            f"patches={patch_count} llm_hidden={llm_hidden} loss={float(loss.detach().cpu()):.4f}"
        )
        loss.backward()
        print("[multimodal] dry_run OK (single batch forward+backward)")
        return

    if args.train_projector_steps <= 0:
        batch = next(iter(loader))
        loss = forward_step(batch)
        print(
            f"[multimodal] rows={len(dataset)} vision={args.vision_model} "
            f"patches={patch_count} llm_hidden={llm_hidden} loss={float(loss.detach().cpu()):.4f}"
        )
        print("[multimodal] No training steps. Use --train_projector_steps N or --dry_run.")
        return

    step = 0
    while step < args.train_projector_steps:
        for batch in loader:
            if step >= args.train_projector_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = forward_step(batch)
            if step == 0:
                print(
                    f"[multimodal] rows={len(dataset)} vision={args.vision_model} "
                    f"patches={patch_count} llm_hidden={llm_hidden} "
                    f"loss={float(loss.detach().cpu()):.4f}"
                )
            loss.backward()
            opt.step()
            if step % 10 == 0 and step > 0:
                print(f"step {step} loss={float(loss.detach().cpu()):.4f}")
            step += 1

    if args.output_dir is not None and args.train_projector_steps > 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(projector.state_dict(), args.output_dir / "projector.pt")
        with (args.output_dir / "attach_vision_multimodal_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "llm_model": args.llm_model,
                    "vision_model": args.vision_model,
                    "image_token": args.image_token,
                    "image_token_id": image_token_id,
                    "patch_count": patch_count,
                    "vision_feature_layer": args.vision_feature_layer,
                    "llm_hidden": llm_hidden,
                    "vision_hidden": vision_hidden,
                },
                f,
                indent=2,
            )
        print(f"Wrote projector to {args.output_dir / 'projector.pt'}")


if __name__ == "__main__":
    main()
