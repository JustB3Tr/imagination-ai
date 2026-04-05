#!/usr/bin/env python3
"""
Train a single linear projection CLIP patch features -> LLM hidden size (LLaVA-style prefix),
with frozen CLIP ViT-L/14-336 and frozen Llama-class LM (e.g. Imagination 1.3 ~3B).

Verified from repo root config.json + vision_cpt_out_real/attach_vision_multimodal_meta.json:
  - Llama hidden_size: 3072 (your Imagination checkpoint in repo root)
  - CLIP ViT-L/14-336 vision hidden: 1024, patch_count: 576
  So nn.Linear(1024, 3072) applied to each patch is consistent.

About vision_cpt_out_real / "10k steps":
  - The JSON meta in this repo does NOT log step count, dataset, or LR (only architecture IDs).
  - The Colab helper colab_prepare_and_train_projector.py defaults to train_projector_steps=3000,
    not 10000. An older notebook used train_projector_steps=1 (one optimizer step).
  - attach_vision_multimodal.py trains a 2-layer MLP projector, not a single Linear; projector.pt
    is not present in git (only meta) — weights may exist only on your Drive machine.

After training, this script can export projector.pt in the same state_dict layout as
imagination_runtime.clip_projector_vlm.MultimodalProjector: trained weights go into linear_1;
linear_2 is identity. Imagination v1.3 still applies GELU between the two layers, so runtime
behavior is linear_2(gelu(linear_1(x))) ≈ gelu(Wx+b), not identical to the linear-only training
forward (Wx+b). For a closer match, fine-tune with attach_vision_multimodal.py (full MLP).

Dependencies: torch, transformers, datasets, pillow, accelerate (optional for future).

Example:
  python scripts/train_linear_clip_llm_projection.py \\
    --llm_model /path/to/imagination-v1.1.0 \\
    --output_dir ./outputs/linear_proj_run1
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):  # type: ignore
        return x
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
)
from transformers.activations import ACT2FN


@dataclass
class TrainConfig:
    llm_model: str
    vision_model: str
    dataset_id: str
    dataset_config: Optional[str]
    image_column: str
    caption_column: str
    max_train_samples: int
    val_size: int
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_length: int
    vision_feature_layer: int
    bf16: bool
    seed: int
    log_every: int
    save_every: int
    image_token: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen CLIP + frozen Llama, train Linear(1024->H) only.")
    p.add_argument("--llm_model", type=str, required=True, help="Local path or hub id (Imagination HF folder).")
    p.add_argument(
        "--vision_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP vision tower (default matches vision_cpt_out_real meta).",
    )
    p.add_argument(
        "--dataset_id",
        type=str,
        default="ethz/food101",
        help="HF dataset with image + text columns. Default food101 (~75k train). "
        "For ~250k+ synthetic image–prompt pairs try poloclub/diffusiondb with a named config "
        "(see dataset card; often trust_remote_code + large download).",
    )
    p.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Second arg to load_dataset, e.g. DiffusionDB subset name (empty = single-file dataset).",
    )
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--caption_column", type=str, default="label", help="food101: label; diffusiondb: prompt")
    p.add_argument("--max_train_samples", type=int, default=250_000)
    p.add_argument("--val_size", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1, help="Raise to 2 on L4 if memory allows.")
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Effective batch = per_device_batch_size * this (L4 24GB: start 1x8).",
    )
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_length", type=int, default=256, help="Caption tokens (keep moderate for VRAM).")
    p.add_argument("--vision_feature_layer", type=int, default=-2, help="CLIP hidden layer index (negative from end).")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 on CUDA (recommended on L4).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=500, help="Optimizer steps between loss logs.")
    p.add_argument("--save_every", type=int, default=5_000, help="Optimizer steps between checkpoints.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--image_token", type=str, default="<image>", help="Must become a single tokenizer id.")
    p.add_argument("--add_image_token", action="store_true", help="Add image_token to tokenizer and resize LM embeddings.")
    p.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help="Use 0 on Windows; 2–4 on Linux/Colab if stable.",
    )
    p.add_argument(
        "--export_mlp_projector",
        action="store_true",
        default=True,
        help="Write projector.pt (2-layer MLP layout) for Imagination v1.3 clip_projector_vlm load.",
    )
    p.add_argument(
        "--no_export_mlp_projector",
        action="store_false",
        dest="export_mlp_projector",
    )
    p.add_argument(
        "--projector_act",
        type=str,
        default="gelu",
        help="Activation between MLP layers in exported projector (must match inference; default gelu).",
    )
    return p.parse_args()


def vision_patch_count(cfg: Any) -> int:
    size = int(cfg.image_size)
    ps = int(cfg.patch_size)
    return (size // ps) ** 2


def vision_patch_embeddings(
    vision: CLIPVisionModel,
    pixel_values: torch.Tensor,
    feature_layer: int,
    patch_count: int,
) -> torch.Tensor:
    out = vision(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    idx = feature_layer if feature_layer >= 0 else len(hs) + feature_layer
    hidden = hs[idx]
    if hidden.shape[1] == patch_count + 1:
        hidden = hidden[:, 1:, :]
    elif hidden.shape[1] != patch_count:
        raise ValueError(f"Vision seq len {hidden.shape[1]} != patch_count {patch_count}")
    return hidden


def merge_prefix_embeddings(
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    projected_patches: torch.Tensor,
) -> torch.Tensor:
    """Replace every position where input_ids == image_token_id with matching projected patch."""
    out = token_embeds.clone()
    b, t, h = out.shape
    for i in range(b):
        mask = input_ids[i] == image_token_id
        n = int(mask.sum().item())
        if n != projected_patches.shape[1]:
            raise ValueError(f"Batch {i}: {n} image tokens vs {projected_patches.shape[1]} patches")
        out[i, mask] = projected_patches[i].to(dtype=out.dtype)
    return out


class HfImageCaptionDataset(Dataset):
    def __init__(
        self,
        hf_split: Any,
        image_key: str,
        caption_key: str,
        max_samples: int,
    ):
        self.ds = hf_split
        self.image_key = image_key
        self.caption_key = caption_key
        self.n = min(max_samples, len(self.ds))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        img = row[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB") if hasattr(img, "read") else Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        cap = (row[self.caption_key] or "").strip()
        if not cap:
            cap = " "
        return {"image": img, "caption": cap}


def collate_batch(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    image_processor: CLIPImageProcessor,
    image_token: str,
    image_token_id: int,
    patch_count: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    texts = []
    for item in batch:
        texts.append(f"User: {image_token}\n\nDescribe this image.\nAssistant: {item['caption']}")

    proc = image_processor(images=[b["image"] for b in batch], return_tensors="pt")
    pixel_values = proc["pixel_values"]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if image_token not in texts[0]:
        raise RuntimeError("Template must include image_token")
    ids_probe = tokenizer.encode(image_token, add_special_tokens=False)
    if len(ids_probe) != 1 or ids_probe[0] != image_token_id:
        raise RuntimeError(f"image_token must map to single id {image_token_id}, got {ids_probe}")

    expanded_ids = []
    expanded_mask = []
    for i in range(input_ids.shape[0]):
        row_ids = input_ids[i].tolist()
        row_m = attention_mask[i].tolist()
        new_ids: List[int] = []
        new_m: List[int] = []
        for tid, m in zip(row_ids, row_m):
            if m == 0:
                continue
            if tid == image_token_id:
                new_ids.extend([image_token_id] * patch_count)
                new_m.extend([1] * patch_count)
            else:
                new_ids.append(tid)
                new_m.append(1)
        expanded_ids.append(new_ids)
        expanded_m.append(new_m)

    max_len = max(len(x) for x in expanded_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bsz = len(expanded_ids)
    padded_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    padded_m = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, (ids, m) in enumerate(zip(expanded_ids, expanded_m)):
        L = len(ids)
        padded_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        padded_m[i, :L] = torch.tensor(m, dtype=torch.long)

    labels = padded_ids.clone()
    labels[padded_ids == image_token_id] = -100
    labels[padded_m == 0] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": padded_ids,
        "attention_mask": padded_m,
        "labels": labels,
    }


def load_train_val_datasets(
    dataset_id: str,
    dataset_config: Optional[str],
    image_column: str,
    caption_column: str,
    max_train_samples: int,
    val_size: int,
    seed: int,
):
    from datasets import load_dataset

    cfg = (dataset_config or "").strip()
    if cfg:
        raw = load_dataset(dataset_id, cfg, trust_remote_code=True)
    else:
        raw = load_dataset(dataset_id, trust_remote_code=True)

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    full = raw[split_name]
    n = len(full)
    if n < val_size + 100:
        raise SystemExit(f"Dataset too small: len={n}")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_idx = set(indices[:val_size])
    train_idx = [i for i in indices[val_size:] if i not in val_idx][:max_train_samples]

    train_ds = HfImageCaptionDataset(Subset(full, train_idx), image_column, caption_column, len(train_idx))
    val_ds = HfImageCaptionDataset(Subset(full, list(val_idx)[:val_size]), image_column, caption_column, val_size)
    return train_ds, val_ds


class MultimodalProjectorExport(nn.Module):
    """Same layout / keys as imagination_runtime.clip_projector_vlm.MultimodalProjector."""

    def __init__(self, vision_hidden: int, text_hidden: int, hidden_act: str = "gelu") -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=True)
        self.act = ACT2FN[hidden_act]
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


def mlp_state_dict_from_trained_linear(
    linear: nn.Module,
    vision_hidden: int,
    llm_hidden: int,
    projector_act: str,
) -> Dict[str, torch.Tensor]:
    """
    Copy trained nn.Linear into linear_1; set linear_2 to identity (I, 0).
    v1.3 inference does y = linear_2(gelu(linear_1(x))), not linear_1(x) alone.
    """
    act = (projector_act or "gelu").strip().lower()
    if act not in ACT2FN:
        raise ValueError(f"Unknown projector_act {act!r}; choose one of {sorted(ACT2FN.keys())}")
    m = MultimodalProjectorExport(vision_hidden, llm_hidden, hidden_act=act)
    lin_sd = {k: v.detach().cpu().float() for k, v in linear.state_dict().items()}
    m.linear_1.load_state_dict(lin_sd)
    with torch.no_grad():
        m.linear_2.weight.copy_(torch.eye(llm_hidden, dtype=torch.float32))
        m.linear_2.bias.zero_()
    return {k: v.cpu().float() for k, v in m.state_dict().items()}


def write_attach_vision_meta(
    path: Path,
    *,
    llm_model: str,
    vision_model: str,
    image_token: str,
    image_token_id: int,
    patch_count: int,
    vision_feature_layer: int,
    llm_hidden: int,
    vision_hidden: int,
    projector_act: str,
    export_note: str,
) -> None:
    payload = {
        "llm_model": llm_model,
        "vision_model": vision_model,
        "image_token": image_token,
        "image_token_id": image_token_id,
        "patch_count": patch_count,
        "vision_feature_layer": vision_feature_layer,
        "llm_hidden": llm_hidden,
        "vision_hidden": vision_hidden,
        "projector_act": projector_act,
        "projector_export_note": export_note,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_checkpoint(
    output_dir: Path,
    projection: nn.Module,
    cfg: TrainConfig,
    optimizer_step: int,
    meta_extra: Dict[str, Any],
    *,
    export_mlp: bool,
    vision_hidden: int,
    llm_hidden: int,
    projector_act: str,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{optimizer_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projection.state_dict(), ckpt_dir / "projection_layer.pt")
    if export_mlp:
        torch.save(
            mlp_state_dict_from_trained_linear(projection, vision_hidden, llm_hidden, projector_act),
            ckpt_dir / "projector.pt",
        )
    with (ckpt_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), **meta_extra, "optimizer_step": optimizer_step}, f, indent=2)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available; training will be extremely slow.")
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_tok = args.image_token
    if args.add_image_token:
        tid_probe = tokenizer.convert_tokens_to_ids(image_tok)
        if tid_probe == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_tok]})

    tok_ids = tokenizer.encode(image_tok, add_special_tokens=False)
    if len(tok_ids) != 1:
        raise SystemExit(
            f"--image_token must encode to exactly 1 id; got {tok_ids}. Use --add_image_token or fix token."
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

    patch_count = vision_patch_count(vision.config)
    llm_hidden = int(llm.config.hidden_size)
    vision_hidden = int(vision.config.hidden_size)

    if vision_hidden != 1024 or llm_hidden != 3072:
        print(
            f"Note: vision_hidden={vision_hidden}, llm_hidden={llm_hidden} "
            f"(repo Imagination uses 1024->3072; Linear will match actual dims).",
            flush=True,
        )

    projection = nn.Linear(vision_hidden, llm_hidden, bias=True)
    projection.to(device=device, dtype=dtype)

    vision.requires_grad_(False)
    llm.requires_grad_(False)
    vision.eval()
    llm.eval()
    vision.to(device=device, dtype=dtype)
    llm.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)

    if hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        print("[train] LLM gradient checkpointing enabled (saves VRAM on forward).", flush=True)

    train_ds, val_ds = load_train_val_datasets(
        args.dataset_id,
        (args.dataset_config or "").strip() or None,
        args.image_column,
        args.caption_column,
        args.max_train_samples,
        args.val_size,
        args.seed,
    )
    print(
        f"[train] Using {len(train_ds)} train and {len(val_ds)} val rows "
        f"(requested max_train={args.max_train_samples}).",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))

    cfg = TrainConfig(
        llm_model=args.llm_model,
        vision_model=args.vision_model,
        dataset_id=args.dataset_id,
        dataset_config=(args.dataset_config or "").strip() or None,
        image_column=args.image_column,
        caption_column=args.caption_column,
        max_train_samples=len(train_ds),
        val_size=len(val_ds),
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        vision_feature_layer=args.vision_feature_layer,
        bf16=use_bf16,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        image_token=args.image_token,
    )
    meta_extra = {
        "image_token_id": image_token_id,
        "patch_count": patch_count,
        "llm_hidden_size": llm_hidden,
        "vision_hidden_size": vision_hidden,
        "note": "Single nn.Linear on patch dim; state_dict keys: weight, bias",
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), **meta_extra}, f, indent=2)

    tokenizer.save_pretrained(output_dir / "tokenizer_snapshot")

    @torch.no_grad()
    def run_validation() -> float:
        projection.eval()
        losses = []
        for batch in val_loader:
            batch_t = collate_batch(
                batch,
                tokenizer,
                image_processor,
                args.image_token,
                image_token_id,
                patch_count,
                args.max_length,
                device,
            )
            pv = batch_t["pixel_values"].to(device=device, dtype=dtype)
            input_ids = batch_t["input_ids"].to(device)
            attention_mask = batch_t["attention_mask"].to(device)
            labels = batch_t["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=device.type == "cuda"):
                img_h = vision_patch_embeddings(vision, pv, args.vision_feature_layer, patch_count)
                proj = projection(img_h)
                emb = llm.get_input_embeddings()(input_ids)
                inputs_embeds = merge_prefix_embeddings(emb, input_ids, image_token_id, proj)
                out = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
            losses.append(float(out.loss.detach().cpu()))
        projection.train()
        return float(sum(losses) / max(1, len(losses)))

    optimizer_step = 0
    accum_counter = 0
    accum_losses: List[float] = []
    loss_window: List[float] = []

    projection.train()
    optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(mean_loss_in_accum: float) -> None:
        nonlocal optimizer_step
        if use_bf16:
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        loss_window.append(mean_loss_in_accum)
        if len(loss_window) > args.log_every:
            loss_window.pop(0)
        if optimizer_step % args.log_every == 0 and loss_window:
            avg = sum(loss_window) / len(loss_window)
            print(f"[step {optimizer_step}] train_loss_recent_avg={avg:.4f} (window={len(loss_window)})", flush=True)
        if optimizer_step % args.save_every == 0:
            save_checkpoint(
                output_dir,
                projection,
                cfg,
                optimizer_step,
                meta_extra,
                export_mlp=args.export_mlp_projector,
                vision_hidden=vision_hidden,
                llm_hidden=llm_hidden,
                projector_act=args.projector_act,
            )
            print(f"[step {optimizer_step}] saved checkpoint.", flush=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch_t = collate_batch(
                batch,
                tokenizer,
                image_processor,
                args.image_token,
                image_token_id,
                patch_count,
                args.max_length,
                device,
            )
            pv = batch_t["pixel_values"].to(device=device, dtype=dtype)
            input_ids = batch_t["input_ids"].to(device)
            attention_mask = batch_t["attention_mask"].to(device)
            labels = batch_t["labels"].to(device)

            def forward_loss() -> torch.Tensor:
                img_h = vision_patch_embeddings(vision, pv, args.vision_feature_layer, patch_count)
                proj = projection(img_h)
                emb = llm.get_input_embeddings()(input_ids)
                inputs_embeds = merge_prefix_embeddings(emb, input_ids, image_token_id, proj)
                out = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                return out.loss

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_loss = forward_loss()
                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    raw_loss = forward_loss()
                loss = raw_loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

            accum_counter += 1
            accum_losses.append(float(raw_loss.detach().cpu()))

            if accum_counter >= args.gradient_accumulation_steps:
                m = sum(accum_losses) / len(accum_losses)
                _optimizer_step(m)
                accum_counter = 0
                accum_losses.clear()

            pbar.set_postfix(opt_step=optimizer_step, loss=float(raw_loss.detach().cpu()))

        val_loss = run_validation()
        print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}", flush=True)

        if accum_counter > 0:
            m = sum(accum_losses) / len(accum_losses)
            _optimizer_step(m)
            accum_counter = 0
            accum_losses.clear()
            print(f"[epoch {epoch+1}] flushed partial gradient accumulation.", flush=True)

    # Final save
    torch.save(projection.state_dict(), output_dir / "projection_layer.pt")
    save_checkpoint(
        output_dir,
        projection,
        cfg,
        optimizer_step,
        meta_extra,
        export_mlp=args.export_mlp_projector,
        vision_hidden=vision_hidden,
        llm_hidden=llm_hidden,
        projector_act=args.projector_act,
    )
    export_note = (
        "Trained as single Linear on patches; exported MLP has linear_1=Linear, linear_2=Identity. "
        "Imagination inference applies gelu between layers, so activations differ from training "
        "(see script docstring). For best alignment, continue with attach_vision_multimodal.py MLP."
    )
    with (output_dir / "attach_vision_multimodal_meta_linear.json").open("w", encoding="utf-8") as f:
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
                "projector_type": "linear",
                "weights_file": "projection_layer.pt",
                "projector_act": args.projector_act,
                "projector_export_note": export_note,
            },
            f,
            indent=2,
        )
    if args.export_mlp_projector:
        torch.save(
            mlp_state_dict_from_trained_linear(
                projection, vision_hidden, llm_hidden, args.projector_act
            ),
            output_dir / "projector.pt",
        )
        write_attach_vision_meta(
            output_dir / "attach_vision_multimodal_meta.json",
            llm_model=args.llm_model,
            vision_model=args.vision_model,
            image_token=args.image_token,
            image_token_id=image_token_id,
            patch_count=patch_count,
            vision_feature_layer=args.vision_feature_layer,
            llm_hidden=llm_hidden,
            vision_hidden=vision_hidden,
            projector_act=args.projector_act,
            export_note=export_note,
        )
        print(
            f"[export] Wrote projector.pt + attach_vision_multimodal_meta.json for v1.3 bundle dir.",
            flush=True,
        )
    print(f"Done. Artifacts under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
