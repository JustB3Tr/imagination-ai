#!/usr/bin/env python3
"""
Causal language modeling (CPT / pretrain) for the Llama architecture in configs/text/config.json.

The ~8B parameter count is fixed by that JSON — this script *trains* those weights (next-token
loss), it does not change model size.

Hardware: full 8B Adam training on a single 24GB GPU usually does *not* fit. Options:
  - Multi-GPU + ZeRO-2/3 (DeepSpeed or FSDP), or
  - --deepspeed ../training/ds_zero3_offload.json (slow: CPU/NVMe offload), or
  - Run on A100/H100 class GPUs.

Dataset: JSONL with one object per line, e.g. {"text": "..."} (column name configurable).

Examples:
  # Random init from config (needs tokenizer files on disk)
  python scripts/train_text_cpt.py \\
    --llama_config ../configs/text/config.json \\
    --tokenizer_dir /path/to/tokenizer \\
    --train_file /path/to/train.jsonl \\
    --output_dir /path/to/out \\
    --from_scratch

  # Continue from HF folder (config + weights + tokenizer)
  python scripts/train_text_cpt.py \\
    --model_dir /path/to/checkpoint \\
    --train_file /path/to/train.jsonl \\
    --output_dir /path/to/out

  # With DeepSpeed ZeRO-3 offload (install: pip install deepspeed)
  python scripts/train_text_cpt.py ... --deepspeed ../training/ds_zero3_offload.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text CPT for Llama config (~8B)")
    p.add_argument(
        "--llama_config",
        type=Path,
        default=_ROOT / "configs" / "text" / "config.json",
        help="Llama config JSON (used with --from_scratch)",
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Existing HF model folder (config + weights). Mutually exclusive with --from_scratch.",
    )
    p.add_argument(
        "--from_scratch",
        action="store_true",
        help="Random-init LlamaForCausalLM from --llama_config (needs --tokenizer_dir)",
    )
    p.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=None,
        help="Tokenizer directory (tokenizer.json / tokenizer.model + tokenizer_config.json)",
    )
    p.add_argument("--train_file", type=Path, required=True, help="JSONL training path")
    p.add_argument("--text_column", type=str, default="text", help="JSONL field with raw text")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--block_size", type=int, default=2048, help="Sequence length for CPT")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed.json")
    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on Ampere+)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if bf16 unavailable")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_scratch and args.model_dir is not None:
        raise SystemExit("Use either --from_scratch or --model_dir, not both.")
    if args.from_scratch and args.tokenizer_dir is None:
        raise SystemExit("--from_scratch requires --tokenizer_dir")
    if not args.train_file.is_file():
        raise SystemExit(f"train_file not found: {args.train_file}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    if args.model_dir is not None:
        tok_path = args.tokenizer_dir or args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_dir),
            use_fast=True,
            trust_remote_code=True,
        )
        cfg = LlamaConfig.from_json_file(str(args.llama_config))
        model = LlamaForCausalLM(cfg)
        if args.bf16:
            model = model.to(dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if len(tokenizer) != model.config.vocab_size:
        print(
            f"[cpt] Resizing embeddings: tokenizer {len(tokenizer)} vs config {model.config.vocab_size}",
            flush=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    ds = load_dataset("json", data_files=str(args.train_file), split="train")

    if args.text_column not in ds.column_names:
        raise SystemExit(
            f"Column '{args.text_column}' not in dataset. Columns: {ds.column_names}",
        )

    def _tok(batch: dict) -> dict:
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.block_size,
        )

    tokenized = ds.map(
        _tok,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
        deepspeed=args.deepspeed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(
        f"[cpt] Train rows={len(tokenized)} block_size={args.block_size} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    with open(Path(args.output_dir) / "cpt_args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    print(f"[cpt] Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Install: pip install torch transformers datasets accelerate", file=sys.stderr)
        raise SystemExit(1) from e
