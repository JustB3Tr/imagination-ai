# Pretrain stack (~8B text + vision roadmap)

This folder is a **clean handoff** for your own **continued pretraining (CPT)** on a **~8B Llama-style** language backbone, then a **Phase 2** path to **multimodal (vision + text)** in the usual LLaVA-style pattern (vision tower + projector + causal LM).

## Layout

| Path | Purpose |
|------|--------|
| `configs/text/` | Hugging Face `LlamaForCausalLM` config (~8B-class, GQA). Start CPT here. |
| `configs/vision/` | `CLIPVisionModel`-style config (ViT-L/14 @ 336). Use a **frozen pretrained** encoder first, or train later. |
| `configs/multimodal/` | Nested **LLaVA-style** skeleton aligning text + vision dims. Wire after text CPT stabilizes. |
| `training/` | Example hyperparameter templates (adapt to your trainer: Megatron, LLaMA-Factory, torchrun, etc.). |
| `scripts/` | Param counter + **random-init** export for a fresh text checkpoint. |

## Phases (recommended)

1. **Text CPT / pretrain**  
   - Use `configs/text/config.json` with your data pipeline.  
   - Target: stable loss, good base LM before adding vision.  
   - Reuse your existing tokenizer assets if vocab matches (`vocab_size` 128256 in this template—**must match** your tokenizer).

2. **Vision encoder**  
   - Default plan: load **OpenAI CLIP ViT-L/14-336** weights (or SigLIP) and **freeze** for first multimodal stage; `configs/vision/` documents the matching architecture.

3. **Multimodal alignment**  
   - Add **image placeholder token(s)** to the tokenizer; extend `vocab_size` in the **multimodal** config accordingly.  
   - Train **projector** (+ optionally LoRA on LLM) on image–caption / interleaved data.  
   - Use `configs/multimodal/llava_skeleton.json` as a structural reference; load with `LlavaConfig` / your trainer’s equivalent.

## Parameter budget

Run:

```bash
cd pretrain-vision-8b
python scripts/count_params.py
python scripts/count_params.py --vision
```

to print **exact** parameter counts for the text config (and optional vision). The bundled text config is **~7.5B** parameters (`LlamaForCausalLM` on meta device).

Random-init checkpoint (large on disk):

```bash
python scripts/init_text_model.py --out ./outputs/init-text-8b
```

## Text CPT trainer (`scripts/train_text_cpt.py`)

Trains **next-token prediction** on a JSONL column (default `text`). The model size is whatever `configs/text/config.json` defines (~8B in this bundle).

**Install:** `pip install torch transformers datasets accelerate` (optional: `deepspeed` for ZeRO-3 offload).

**Train from scratch** (random weights + your tokenizer — must match `vocab_size` or embeddings are resized):

```bash
cd pretrain-vision-8b
python scripts/train_text_cpt.py \
  --from_scratch \
  --tokenizer_dir /path/to/tokenizer \
  --train_file /path/to/train.jsonl \
  --output_dir /path/to/cpt_out \
  --block_size 2048 \
  --max_steps 5000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16
```

**Continue** from an existing HF folder (weights + tokenizer):

```bash
python scripts/train_text_cpt.py \
  --model_dir /path/to/existing_llama_hf \
  --train_file /path/to/train.jsonl \
  --output_dir /path/to/cpt_out \
  --bf16
```

**Single 24GB GPU + ~8B full Adam** often OOMs. Try **`--deepspeed training/ds_zero3_offload.json`** (install DeepSpeed; expect slower offload) or use **multi-GPU / larger GPU**.

**JSONL format:**

```json
{"text": "First document plain text..."}
{"text": "Second document..."}
```

## Important

- **Tokenizer + `vocab_size`** in `configs/text/config.json` **must** match the tokenizer you train with. If your tokenizer differs, edit `vocab_size` (and special token IDs) before pretraining.  
- **Vision** adds a separate tower; total multimodal params = vision + projector + LLM (often LLM starts from your CPT checkpoint).  
- This repo folder does **not** run pretraining by itself—it supplies **configs + scripts** you plug into your training stack.
