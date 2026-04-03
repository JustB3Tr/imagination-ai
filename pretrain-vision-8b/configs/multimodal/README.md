# Multimodal skeleton (Phase 2+)

`llava_skeleton.json` is a **reference** layout for a **LLaVA-style** model: **CLIP ViT-L/14 @ 336** + **Llama ~8B** text stack.

## Before you load this in code

1. **Add an image token** to your tokenizer (e.g. `<image>`) and set **`image_token_index`** to that token’s integer id.
2. Set **`vocab_size`** (top-level and inside `text_config`) to **`len(tokenizer)`** after adding special tokens.
3. **`image_seq_length`**: 576 matches (336/14)² patch grid for this ViT; change if you change resolution or vision model.
4. **Transformers class names** drift by version (`LlavaForConditionalGeneration`, `LlavaNextForConditionalGeneration`, etc.). Validate with:

   ```python
   from transformers import AutoConfig
   AutoConfig.from_pretrained("path/to/folder", trust_remote_code=True)
   ```

5. **Weights**: you normally **initialize** from (a) CPT **text** checkpoint + (b) pretrained **CLIP** vision weights + (c) **random** projector, then run alignment training—not from this JSON alone.

## Text-only CPT first

Complete **`configs/text/`** pretraining, **then** stitch weights into your multimodal trainer per its docs (often “load LLM from checkpoint, load vision from HF, init projector”).
