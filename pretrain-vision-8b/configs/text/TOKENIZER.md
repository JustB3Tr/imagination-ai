# Tokenizer alignment

The text `config.json` sets `vocab_size`: **128256** and BOS/EOS IDs consistent with **Llama 3–style** chat tokenization.

**Before pretraining:**

1. Point your dataloader at the **same** tokenizer you will ship in production (e.g. copy `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json` from your current `IMAGINATION_ROOT` if compatible).
2. If your tokenizer’s `len(tokenizer) != 128256`, update **`vocab_size`** in `config.json` to match, and fix **`bos_token_id` / `eos_token_id`** to match `tokenizer_config.json`.
3. For multimodal later, reserve at least one unused ID for `<image>` (or use the model’s image token scheme) and bump `vocab_size` if you add new tokens.
