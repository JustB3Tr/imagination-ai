# Vision tower (Phase 2)

`clip_vit_large_patch14_336_config.json` matches the **vision** branch of **`openai/clip-vit-large-patch14-336`** (ViT-L/14, 336×336, `quick_gelu`).

**Practical path**

1. **Load pretrained weights** from Hugging Face (`CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")`) and **freeze** for the first multimodal stage.
2. **Projector** maps vision outputs (dim **1024** per patch embedding, before or after CLIP projection depending on your code) into the LLM hidden size (**4096** in this repo’s text config). LLaVA-style code usually implements a 2-layer MLP; your trainer may build this automatically from `LlavaConfig`.
3. **Image tokens**: (336/14)² = **576** patch tokens per image (no class token in this ViT setup for patch grid—confirm in your `Llava` implementation).

**Alternatives**

- **SigLIP** (e.g. `google/siglip-so400m-patch14-384`) — stronger on many benchmarks; swap `vision_config` and preprocessing (image size, normalize).
