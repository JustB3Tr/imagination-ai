from __future__ import annotations

import os
from dataclasses import dataclass


def resolve_root_path(root_path: str | None) -> str:
    """
    Resolve the mounted imagination root path for local or Colab.
    Priority:
      1) explicit function argument
      2) IMAGINATION_ROOT env var
      3) default Colab mount path
    """
    p = (root_path or "").strip()
    if not p:
        p = (os.getenv("IMAGINATION_ROOT") or "").strip()
    if not p:
        p = "/content/imagination-v1.1.0"
    return p.rstrip("/\\")


def resolve_vision_projector_bundle_dir(llm_root: str) -> str | None:
    """
    Directory containing projector.pt + attach_vision_multimodal_meta.json (CLIP + projector CPT).

    - Set IMAGINATION_VISION_PROJECTOR_DIR to an absolute path, or a path relative to IMAGINATION_ROOT.
    - If unset, uses <IMAGINATION_ROOT>/vision_cpt_out_real when both files exist.
    - Set IMAGINATION_USE_CLIP_PROJECTOR=0 to disable and use a native HF VLM folder instead.
    """
    if (os.getenv("IMAGINATION_USE_CLIP_PROJECTOR") or "").strip().lower() in ("0", "false", "no"):
        return None
    override = (os.getenv("IMAGINATION_VISION_PROJECTOR_DIR") or "").strip()
    if override:
        p = override if os.path.isabs(override) else os.path.join(llm_root, override)
        p = os.path.normpath(p)
    else:
        p = os.path.join(llm_root, "vision_cpt_out_real")
    proj = os.path.join(p, "projector.pt")
    meta = os.path.join(p, "attach_vision_multimodal_meta.json")
    if os.path.isfile(proj) and os.path.isfile(meta):
        return p
    return None


def resolve_cad_coder_dir(root: str) -> str:
    """
    Directory for the coder module weights.
    Set IMAGINATION_CAD_CODER_PATH to an absolute path, or a path relative to IMAGINATION_ROOT,
    if your folder name differs from the default (e.g. after swapping 3B → 7B weights).
    """
    override = (os.getenv("IMAGINATION_CAD_CODER_PATH") or "").strip()
    if override:
        p = override
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        return os.path.normpath(p)
    return os.path.join(root, "modules", "cad", "qwen finetuned coder(3b)")


@dataclass(frozen=True)
class ModelPaths:
    root: str

    @property
    def main_llm(self) -> str:
        return self.root

    @property
    def main_vlm(self) -> str:
        """HF folder for 8B VLM; override with IMAGINATION_MAIN_VLM_PATH (abs or relative to root)."""
        override = (os.getenv("IMAGINATION_MAIN_VLM_PATH") or "").strip()
        if override:
            p = override
            if not os.path.isabs(p):
                p = os.path.join(self.root, p)
            return os.path.normpath(p)
        return self.root

    @property
    def cad_coder(self) -> str:
        return resolve_cad_coder_dir(self.root)

    @property
    def reasoning_llm(self) -> str:
        return os.path.join(self.root, "modules", "reasoning", "7b qwen reasoning finetuned")

    @property
    def embeddings(self) -> str:
        return os.path.join(self.root, "modules", "research", "embeddings", "bge-small-en-v1.5")

    @property
    def reranker(self) -> str:
        return os.path.join(self.root, "modules", "research", "reranker", "bge-reranker-v2-m3")

    @property
    def tiny_sd(self) -> str:
        return os.path.join(self.root, "modules", "image", "tiny image dev model")
