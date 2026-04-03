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
