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


@dataclass(frozen=True)
class ModelPaths:
    root: str

    @property
    def main_llm(self) -> str:
        return self.root

    @property
    def cad_coder(self) -> str:
        return os.path.join(self.root, "modules", "cad", "qwen finetuned coder(3b)")

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
