from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from .paths import ModelPaths


class TaskId(str, Enum):
    CHAT_MAIN = "chat_main"
    CAD_CODER = "cad_coder"
    DEEP_RESEARCH = "deep_research"
    IMAGE_TINY = "image_tiny"


@dataclass(frozen=True)
class TaskSpec:
    id: TaskId
    label: str
    description: str
    required_keys: List[str]


def get_task_specs() -> List[TaskSpec]:
    return [
        TaskSpec(
            id=TaskId.CHAT_MAIN,
            label="Chat (main)",
            description="Always-on main chat model loaded from the repo root.",
            required_keys=[],
        ),
        TaskSpec(
            id=TaskId.CAD_CODER,
            label="CAD / Coder (Qwen finetuned 3B)",
            description="Loads the CAD/coder finetune when selected.",
            required_keys=["cad_coder"],
        ),
        TaskSpec(
            id=TaskId.DEEP_RESEARCH,
            label="Deep research (embeddings + reranker + reasoning)",
            description="Loads retrieval models when selected.",
            required_keys=["embeddings", "reranker", "reasoning_llm"],
        ),
        TaskSpec(
            id=TaskId.IMAGE_TINY,
            label="Image (tiny SD)",
            description="Loads tiny image pipeline when selected.",
            required_keys=["tiny_sd"],
        ),
    ]


def build_loaders(model_paths: ModelPaths) -> Dict[str, Callable[[], object]]:
    raise NotImplementedError
