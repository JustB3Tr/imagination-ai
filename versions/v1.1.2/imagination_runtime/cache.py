from __future__ import annotations

import gc
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Optional

import torch


@dataclass
class RuntimeCache:
    """
    Process-wide caches so the main LLM stays loaded and module models
    are lazy-loaded and reused.
    """

    lock: RLock = field(default_factory=RLock)

    main_tokenizer: Optional[Any] = None
    main_model: Optional[Any] = None

    module_objects: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        with self.lock:
            return self.module_objects.get(key)

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.module_objects[key] = value

    def clear_modules(self) -> None:
        with self.lock:
            self.module_objects.clear()
        self._free_memory()

    def _free_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
