"""
Compute budget tracker for Colab L4: 100 points/month at 1.71 points/hour.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional

MONTHLY_POINTS = 100
L4_RATE = 1.71  # points per hour
WARN_THRESHOLD_HOURS = 5.0


def _budget_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "budget.json")


def _load_budget(root: str) -> dict:
    path = _budget_path(root)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"month": datetime.now().strftime("%Y-%m"), "points_used": 0.0, "session_start": None}


def _save_budget(root: str, data: dict) -> None:
    path = _budget_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _reset_if_new_month(data: dict) -> dict:
    now = datetime.now()
    current_month = now.strftime("%Y-%m")
    if data.get("month") != current_month:
        data["month"] = current_month
        data["points_used"] = 0.0
        data["session_start"] = None
    return data


def remaining_hours(root: str) -> float:
    """Estimated remaining GPU hours this month."""
    data = _load_budget(root)
    data = _reset_if_new_month(data)
    points_used = float(data.get("points_used", 0))
    if data.get("session_start"):
        elapsed = time.time() - data["session_start"]
        points_used += (elapsed / 3600.0) * L4_RATE
    remaining_points = max(0, MONTHLY_POINTS - points_used)
    return remaining_points / L4_RATE


def log_session_start(root: str) -> None:
    """Record that a GPU session has started."""
    data = _load_budget(root)
    data = _reset_if_new_month(data)
    data["session_start"] = time.time()
    _save_budget(root, data)


def log_session_end(root: str) -> None:
    """Record that a GPU session has ended; add elapsed time to points_used."""
    data = _load_budget(root)
    data = _reset_if_new_month(data)
    if data.get("session_start"):
        elapsed = time.time() - data["session_start"]
        data["points_used"] = data.get("points_used", 0) + (elapsed / 3600.0) * L4_RATE
    data["session_start"] = None
    _save_budget(root, data)


def is_low_budget(root: str) -> bool:
    """True if remaining hours < WARN_THRESHOLD_HOURS."""
    return remaining_hours(root) < WARN_THRESHOLD_HOURS
