from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["training"]["effective_batch_size"] != (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    ):
        raise ValueError("effective batch size does not match batch x accumulation")
    if config["data"]["evaluation_size"] <= 0:
        raise ValueError("evaluation_size must be positive")
    return config
