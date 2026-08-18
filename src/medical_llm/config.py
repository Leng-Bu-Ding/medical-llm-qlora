from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["training"]["effective_batch_size"] != (
        config["training"]["per_device_train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    ):
        raise ValueError("effective batch size does not match batch x accumulation")
    if config["data"]["evaluation_size"] <= 0:
        raise ValueError("evaluation_size must be positive")
    if config["data"]["protocol"] not in {"legacy", "clean"}:
        raise ValueError("data.protocol must be legacy or clean")
    if not 0 < float(config["data"]["test_size"]) < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < float(config["data"]["validation_size"]) < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if float(config["data"]["test_size"]) + float(config["data"]["validation_size"]) >= 1:
        raise ValueError("validation_size + test_size must be below 1")
    if not 0 < float(config["data"]["near_duplicate_threshold"]) <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    return config


def config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
