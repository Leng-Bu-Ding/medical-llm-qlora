from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def installed_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_summary(root: str | Path | None = None) -> dict[str, Any]:
    return {
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": installed_versions(
            [
                "torch",
                "transformers",
                "datasets",
                "trl",
                "peft",
                "unsloth",
                "bitsandbytes",
                "scikit-learn",
                "bert-score",
            ]
        ),
    }


def write_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
