from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = output / "pubmedqa_100.jsonl"
    predictions = output / "pubmedqa_predictions.jsonl"
    _run(
        [
            sys.executable,
            "scripts/prepare_pubmedqa.py",
            "--config",
            args.config,
            "--output",
            str(data),
        ]
    )
    inference = [
        sys.executable,
        "scripts/run_inference.py",
        "--config",
        args.config,
        "--data",
        str(data),
        "--adapter",
        args.adapter,
        "--output",
        str(predictions),
    ]
    if args.limit is not None:
        inference.extend(["--limit", str(args.limit)])
    _run(inference)
    _run(
        [
            sys.executable,
            "scripts/evaluate_pubmedqa.py",
            "--config",
            args.config,
            "--predictions",
            str(predictions),
            "--output",
            str(output / "pubmedqa_summary.json"),
        ]
    )


if __name__ == "__main__":
    main()
