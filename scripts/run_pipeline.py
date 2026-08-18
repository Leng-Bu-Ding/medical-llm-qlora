from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def _copy_public(source: Path, public_dir: Path) -> None:
    if source.exists():
        public_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, public_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", choices=("legacy", "clean"), default="clean")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--include-external-eval", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / args.run_id
    data_dir = run_dir / "data"
    training_dir = run_dir / "training"
    predictions = run_dir / "predictions_300.jsonl"
    evaluation_summary = run_dir / "evaluation_summary.json"
    error_cases = run_dir / "error_cases.jsonl"
    safety_predictions = run_dir / "safety_predictions.jsonl"
    safety_summary = run_dir / "safety_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config",
            args.config,
            "--protocol",
            args.protocol,
            "--output",
            str(data_dir),
        ]
    )
    train_command = [
        sys.executable,
        "scripts/train_qlora.py",
        "--config",
        args.config,
        "--train-data",
        str(data_dir / "train.jsonl"),
        "--validation-data",
        str(data_dir / "validation.jsonl"),
        "--output-dir",
        str(training_dir),
    ]
    if args.resume_from_checkpoint:
        train_command.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
    if args.smoke:
        train_command.extend(
            ["--max-steps", "10", "--train-limit", "8", "--validation-limit", "8"]
        )
    _run(train_command)

    inference_command = [
        sys.executable,
        "scripts/run_inference.py",
        "--config",
        args.config,
        "--data",
        str(data_dir / "evaluation_300.jsonl"),
        "--adapter",
        str(training_dir / "adapter"),
        "--output",
        str(predictions),
    ]
    if args.smoke:
        inference_command.extend(["--limit", "8"])
    _run(inference_command)
    _run(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--config",
            args.config,
            "--predictions",
            str(predictions),
            "--output",
            str(evaluation_summary),
            "--errors-output",
            str(error_cases),
        ]
    )

    safety_command = [
        sys.executable,
        "scripts/run_safety_inference.py",
        "--config",
        args.config,
        "--adapter",
        str(training_dir / "adapter"),
        "--output",
        str(safety_predictions),
    ]
    if args.smoke:
        safety_command.extend(["--limit", "8"])
    _run(safety_command)
    _run(
        [
            sys.executable,
            "scripts/evaluate_safety.py",
            "--predictions",
            str(safety_predictions),
            "--output",
            str(safety_summary),
        ]
    )

    public_sources = [
        data_dir / "data_manifest.json",
        training_dir / "training_summary.json",
        evaluation_summary,
        error_cases,
        safety_summary,
    ]
    if args.include_external_eval:
        pubmedqa_data = run_dir / "pubmedqa_100.jsonl"
        pubmedqa_predictions = run_dir / "pubmedqa_predictions.jsonl"
        pubmedqa_summary = run_dir / "pubmedqa_summary.json"
        _run(
            [
                sys.executable,
                "scripts/prepare_pubmedqa.py",
                "--config",
                args.config,
                "--output",
                str(pubmedqa_data),
            ]
        )
        pubmedqa_inference = [
            sys.executable,
            "scripts/run_inference.py",
            "--config",
            args.config,
            "--data",
            str(pubmedqa_data),
            "--adapter",
            str(training_dir / "adapter"),
            "--output",
            str(pubmedqa_predictions),
        ]
        if args.smoke:
            pubmedqa_inference.extend(["--limit", "8"])
        _run(pubmedqa_inference)
        _run(
            [
                sys.executable,
                "scripts/evaluate_pubmedqa.py",
                "--config",
                args.config,
                "--predictions",
                str(pubmedqa_predictions),
                "--output",
                str(pubmedqa_summary),
            ]
        )
        public_sources.extend([run_dir / "pubmedqa_manifest.json", pubmedqa_summary])

    manifest = {
        "run_id": args.run_id,
        "protocol": args.protocol,
        "smoke": args.smoke,
        "include_external_eval": args.include_external_eval,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_tested" if args.smoke else "measured",
        "artifacts": [str(path.relative_to(ROOT)) for path in public_sources if path.exists()],
    }
    (run_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.smoke:
        public_dir = ROOT / "results" / "public" / args.run_id
        for source in [*public_sources, run_dir / "pipeline_manifest.json"]:
            _copy_public(source, public_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
