from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    summaries = []
    for rank in (8, 16):
        rank_dir = Path(args.output_dir) / f"rank_{rank}"
        command = [
            sys.executable,
            "scripts/train_qlora.py",
            "--config",
            args.config,
            "--train-data",
            str(Path(args.data_dir) / "train.jsonl"),
            "--validation-data",
            str(Path(args.data_dir) / "validation.jsonl"),
            "--output-dir",
            str(rank_dir),
            "--max-steps",
            str(args.max_steps),
            "--train-limit",
            "2000",
            "--validation-limit",
            "300",
            "--lora-rank",
            str(rank),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        training = json.loads((rank_dir / "training_summary.json").read_text(encoding="utf-8"))
        history = json.loads((rank_dir / "trainer_log.json").read_text(encoding="utf-8"))
        eval_losses = [row["eval_loss"] for row in history if "eval_loss" in row]
        summaries.append(
            {
                "rank": rank,
                "trainable_parameters": training["trainable_parameters"],
                "peak_memory_gib": training["peak_memory_gib"],
                "runtime_seconds": training["runtime_seconds"],
                "steps_per_second": training["steps_per_second"],
                "final_validation_loss": eval_losses[-1] if eval_losses else None,
            }
        )
    output = Path(args.output_dir) / "ablation_summary.json"
    output.write_text(
        json.dumps(
            {
                "status": "measured_pilot_ablation",
                "train_limit": 2000,
                "validation_limit": 300,
                "max_steps": args.max_steps,
                "runs": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
