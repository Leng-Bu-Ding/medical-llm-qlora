from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.config import load_config
from medical_llm.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--train-data", default=str(ROOT / "data/processed/train.jsonl"))
    parser.add_argument("--validation-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--lora-rank", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                load_config(args.config),
                args.train_data,
                args.validation_data,
                output_dir=args.output_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
                max_steps=args.max_steps,
                train_limit=args.train_limit,
                validation_limit=args.validation_limit,
                lora_rank=args.lora_rank,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
