from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.config import load_config
from medical_llm.data import read_jsonl
from medical_llm.inference import generate_base_and_finetuned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data/processed/evaluation_300.jsonl"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/predictions_300.jsonl"))
    args = parser.parse_args()
    count = generate_base_and_finetuned(
        load_config(args.config), read_jsonl(args.data), args.adapter, args.output
    )
    print(f"Saved {count} paired predictions to {args.output}")


if __name__ == "__main__":
    main()
