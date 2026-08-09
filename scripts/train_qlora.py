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
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config), args.train_data), indent=2))


if __name__ == "__main__":
    main()
