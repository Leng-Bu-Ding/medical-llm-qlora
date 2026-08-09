from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.config import load_config
from medical_llm.evaluation import evaluate_predictions, read_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/evaluation_summary.json"))
    args = parser.parse_args()
    report = evaluate_predictions(read_prediction_jsonl(args.predictions), load_config(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
