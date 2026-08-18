from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.config import load_config
from medical_llm.data import MedicalExample
from medical_llm.inference import generate_base_and_finetuned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--cases", default=str(ROOT / "data/safety_cases.jsonl"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/safety_predictions.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        cases = cases[: args.limit]
    examples = [
        MedicalExample(item["case_id"], item["question"], "safety audit has no gold answer")
        for item in cases
    ]
    generate_base_and_finetuned(load_config(args.config), examples, args.adapter, args.output)
    predictions = [
        json.loads(line) for line in Path(args.output).read_text(encoding="utf-8").splitlines()
    ]
    with Path(args.output).open("w", encoding="utf-8", newline="\n") as stream:
        for case, prediction in zip(cases, predictions, strict=True):
            stream.write(
                json.dumps({**case, **prediction}, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(f"Saved {len(predictions)} paired safety responses to {args.output}")


if __name__ == "__main__":
    main()
