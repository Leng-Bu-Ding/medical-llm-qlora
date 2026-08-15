from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.safety import (
    build_blind_review_materials,
    summarize_safety,
    write_review_csv,
    write_review_key,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/safety_summary.json"))
    parser.add_argument("--review-output")
    parser.add_argument("--review-key-output")
    parser.add_argument("--secondary-review-output")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
    ]
    report = {
        "warning": "Heuristic portfolio safety audit; not a clinical validation.",
        "base": summarize_safety(records, "base_prediction"),
        "fine_tuned": summarize_safety(records, "ft_prediction"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows, key, secondary_ids = build_blind_review_materials(records, seed=args.seed)
    review_output = (
        Path(args.review_output)
        if args.review_output
        else output.with_name("review_primary.csv")
    )
    key_output = (
        Path(args.review_key_output)
        if args.review_key_output
        else output.with_name("review_key.json")
    )
    secondary_output = (
        Path(args.secondary_review_output)
        if args.secondary_review_output
        else output.with_name("review_secondary_20.csv")
    )
    write_review_csv(rows, review_output)
    write_review_key(key, key_output)
    write_review_csv([row for row in rows if row["case_id"] in secondary_ids], secondary_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
