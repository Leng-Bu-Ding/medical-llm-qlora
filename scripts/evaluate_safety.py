from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.safety import summarize_safety


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs/safety_summary.json"))
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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
