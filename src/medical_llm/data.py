from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MedicalExample:
    sample_id: str
    question: str
    reference: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def stable_sample_id(question: str, reference: str) -> str:
    return hashlib.sha256(f"{question}\n{reference}".encode("utf-8")).hexdigest()[:16]


def to_example(record: dict[str, Any], question_field: str, answer_field: str) -> MedicalExample:
    question = record.get(question_field)
    answer = record.get(answer_field)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("medical example has no valid question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("medical example has no valid answer")
    return MedicalExample(stable_sample_id(question, answer), question.strip(), answer.strip())


def format_chat(example: MedicalExample, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.reference},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def deterministic_eval_subset(records: Sequence[MedicalExample], size: int, seed: int) -> list[MedicalExample]:
    if size <= 0 or size > len(records):
        raise ValueError("evaluation size must be between 1 and test set size")
    ordered = sorted(
        records,
        key=lambda item: (
            hashlib.sha256(f"{seed}|{item.sample_id}".encode()).hexdigest(),
            item.sample_id,
        ),
    )
    return ordered[:size]


def write_jsonl(records: Iterable[MedicalExample], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[MedicalExample]:
    return [MedicalExample(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()]
