from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_llm.data import MedicalExample, deterministic_eval_subset, to_example, write_jsonl


def prepare_medquad(config: dict[str, Any], tokenizer: Any, output_dir: str | Path) -> dict[str, int]:
    from datasets import load_dataset

    data = config["data"]
    dataset = load_dataset(
        data["dataset_name"],
        revision=data["dataset_revision"],
        split=data["split"],
    )
    split = dataset.train_test_split(
        test_size=float(data["test_size"]), seed=int(config["project"]["seed"])
    )
    max_length = int(data["max_sequence_length"])

    def convert(rows) -> list[MedicalExample]:
        output = []
        for row in rows:
            try:
                example = to_example(row, data["question_field"], data["answer_field"])
            except ValueError:
                continue
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example.question},
                    {"role": "assistant", "content": example.reference},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            if len(tokenizer(text)["input_ids"]) <= max_length:
                output.append(example)
        return output

    train = convert(split["train"])
    test = convert(split["test"])
    evaluation = deterministic_eval_subset(
        test, int(data["evaluation_size"]), int(config["project"]["seed"])
    )
    directory = Path(output_dir)
    write_jsonl(train, directory / "train.jsonl")
    write_jsonl(test, directory / "test.jsonl")
    write_jsonl(evaluation, directory / "evaluation_300.jsonl")
    return {
        "raw": len(dataset),
        "train": len(train),
        "test": len(test),
        "evaluation": len(evaluation),
    }
