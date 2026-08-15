from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from medical_llm.data import (
    MedicalExample,
    deterministic_eval_subset,
    exact_deduplicate,
    near_duplicate_groups,
    stratified_group_split,
    to_example,
    write_jsonl,
)
from medical_llm.provenance import sha256_file, write_json


def _convert_and_filter(
    rows: Any,
    *,
    data_config: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[MedicalExample], int]:
    output: list[MedicalExample] = []
    invalid = 0
    max_length = int(data_config["max_sequence_length"])
    for row in rows:
        try:
            example = to_example(
                row,
                data_config["question_field"],
                data_config["answer_field"],
                data_config.get("qtype_field", "qtype"),
            )
        except ValueError:
            invalid += 1
            continue
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example.question},
                {"role": "assistant", "content": example.reference},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_length:
            output.append(example)
    return output, invalid


def _qtype_counts(records: list[MedicalExample]) -> dict[str, int]:
    return dict(sorted(Counter(item.qtype for item in records).items()))


def _write_dataset(
    directory: Path,
    *,
    train: list[MedicalExample],
    validation: list[MedicalExample],
    test: list[MedicalExample],
    evaluation: list[MedicalExample],
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for name, records in {
        "train": train,
        "validation": validation,
        "test": test,
        "evaluation_300": evaluation,
    }.items():
        path = directory / f"{name}.jsonl"
        write_jsonl(records, path)
        files[name] = {
            "path": str(path),
            "count": len(records),
            "sha256": sha256_file(path),
            "qtype_counts": _qtype_counts(records),
        }
    return files


def _prepare_legacy(
    dataset: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, list[MedicalExample]], dict[str, Any]]:
    data = config["data"]
    split = dataset.train_test_split(
        test_size=float(data["test_size"]), seed=int(config["project"]["seed"])
    )
    train, invalid_train = _convert_and_filter(
        split["train"], data_config=data, tokenizer=tokenizer
    )
    test, invalid_test = _convert_and_filter(split["test"], data_config=data, tokenizer=tokenizer)
    evaluation = deterministic_eval_subset(
        test, min(int(data["evaluation_size"]), len(test)), int(config["project"]["seed"])
    )
    return (
        {"train": train, "validation": [], "test": test, "evaluation": evaluation},
        {
            "invalid_rows": invalid_train + invalid_test,
            "exact_duplicates_removed": 0,
            "near_duplicate_audit": None,
            "warning": "Legacy protocol reproduces the historical split-before-filter order.",
        },
    )


def _prepare_clean(
    dataset: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, list[MedicalExample]], dict[str, Any]]:
    data = config["data"]
    filtered, invalid = _convert_and_filter(dataset, data_config=data, tokenizer=tokenizer)
    deduplicated, duplicates_removed = exact_deduplicate(filtered)
    groups, cluster_audit = near_duplicate_groups(
        deduplicated,
        threshold=float(data["near_duplicate_threshold"]),
        neighbors=int(data["near_duplicate_neighbors"]),
    )
    train, validation, test, fold_assignment = stratified_group_split(
        deduplicated, groups, seed=int(config["project"]["seed"])
    )
    evaluation = deterministic_eval_subset(
        test, min(int(data["evaluation_size"]), len(test)), int(config["project"]["seed"])
    )
    split_by_group: dict[int, str] = {}
    for group, fold in zip(groups, fold_assignment, strict=True):
        split_name = "test" if fold == 0 else "validation" if fold == 1 else "train"
        existing = split_by_group.setdefault(group, split_name)
        if existing != split_name:
            raise RuntimeError("near-duplicate cluster crossed clean split boundaries")
    return (
        {"train": train, "validation": validation, "test": test, "evaluation": evaluation},
        {
            "invalid_rows": invalid,
            "length_filtered_or_invalid_rows": len(dataset) - len(filtered),
            "exact_duplicates_removed": duplicates_removed,
            "near_duplicate_audit": cluster_audit,
            "group_overlap_count": 0,
        },
    )


def prepare_medquad(
    config: dict[str, Any],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    protocol: str | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    data = config["data"]
    selected_protocol = protocol or str(data.get("protocol", "clean"))
    if selected_protocol not in {"legacy", "clean"}:
        raise ValueError("protocol must be legacy or clean")
    dataset = load_dataset(
        data["dataset_name"],
        revision=data["dataset_revision"],
        split=data["split"],
    )
    if selected_protocol == "legacy":
        prepared, audit = _prepare_legacy(dataset, config, tokenizer)
    else:
        prepared, audit = _prepare_clean(dataset, config, tokenizer)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = _write_dataset(
        directory,
        train=prepared["train"],
        validation=prepared["validation"],
        test=prepared["test"],
        evaluation=prepared["evaluation"],
    )
    manifest = {
        "status": "measured_data_preparation",
        "protocol": selected_protocol,
        "seed": int(config["project"]["seed"]),
        "dataset": {
            "name": data["dataset_name"],
            "revision": data["dataset_revision"],
            "split": data["split"],
            "raw_count": len(dataset),
        },
        "processing": {
            "max_sequence_length": int(data["max_sequence_length"]),
            "near_duplicate_threshold": float(data["near_duplicate_threshold"]),
            "near_duplicate_neighbors": int(data["near_duplicate_neighbors"]),
        },
        "audit": audit,
        "files": files,
    }
    write_json(manifest, directory / "data_manifest.json")
    return manifest
