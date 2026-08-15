from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from medical_llm.data import (
    MedicalExample,
    deterministic_eval_subset,
    stable_sample_id,
    write_jsonl,
)
from medical_llm.evaluation import bootstrap_ci
from medical_llm.provenance import sha256_file, write_json


def _pubmedqa_prompt(row: dict[str, Any]) -> str:
    contexts = row.get("context", {}).get("contexts", [])
    context = " ".join(str(item).strip() for item in contexts if str(item).strip())
    return (
        "Based only on the biomedical abstract below, answer the research question. "
        "Start with exactly one label: yes, no, or maybe.\n\n"
        f"Abstract: {context}\n\nQuestion: {str(row['question']).strip()}"
    )


def prepare_pubmedqa(
    config: dict[str, Any], tokenizer: Any, output_path: str | Path
) -> dict[str, Any]:
    from datasets import load_dataset

    settings = config["external_evaluation"]
    dataset = load_dataset(
        settings["dataset_name"],
        settings["subset"],
        revision=settings["dataset_revision"],
        split=settings["split"],
    )
    eligible: list[MedicalExample] = []
    for row in dataset:
        prompt = _pubmedqa_prompt(row)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) > int(
            settings["max_sequence_length"]
        ):
            continue
        reference = str(row["final_decision"]).strip().casefold()
        if reference not in {"yes", "no", "maybe"}:
            continue
        eligible.append(
            MedicalExample(stable_sample_id(prompt, reference), prompt, reference, "pubmedqa")
        )
    selected = deterministic_eval_subset(
        eligible,
        min(int(settings["evaluation_size"]), len(eligible)),
        int(config["project"]["seed"]),
    )
    output = Path(output_path)
    write_jsonl(selected, output)
    manifest = {
        "status": "measured_data_preparation",
        "dataset": settings["dataset_name"],
        "subset": settings["subset"],
        "revision": settings["dataset_revision"],
        "raw_count": len(dataset),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "sha256": sha256_file(output),
    }
    write_json(manifest, output.with_name("pubmedqa_manifest.json"))
    return manifest


def extract_decision(text: str) -> str | None:
    match = re.search(r"\b(yes|no|maybe)\b", text.casefold())
    return match.group(1) if match else None


def evaluate_pubmedqa(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("PubMedQA predictions cannot be empty")
    base_scores: list[float] = []
    ft_scores: list[float] = []
    invalid = {"base": 0, "fine_tuned": 0}
    for item in records:
        reference = str(item["reference"]).casefold()
        base = extract_decision(str(item["base_prediction"]))
        fine_tuned = extract_decision(str(item["ft_prediction"]))
        invalid["base"] += base is None
        invalid["fine_tuned"] += fine_tuned is None
        base_scores.append(float(base == reference))
        ft_scores.append(float(fine_tuned == reference))
    deltas = [fine_tuned - base for base, fine_tuned in zip(base_scores, ft_scores, strict=True)]
    settings = config["evaluation"]
    low, high = bootstrap_ci(
        deltas,
        samples=int(settings["bootstrap_samples"]),
        confidence=float(settings["confidence_level"]),
        seed=int(settings["bootstrap_seed"]),
    )
    return {
        "status": "measured_external_transfer_check",
        "warning": "Capability-transfer check, not a clinical validation.",
        "sample_count": len(records),
        "base_accuracy": sum(base_scores) / len(base_scores),
        "fine_tuned_accuracy": sum(ft_scores) / len(ft_scores),
        "paired_accuracy_delta": sum(deltas) / len(deltas),
        "paired_delta_ci_low": low,
        "paired_delta_ci_high": high,
        "improved": sum(delta > 0 for delta in deltas),
        "regressed": sum(delta < 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "invalid_label_count": invalid,
    }


def read_records(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
