from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap_ci(
    values: Sequence[float], *, samples: int, confidence: float, seed: int
) -> tuple[float, float]:
    if not values or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [generator.choice(array, size=len(array), replace=True).mean() for _ in range(samples)]
    )
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _repeated_ngram_rate(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+", text.casefold())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _integrity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_ids = [str(item["sample_id"]) for item in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prediction records contain duplicate sample_id values")
    missing = [
        item["sample_id"]
        for item in records
        if not item.get("prompt_hash") or not item.get("generation_hash")
    ]
    if missing:
        raise ValueError(f"prediction records lack prompt/generation hashes: {missing[:5]}")
    generation_hashes = {str(item["generation_hash"]) for item in records}
    if len(generation_hashes) != 1:
        raise ValueError("paired predictions used more than one generation configuration")
    return {
        "unique_sample_ids": len(sample_ids),
        "generation_hash": next(iter(generation_hashes)),
        "paired_fields_present": True,
    }


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    if not records:
        raise ValueError("prediction records cannot be empty")
    integrity = _integrity(records)
    references = [str(item["reference"]) for item in records]
    systems = ("base_prediction", "ft_prediction")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_system: dict[str, dict[str, list[float]]] = {}
    for system in systems:
        predictions = [str(item[system]) for item in records]
        metrics: dict[str, list[float]] = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "output_tokens": [],
            "repeated_4gram_rate": [],
        }
        for prediction, reference in zip(predictions, references, strict=True):
            scores = scorer.score(reference, prediction)
            for name in ("rouge1", "rouge2", "rougeL"):
                metrics[name].append(scores[name].fmeasure)
            metrics["output_tokens"].append(float(len(re.findall(r"\w+", prediction))))
            metrics["repeated_4gram_rate"].append(_repeated_ngram_rate(prediction))
        _, _, bert_f1 = bert_score(
            predictions,
            references,
            model_type=config["evaluation"]["bertscore_model"],
            lang="en",
            verbose=True,
        )
        metrics["bertscore_f1"] = bert_f1.cpu().numpy().tolist()
        per_system[system] = metrics

    bootstrap = config["evaluation"]
    bootstrap_samples = int(bootstrap["bootstrap_samples"])
    confidence = float(bootstrap["confidence_level"])
    seed = int(bootstrap["bootstrap_seed"])
    summary: dict[str, Any] = {}
    for system, metrics in per_system.items():
        summary[system] = {}
        for name, values in metrics.items():
            low, high = bootstrap_ci(
                values,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
            )
            summary[system][name] = {
                "mean": round(float(np.mean(values)), 8),
                "ci_low": round(low, 8),
                "ci_high": round(high, 8),
            }

    paired: dict[str, Any] = {}
    for name in per_system["base_prediction"]:
        deltas = np.asarray(per_system["ft_prediction"][name]) - np.asarray(
            per_system["base_prediction"][name]
        )
        low, high = bootstrap_ci(
            deltas.tolist(),
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
        paired[name] = {
            "mean_delta": round(float(deltas.mean()), 8),
            "ci_low": round(low, 8),
            "ci_high": round(high, 8),
            "direction": "positive" if low > 0 else "negative" if high < 0 else "inconclusive",
        }

    rouge_l_deltas = np.asarray(per_system["ft_prediction"]["rougeL"]) - np.asarray(
        per_system["base_prediction"]["rougeL"]
    )
    summary["paired_delta"] = paired
    summary["comparison"] = {
        "improved": int(np.sum(rouge_l_deltas > 1e-12)),
        "regressed": int(np.sum(rouge_l_deltas < -1e-12)),
        "tied": int(np.sum(np.abs(rouge_l_deltas) <= 1e-12)),
    }

    error_count = min(int(config["evaluation"].get("error_case_count", 20)), len(records))
    ordered = np.argsort(rouge_l_deltas)
    error_cases = []
    for label, indices in (
        ("largest_regressions", ordered[:error_count]),
        ("largest_improvements", ordered[-error_count:][::-1]),
    ):
        for index in indices:
            item = records[int(index)]
            error_cases.append(
                {
                    "bucket": label,
                    "sample_id": item["sample_id"],
                    "qtype": item.get("qtype", "unknown"),
                    "question": item["question"],
                    "reference": item["reference"],
                    "base_prediction": item["base_prediction"],
                    "ft_prediction": item["ft_prediction"],
                    "rougeL_delta": round(float(rouge_l_deltas[int(index)]), 8),
                }
            )
    return {
        "status": "measured",
        "sample_count": len(records),
        "integrity": integrity,
        "metrics": summary,
        "error_cases": error_cases,
    }


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
