from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap_ci(values: Sequence[float], *, samples: int, confidence: float, seed: int) -> tuple[float, float]:
    if not values or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [generator.choice(array, size=len(array), replace=True).mean() for _ in range(samples)]
    )
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def evaluate_predictions(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    if not records:
        raise ValueError("prediction records cannot be empty")
    references = [str(item["reference"]) for item in records]
    systems = ("base_prediction", "ft_prediction")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_system: dict[str, dict[str, list[float]]] = {}
    for system in systems:
        predictions = [str(item[system]) for item in records]
        metrics = {"rouge1": [], "rouge2": [], "rougeL": []}
        for prediction, reference in zip(predictions, references, strict=True):
            scores = scorer.score(reference, prediction)
            for name in metrics:
                metrics[name].append(scores[name].fmeasure)
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
    summary = {}
    for system, metrics in per_system.items():
        summary[system] = {}
        for name, values in metrics.items():
            low, high = bootstrap_ci(
                values,
                samples=int(bootstrap["bootstrap_samples"]),
                confidence=float(bootstrap["confidence_level"]),
                seed=int(bootstrap["bootstrap_seed"]),
            )
            summary[system][name] = {
                "mean": round(float(np.mean(values)), 8),
                "ci_low": round(low, 8),
                "ci_high": round(high, 8),
            }
    deltas = np.asarray(per_system["ft_prediction"]["rougeL"]) - np.asarray(
        per_system["base_prediction"]["rougeL"]
    )
    summary["comparison"] = {
        "improved": int(np.sum(deltas > 1e-12)),
        "regressed": int(np.sum(deltas < -1e-12)),
        "tied": int(np.sum(np.abs(deltas) <= 1e-12)),
    }
    return {"sample_count": len(records), "metrics": summary}


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
