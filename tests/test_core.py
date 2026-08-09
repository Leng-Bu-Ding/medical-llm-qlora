from __future__ import annotations

from medical_llm.config import load_config
from medical_llm.data import MedicalExample, deterministic_eval_subset, stable_sample_id
from medical_llm.evaluation import bootstrap_ci
import json
from collections import Counter
from pathlib import Path


def test_config_preserves_original_effective_batch_size() -> None:
    config = load_config("configs/qlora_llama3_8b.yaml")
    assert config["training"]["effective_batch_size"] == 8
    assert config["project"]["seed"] == 3407


def test_evaluation_subset_is_deterministic_and_unique() -> None:
    records = [MedicalExample(str(i), f"q{i}", f"a{i}") for i in range(20)]
    first = deterministic_eval_subset(records, 10, 3407)
    second = deterministic_eval_subset(records, 10, 3407)
    assert first == second
    assert len({item.sample_id for item in first}) == 10


def test_sample_id_depends_on_question_and_reference() -> None:
    assert stable_sample_id("q", "a") == stable_sample_id("q", "a")
    assert stable_sample_id("q", "a") != stable_sample_id("q", "b")


def test_bootstrap_ci_contains_constant_mean() -> None:
    low, high = bootstrap_ci([0.5] * 20, samples=100, confidence=0.95, seed=3407)
    assert low == high == 0.5


def test_safety_portfolio_has_planned_category_counts() -> None:
    records = [json.loads(line) for line in Path("data/safety_cases.jsonl").read_text().splitlines()]
    assert len(records) == 50
    assert Counter(item["category"] for item in records) == {
        "urgent_care": 20,
        "medication_safety": 15,
        "insufficient_information": 15,
    }
