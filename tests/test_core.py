from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from medical_llm.config import config_sha256, load_config
from medical_llm.data import (
    MedicalExample,
    deterministic_eval_subset,
    exact_deduplicate,
    near_duplicate_groups,
    stable_sample_id,
    stratified_group_split,
)
from medical_llm.evaluation import bootstrap_ci
from medical_llm.external_eval import extract_decision
from medical_llm.inference import prompt_hash
from medical_llm.provenance import sha256_file
from medical_llm.safety import (
    build_blind_review_materials,
    summarize_reviewer_agreement,
)
from medical_llm.training import (
    audit_completion_loss_mask,
    pinned_adapter_config,
    prompt_completion_record,
    validated_eos_token,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        suffix = "<assistant>" if add_generation_prompt else ""
        return "|".join(f"{item['role']}:{item['content']}" for item in messages) + suffix


def test_config_preserves_effective_batch_size_and_protocol() -> None:
    config = load_config("configs/qlora_llama3_8b.yaml")
    assert config["training"]["effective_batch_size"] == 8
    assert config["project"]["seed"] == 3407
    assert config["data"]["protocol"] == "clean"
    assert config["training"]["completion_only_loss"] is True
    assert config["evaluation"]["bertscore_batch_size"] == 4
    assert len(config_sha256(config)) == 64


def test_evaluation_subset_is_deterministic_and_unique() -> None:
    records = [MedicalExample(str(i), f"q{i}", f"a{i}") for i in range(20)]
    first = deterministic_eval_subset(records, 10, 3407)
    second = deterministic_eval_subset(records, 10, 3407)
    assert first == second
    assert len({item.sample_id for item in first}) == 10


def test_sample_id_depends_on_question_and_reference() -> None:
    assert stable_sample_id("q", "a") == stable_sample_id("q", "a")
    assert stable_sample_id("q", "a") != stable_sample_id("q", "b")


def test_exact_deduplicate_normalizes_text() -> None:
    records = [
        MedicalExample("1", "What is flu?", "An infection", "information"),
        MedicalExample("2", " what IS flu ", "An infection.", "information"),
        MedicalExample("3", "What is a cold?", "Different", "information"),
    ]
    deduplicated, removed = exact_deduplicate(records)
    assert [item.sample_id for item in deduplicated] == ["1", "3"]
    assert removed == 1


def test_near_duplicate_groups_join_similar_questions() -> None:
    records = [
        MedicalExample("1", "What are the symptoms of influenza?", "a"),
        MedicalExample("2", "What are symptoms of influenza?", "b"),
        MedicalExample("3", "How is kidney failure treated?", "c"),
    ]
    groups, audit = near_duplicate_groups(records, threshold=0.75, neighbors=3)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert audit["multi_member_clusters"] == 1


def test_clean_split_is_deterministic_and_has_no_group_overlap() -> None:
    records = [
        MedicalExample(str(i), f"question {i}", f"answer {i}", f"type_{i % 2}")
        for i in range(100)
    ]
    groups = list(range(100))
    first = stratified_group_split(records, groups, seed=3407)
    second = stratified_group_split(records, groups, seed=3407)
    assert [[item.sample_id for item in split] for split in first[:3]] == [
        [item.sample_id for item in split] for split in second[:3]
    ]
    split_ids = [set(item.sample_id for item in split) for split in first[:3]]
    assert not (
        split_ids[0] & split_ids[1]
        or split_ids[0] & split_ids[2]
        or split_ids[1] & split_ids[2]
    )


def test_prompt_completion_record_separates_user_and_assistant() -> None:
    record = prompt_completion_record(MedicalExample("id", "question", "answer", "type"))
    assert record["prompt"] == [{"role": "user", "content": "question"}]
    assert record["completion"] == [{"role": "assistant", "content": "answer"}]


def test_sft_uses_a_real_tokenizer_eos_token() -> None:
    class Tokenizer:
        eos_token = "<|eot_id|>"
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<|eot_id|>": 128009}.get(token, self.unk_token_id)

    assert validated_eos_token(Tokenizer()) == "<|eot_id|>"


def test_completion_loss_mask_requires_prompt_and_answer_tokens() -> None:
    class FakeCount:
        def __init__(self, value: int) -> None:
            self.value = value

        def sum(self) -> FakeCount:
            return self

        def item(self) -> int:
            return self.value

    class FakeLabels:
        def reshape(self, _value: int) -> FakeLabels:
            return self

        def __eq__(self, value: int) -> FakeCount:
            return FakeCount(3 if value == -100 else 0)

        def __ne__(self, value: int) -> FakeCount:
            return FakeCount(2 if value == -100 else 0)

    assert audit_completion_loss_mask({"labels": FakeLabels()}) == {
        "masked_prompt_or_padding_tokens": 3,
        "supervised_answer_tokens": 2,
    }


def test_prompt_hash_is_stable_and_question_specific() -> None:
    tokenizer = FakeTokenizer()
    assert prompt_hash(tokenizer, "q") == prompt_hash(tokenizer, "q")
    assert prompt_hash(tokenizer, "q") != prompt_hash(tokenizer, "different")


def test_bootstrap_ci_contains_constant_mean_and_paired_direction() -> None:
    low, high = bootstrap_ci([0.5] * 20, samples=100, confidence=0.95, seed=3407)
    assert low == high == 0.5
    delta_low, delta_high = bootstrap_ci([0.1] * 20, samples=100, confidence=0.95, seed=3407)
    assert delta_low > 0 and delta_high > 0


def test_pubmedqa_decision_extraction() -> None:
    assert extract_decision("Yes. The evidence supports it.") == "yes"
    assert extract_decision("The answer is maybe") == "maybe"
    assert extract_decision("unclear") is None


def test_safety_portfolio_has_planned_category_counts() -> None:
    records = [
        json.loads(line)
        for line in Path("data/safety_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 50
    assert Counter(item["category"] for item in records) == {
        "urgent_care": 20,
        "medication_safety": 15,
        "insufficient_information": 15,
    }


def test_blind_review_is_deterministic_and_balanced() -> None:
    records = [
        {
            "case_id": f"case_{i}",
            "category": "urgent_care",
            "question": f"q{i}",
            "base_prediction": f"base{i}",
            "ft_prediction": f"ft{i}",
        }
        for i in range(30)
    ]
    first = build_blind_review_materials(records, seed=3407)
    second = build_blind_review_materials(records, seed=3407)
    assert first == second
    rows, key, secondary = first
    assert len(rows) == 30 and len(secondary) == 20
    assert set(key["mapping"]["case_0"].values()) == {"base", "fine_tuned"}


def test_reviewer_agreement_reports_perfect_match() -> None:
    row = {"case_id": "c1"}
    for side in ("a", "b"):
        for field in (
            "urgent_escalation",
            "specific_dosage",
            "definite_diagnosis",
            "uncertainty",
            "actionable_next_step",
            "potentially_harmful",
        ):
            row[f"{side}_{field}"] = "yes"
    report = summarize_reviewer_agreement([row], [dict(row)])
    assert report["fields"]["urgent_escalation"]["raw_agreement"] == 1.0
    assert report["fields"]["urgent_escalation"]["cohen_kappa"] == 1.0


def test_sha256_file() -> None:
    path = Path("pyproject.toml")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected


def test_pin_adapter_base_revision() -> None:
    updated = pinned_adapter_config(
        {"r": 16}, base_model="base/model", revision="immutable-sha"
    )
    assert updated["base_model_name_or_path"] == "base/model"
    assert updated["revision"] == "immutable-sha"
