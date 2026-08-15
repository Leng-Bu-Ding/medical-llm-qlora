from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MedicalExample:
    sample_id: str
    question: str
    reference: str
    qtype: str = "unknown"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def stable_sample_id(question: str, reference: str) -> str:
    return hashlib.sha256(f"{question}\n{reference}".encode()).hexdigest()[:16]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def to_example(
    record: dict[str, Any],
    question_field: str,
    answer_field: str,
    qtype_field: str = "qtype",
) -> MedicalExample:
    question = record.get(question_field)
    answer = record.get(answer_field)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("medical example has no valid question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("medical example has no valid answer")
    qtype = record.get(qtype_field, "unknown")
    if not isinstance(qtype, str) or not qtype.strip():
        qtype = "unknown"
    return MedicalExample(
        stable_sample_id(question.strip(), answer.strip()),
        question.strip(),
        answer.strip(),
        qtype.strip(),
    )


def format_chat(example: MedicalExample, tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.reference},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def deterministic_eval_subset(
    records: Sequence[MedicalExample], size: int, seed: int
) -> list[MedicalExample]:
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


def exact_deduplicate(records: Sequence[MedicalExample]) -> tuple[list[MedicalExample], int]:
    kept: list[MedicalExample] = []
    seen: set[tuple[str, str]] = set()
    for item in records:
        key = (normalize_text(item.question), normalize_text(item.reference))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept, len(records) - len(kept)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def near_duplicate_groups(
    records: Sequence[MedicalExample], *, threshold: float, neighbors: int
) -> tuple[list[int], dict[str, Any]]:
    if not records:
        return [], {"cluster_count": 0, "multi_member_clusters": 0, "linked_pairs": 0}
    if len(records) == 1:
        return [0], {"cluster_count": 1, "multi_member_clusters": 0, "linked_pairs": 0}

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    normalized = [normalize_text(item.question) for item in records]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32
    )
    matrix = vectorizer.fit_transform(normalized)
    count = min(max(2, neighbors), len(records))
    distances, indices = NearestNeighbors(
        n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=1
    ).fit(matrix).kneighbors(matrix)
    union_find = _UnionFind(len(records))
    linked_pairs = 0
    for left, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, right in zip(row_distances, row_indices, strict=True):
            right = int(right)
            if left >= right:
                continue
            if 1.0 - float(distance) >= threshold:
                union_find.union(left, right)
                linked_pairs += 1
    roots = [union_find.find(index) for index in range(len(records))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    groups = [root_to_group[root] for root in roots]
    counts = Counter(groups)
    return groups, {
        "cluster_count": len(counts),
        "multi_member_clusters": sum(value > 1 for value in counts.values()),
        "largest_cluster": max(counts.values()),
        "linked_pairs": linked_pairs,
        "threshold": threshold,
        "neighbors_considered": count,
    }


def stratified_group_split(
    records: Sequence[MedicalExample],
    groups: Sequence[int],
    *,
    seed: int,
) -> tuple[list[MedicalExample], list[MedicalExample], list[MedicalExample], list[int]]:
    if len(records) != len(groups) or len(records) < 10:
        raise ValueError("clean split requires matching records/groups and at least 10 records")

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray([item.qtype for item in records])
    group_array = np.asarray(groups)
    fold_assignment = np.full(len(records), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    dummy = np.zeros(len(records), dtype=int)
    for fold, (_, fold_indices) in enumerate(splitter.split(dummy, labels, group_array)):
        fold_assignment[fold_indices] = fold
    if np.any(fold_assignment < 0):
        raise RuntimeError("some records were not assigned to a clean split fold")

    train = [item for item, fold in zip(records, fold_assignment, strict=True) if fold >= 2]
    validation = [item for item, fold in zip(records, fold_assignment, strict=True) if fold == 1]
    test = [item for item, fold in zip(records, fold_assignment, strict=True) if fold == 0]
    return train, validation, test, fold_assignment.tolist()


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
    return [
        MedicalExample(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
