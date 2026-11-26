"""Tests verifying the benchmark dataset integrity, coverage, schema, and absence of data leakage."""

import json
from pathlib import Path
import pytest

from softmaxshield.evaluation.benchmark_generator import build_benchmark_dataset, BenchmarkTask


def test_benchmark_dataset_size_and_schema():
    """Verify that benchmark dataset contains at least 250 tasks with all required fields."""
    tasks = build_benchmark_dataset()
    assert len(tasks) >= 250, f"Expected at least 250 benchmark tasks, got {len(tasks)}"

    required_fields = [
        "id",
        "prompt",
        "expected_libraries",
        "expected_apis",
        "supported_versions",
        "expected_behavior",
        "expected_output",
        "difficulty",
        "risk_category",
    ]

    task_ids = set()
    for t in tasks:
        d = t.to_dict()
        for field in required_fields:
            assert field in d, f"Task {t.id} missing required field '{field}'"
        assert t.id not in task_ids, f"Duplicate task ID detected: {t.id}"
        task_ids.add(t.id)


def test_benchmark_library_coverage():
    """Verify coverage across all 10 specified Python libraries."""
    tasks = build_benchmark_dataset()
    target_libraries = {
        "numpy",
        "pandas",
        "requests",
        "flask",
        "fastapi",
        "scikit-learn",
        "torch",
        "matplotlib",
        "sqlalchemy",
        "pydantic",
    }

    covered_libs = set()
    for t in tasks:
        for lib in t.expected_libraries:
            covered_libs.add(lib)

    for target in target_libraries:
        assert target in covered_libs, f"Library '{target}' is not covered in benchmark tasks"


def test_benchmark_risk_categories():
    """Verify coverage across all 8 expected hallucination and task risk categories."""
    tasks = build_benchmark_dataset()
    expected_categories = {
        "basic_api",
        "intermediate_api",
        "advanced_niche",
        "version_specific",
        "hallucination_trap",
        "nonexistent_package",
        "invalid_kwarg",
        "unverifiable_api",
    }

    present_categories = {t.risk_category for t in tasks}
    for cat in expected_categories:
        assert cat in present_categories, f"Risk category '{cat}' missing from benchmark"


def test_zero_data_leakage():
    """Verify strictly zero prompt leakage between training dataset and benchmark dataset."""
    train_file = Path("data/finetuning_dataset.jsonl")
    bench_file = Path("data/benchmark_dataset.json")

    if not train_file.exists() or not bench_file.exists():
        pytest.skip("Dataset files not present on disk for leakage test")

    train_prompts = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.add(json.loads(line)["prompt"].strip().lower())

    with open(bench_file, "r", encoding="utf-8") as f:
        bench_tasks = json.load(f)

    overlapping = []
    for b in bench_tasks:
        b_prompt = b["prompt"].strip().lower()
        if b_prompt in train_prompts:
            overlapping.append(b["prompt"])

    assert len(overlapping) == 0, f"Detected {len(overlapping)} overlapping prompts between train and bench!"

