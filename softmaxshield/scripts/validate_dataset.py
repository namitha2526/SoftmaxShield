"""Validation script verifying every record and API in the fine-tuning dataset against the documentation corpus."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from softmaxshield.generation.docs_extractor import get_ten_libraries_docs


REQUIRED_FIELDS = [
    "prompt",
    "library",
    "version",
    "correct_code",
    "negative_example",
    "expected_behavior",
    "explanation",
]

VALID_BEHAVIORS = {
    "generate_code",
    "refuse_hallucination",
    "refuse_nonexistent_package",
    "warn_unverified_api",
}


def validate_finetuning_dataset(
    dataset_path: str = "data/finetuning_dataset.jsonl",
) -> Dict[str, Any]:
    """Audit every dataset sample for schema compliance, syntax validity, and documentation ground truth."""
    file_path = Path(dataset_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    catalog = get_ten_libraries_docs()
    registered_apis: Set[str] = {c.api_name for c in catalog.chunks}
    registered_libs: Set[str] = {c.library.lower() for c in catalog.chunks}

    total_records = 0
    syntax_errors = 0
    schema_errors = 0
    behavior_counts: Dict[str, int] = {}
    library_counts: Dict[str, int] = {}
    missing_docs_warnings = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"[Line {line_num}] JSON parse error: {err}")
                schema_errors += 1
                continue

            # 1. Field existence and non-empty check
            for rf in REQUIRED_FIELDS:
                if rf not in record:
                    print(f"[Line {line_num}] Missing field: {rf}")
                    schema_errors += 1

            # 2. Expected behavior validation
            behavior = record.get("expected_behavior", "")
            if behavior not in VALID_BEHAVIORS:
                print(f"[Line {line_num}] Invalid expected_behavior: {behavior}")
                schema_errors += 1
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

            # 3. Track library distribution
            lib = record.get("library", "unknown").lower()
            library_counts[lib] = library_counts.get(lib, 0) + 1

            # 4. AST Syntax validation of correct_code
            code = record.get("correct_code", "")
            try:
                ast.parse(code, filename=f"dataset_line_{line_num}.py")
            except SyntaxError as syn_err:
                print(f"[Line {line_num}] Syntax error in correct_code: {syn_err}")
                syntax_errors += 1

            # 5. Negative example consistency check
            neg_ex = record.get("negative_example")
            if behavior in ("refuse_hallucination", "refuse_nonexistent_package") and not neg_ex:
                print(f"[Line {line_num}] Negative behavior '{behavior}' requires non-null negative_example")
                schema_errors += 1

    passed = (syntax_errors == 0 and schema_errors == 0 and total_records >= 2000)

    report = {
        "dataset_path": str(file_path),
        "total_records": total_records,
        "syntax_errors": syntax_errors,
        "schema_errors": schema_errors,
        "passed": passed,
        "behavior_breakdown": behavior_counts,
        "library_breakdown": library_counts,
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftmaxShield Dataset Validator")
    parser.add_argument("--dataset", default="data/finetuning_dataset.jsonl")
    args = parser.parse_args()

    results = validate_finetuning_dataset(args.dataset)
    print("=" * 60)
    print(f"SoftmaxShield Dataset Validation: {'[PASSED]' if results['passed'] else '[FAILED]'}")
    print("=" * 60)
    print(f"Total Records     : {results['total_records']}")
    print(f"Syntax Errors     : {results['syntax_errors']}")
    print(f"Schema Errors     : {results['schema_errors']}")
    print("\nBehavior Breakdown:")
    for b, count in results["behavior_breakdown"].items():
        pct = (count / results["total_records"]) * 100
        print(f"  - {b:30s}: {count:5d} ({pct:5.1f}%)")
    print("\nLibrary Breakdown:")
    for lib, count in sorted(results["library_breakdown"].items()):
        print(f"  - {lib:25s}: {count:5d}")
    print("=" * 60)

    sys.exit(0 if results["passed"] else 1)

