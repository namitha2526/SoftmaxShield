"""Tests for the comparative evaluation harness and visualization generator."""

import tempfile
from pathlib import Path
import pytest

from softmaxshield.core.config import ShieldConfig
from softmaxshield.evaluation.benchmark_generator import build_benchmark_dataset
from softmaxshield.evaluation.evaluator import BenchmarkEvaluator, format_markdown_table
from softmaxshield.evaluation.visualizer import BenchmarkVisualizer


def test_evaluator_task_slice():
    """Verify evaluation execution and metrics calculation on a benchmark task slice."""
    all_tasks = build_benchmark_dataset()
    test_slice = all_tasks[:5]

    config = ShieldConfig(offline_mode=True, strict_mode=True)
    evaluator = BenchmarkEvaluator(tasks=test_slice, config=config)

    summary_baseline, logs_baseline = evaluator.evaluate_configuration("baseline_llm", task_subset=test_slice)
    assert summary_baseline.total_tasks == 5
    assert len(logs_baseline) == 5
    assert isinstance(summary_baseline.hallucinated_package_rate, float)
    assert isinstance(summary_baseline.task_success_rate, float)

    summary_full, logs_full = evaluator.evaluate_configuration("full_softmaxshield", task_subset=test_slice)
    assert summary_full.total_tasks == 5
    assert len(logs_full) == 5

    # Markdown table formatting
    table_str = format_markdown_table([summary_baseline, summary_full])
    assert "Baseline Llm" in table_str
    assert "Full Softmaxshield" in table_str
    assert "|" in table_str


def test_visualizer_chart_generation():
    """Verify that the visualizer correctly renders and saves all 4 benchmarking PNG figures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vis = BenchmarkVisualizer(output_dir=tmpdir)

        # Mock results list
        mock_results = [
            {
                "config": "baseline_llm",
                "pkg_halluc_rate": 18.5,
                "invalid_api_rate": 3.6,
                "invalid_kwarg_rate": 5.0,
                "task_success_rate": 57.7,
                "false_positive_rate": 0.0,
                "avg_latency": 128.6,
                "abstentions": 9,
                "corrections": 0,
                "corr_success": 100.0,
                "retrieval_lat": 0.0,
                "gen_lat": 50.0,
                "verif_lat": 10.0,
                "corr_lat": 0.0,
            },
            {
                "config": "full_softmaxshield",
                "pkg_halluc_rate": 0.0,
                "invalid_api_rate": 0.0,
                "invalid_kwarg_rate": 0.0,
                "task_success_rate": 96.5,
                "false_positive_rate": 0.0,
                "avg_latency": 46.3,
                "abstentions": 60,
                "corrections": 25,
                "corr_success": 96.0,
                "retrieval_lat": 8.0,
                "gen_lat": 15.0,
                "verif_lat": 8.0,
                "corr_lat": 15.0,
            },
        ]

        p1 = vis.plot_before_after(mock_results, filename="test_before_after.png")
        p2 = vis.plot_ablation_study(mock_results, filename="test_ablation.png")
        p3 = vis.plot_latency_comparison(mock_results, filename="test_latency.png")
        p4 = vis.plot_false_positive_rate(mock_results, filename="test_fp.png")

        for p in [p1, p2, p3, p4]:
            assert p.exists()
            assert p.stat().st_size > 1000  # Figure was populated with data

