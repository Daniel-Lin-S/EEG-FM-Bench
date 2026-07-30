"""Tests for YAML-driven cross-artifact result comparison outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

import result_vis
from plot.results.comparison import (
    _add_bh_q_values,
    collect_comparison_result,
    create_table_display,
)
from plot.results.config import load_result_comparison_config


ACC_METRIC = "acc"
LOSS_METRIC = "loss"
DEFAULT_Q_THRESHOLD = 0.05


def _write_artifact(
    root: Path,
    rows: list[dict[str, object]],
    status: str = "complete",
) -> None:
    """Create one synthetic complete artifact summary.

    Parameters
    ----------
    root : pathlib.Path
        Temporary artifact root.
    rows : list[dict[str, object]]
        Canonical per-seed test metric rows.
    status : str, optional
        Summary completion status, default="complete".
    """
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "test_runs.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["dataset", "seed", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (summary_dir / "summary.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )


def _rows(
    acc_values: list[float],
    loss_values: list[float],
) -> list[dict[str, object]]:
    """Build two selected metric rows for a synthetic one-dataset artifact."""
    rows = []
    for seed, value in enumerate(acc_values, start=1):
        rows.append(
            {
                "dataset": "toy",
                "seed": seed,
                "metric": ACC_METRIC,
                "value": value,
            }
        )
    for seed, value in enumerate(loss_values, start=1):
        rows.append(
            {
                "dataset": "toy",
                "seed": seed,
                "metric": LOSS_METRIC,
                "value": value,
            }
        )
    return rows


def _write_comparison_config(
    path: Path,
    artifacts: list[tuple[str, Path]],
    output_dir: Path,
    metrics: list[dict[str, str]] | None = None,
) -> None:
    """Write one temporary local comparison YAML specification."""
    selected_metrics = metrics or [
        {"name": ACC_METRIC, "direction": "maximize"},
        {"name": LOSS_METRIC, "direction": "minimize"},
    ]
    metric_names = [metric["name"] for metric in selected_metrics]
    payload = {
        "artifacts": [
            {"label": label, "root": str(root.resolve())}
            for label, root in artifacts
        ],
        "metrics": selected_metrics,
        "datasets": [{"name": "toy", "task": "binary"}],
        "task_metrics": {
            "binary": metric_names,
            "multiclass": metric_names,
        },
        "output_dir": str(output_dir.resolve()),
        "statistics": {
            "near_best_q_threshold": DEFAULT_Q_THRESHOLD,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_yaml_comparison_writes_read_only_source_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The entry point writes figures, tables, and data without source edits."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    artifact_c = tmp_path / "artifact_c"
    artifact_a_rows = _rows([0.8, 0.9], [0.2, 0.3])
    artifact_a_rows.append(
        {
            "dataset": "toy",
            "seed": 1,
            "metric": "epoch",
            "value": 7.0,
        }
    )
    _write_artifact(artifact_a, artifact_a_rows)
    _write_artifact(artifact_b, _rows([0.7, 0.8], [0.4, 0.5]))
    _write_artifact(artifact_c, _rows([0.79], [0.35]))
    source_before = {
        path: path.read_text(encoding="utf-8")
        for path in tmp_path.glob("artifact_*/summary/*")
    }
    output_dir = tmp_path / "comparison"
    config_path = tmp_path / "comparison.local.yaml"
    _write_comparison_config(
        config_path,
        [
            ("condition_a", artifact_a),
            ("condition_b", artifact_b),
            ("deterministic", artifact_c),
        ],
        output_dir,
    )
    monkeypatch.setattr(sys, "argv", ["result_vis.py", str(config_path)])

    result_vis.main()

    assert output_dir.joinpath("comparison_runs.csv").is_file()
    assert "epoch" not in output_dir.joinpath(
        "comparison_runs.csv"
    ).read_text(encoding="utf-8")
    assert output_dir.joinpath("comparison_summary.csv").is_file()
    assert output_dir.joinpath("comparison_statistics.csv").is_file()
    assert output_dir.joinpath("comparison_diagnostics.json").is_file()
    assert output_dir.joinpath("figures", "acc.png").is_file()
    assert output_dir.joinpath("figures", "loss.png").is_file()
    assert output_dir.joinpath("tables", "performance.md").is_file()
    assert output_dir.joinpath("tables", "performance.png").is_file()
    markdown = output_dir.joinpath("tables", "performance.md").read_text(
        encoding="utf-8",
    )
    assert '<td rowspan="2" align="center"' in markdown
    assert 'style="vertical-align: middle;">toy</td>' in markdown
    assert "<strong>0.8500 ± 0.0707</strong>" in markdown
    assert "0.7500 ± 0.0707†" in markdown
    assert "0.7900" in markdown
    assert "not statistically distinguishable" in markdown
    manifest = output_dir.joinpath("comparison_manifest.json")
    assert str(artifact_a.resolve()) not in manifest.read_text(encoding="utf-8")
    diagnostics = json.loads(
        output_dir.joinpath("comparison_diagnostics.json").read_text(
            encoding="utf-8",
        )
    )
    assert diagnostics["diagnostic_count"] == 4
    with output_dir.joinpath("comparison_statistics.csv").open(
        newline="",
        encoding="utf-8",
    ) as file_obj:
        statistics = list(csv.DictReader(file_obj))
    assert all(
        float(row["p_value"]) == pytest.approx(float(row["q_value"]))
        for row in statistics
    )
    assert float(statistics[0]["cliffs_delta"]) == pytest.approx(0.75)
    assert all(
        path.read_text(encoding="utf-8") == contents
        for path, contents in source_before.items()
    )
    assert str(output_dir.resolve()) in capsys.readouterr().out


def test_table_bolds_exact_best_ties_and_marks_near_best(
    tmp_path: Path,
) -> None:
    """Exact best means are bold while near-best means get daggers."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    artifact_c = tmp_path / "artifact_c"
    _write_artifact(artifact_a, _rows([0.8, 0.9], [0.2, 0.3]))
    _write_artifact(artifact_b, _rows([0.8, 0.9], [0.4, 0.5]))
    _write_artifact(artifact_c, _rows([0.7, 0.8], [0.6, 0.7]))
    config_path = tmp_path / "comparison.local.yaml"
    _write_comparison_config(
        config_path,
        [("a", artifact_a), ("b", artifact_b), ("c", artifact_c)],
        tmp_path / "output",
    )

    config = load_result_comparison_config(config_path)
    display = create_table_display(collect_comparison_result(config), config)

    assert (0, 2) in display.bold_cells
    assert (0, 3) in display.bold_cells
    assert display.rows[0][4].endswith("†")


def test_single_seed_comparison_omits_statistics(tmp_path: Path) -> None:
    """Deterministic comparisons render without an empty statistics CSV."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    _write_artifact(artifact_a, _rows([0.8], [0.2]))
    _write_artifact(artifact_b, _rows([0.7], [0.3]))
    output_dir = tmp_path / "output"
    config_path = tmp_path / "comparison.local.yaml"
    _write_comparison_config(
        config_path,
        [("a", artifact_a), ("b", artifact_b)],
        output_dir,
    )
    original_argv = sys.argv
    sys.argv = ["result_vis.py", str(config_path)]
    try:
        result_vis.main()
    finally:
        sys.argv = original_argv

    assert not output_dir.joinpath("comparison_statistics.csv").exists()
    diagnostics = json.loads(
        output_dir.joinpath("comparison_diagnostics.json").read_text(
            encoding="utf-8",
        )
    )
    assert diagnostics["diagnostic_count"] == 2


def test_refresh_removes_stale_managed_figure_and_keeps_other_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshes remove only workflow-owned stale files."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    _write_artifact(artifact_a, _rows([0.8, 0.9], [0.2, 0.3]))
    _write_artifact(artifact_b, _rows([0.7, 0.8], [0.4, 0.5]))
    output_dir = tmp_path / "output"
    config_path = tmp_path / "comparison.local.yaml"
    _write_comparison_config(
        config_path,
        [("a", artifact_a), ("b", artifact_b)],
        output_dir,
    )
    monkeypatch.setattr(sys, "argv", ["result_vis.py", str(config_path)])
    result_vis.main()
    output_dir.joinpath("keep.txt").write_text("keep", encoding="utf-8")
    _write_comparison_config(
        config_path,
        [("a", artifact_a), ("b", artifact_b)],
        output_dir,
        metrics=[{"name": ACC_METRIC, "direction": "maximize"}],
    )

    result_vis.main()

    assert output_dir.joinpath("figures", "acc.png").is_file()
    assert not output_dir.joinpath("figures", "loss.png").exists()
    assert output_dir.joinpath("keep.txt").read_text(encoding="utf-8") == "keep"


def test_config_rejects_epoch_and_output_inside_source(tmp_path: Path) -> None:
    """Epoch and source-descendant outputs cannot enter a comparison run."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    _write_artifact(artifact_a, _rows([0.8, 0.9], [0.2, 0.3]))
    _write_artifact(artifact_b, _rows([0.7, 0.8], [0.4, 0.5]))
    epoch_config = tmp_path / "epoch.local.yaml"
    _write_comparison_config(
        epoch_config,
        [("a", artifact_a), ("b", artifact_b)],
        tmp_path / "output",
        metrics=[{"name": "epoch", "direction": "maximize"}],
    )

    with pytest.raises(ValueError, match="Metric 'epoch'"):
        load_result_comparison_config(epoch_config)

    unsafe_config = tmp_path / "unsafe.local.yaml"
    _write_comparison_config(
        unsafe_config,
        [("a", artifact_a), ("b", artifact_b)],
        artifact_a / "comparison",
    )
    config = load_result_comparison_config(unsafe_config)
    with pytest.raises(ValueError, match="must not be an artifact root"):
        collect_comparison_result(config)


def test_incomplete_and_non_finite_sources_fail_clearly(tmp_path: Path) -> None:
    """Incomplete statuses and non-finite selected values are rejected."""
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    _write_artifact(artifact_a, _rows([0.8, 0.9], [0.2, 0.3]), status="partial")
    _write_artifact(artifact_b, _rows([0.7, 0.8], [0.4, 0.5]))
    config_path = tmp_path / "comparison.local.yaml"
    _write_comparison_config(
        config_path,
        [("a", artifact_a), ("b", artifact_b)],
        tmp_path / "output",
        metrics=[{"name": ACC_METRIC, "direction": "maximize"}],
    )
    config = load_result_comparison_config(config_path)

    with pytest.raises(ValueError, match="complete summary status"):
        collect_comparison_result(config)

    _write_artifact(artifact_a, _rows([float("nan"), 0.9], [0.2, 0.3]))
    with pytest.raises(ValueError, match="Expected finite value"):
        collect_comparison_result(config)



def test_bh_adjustment_is_applied_per_metric() -> None:
    """Benjamini-Hochberg adjustment preserves each metric test family."""
    statistics = [
        {"metric": ACC_METRIC, "p_value": 0.01},
        {"metric": ACC_METRIC, "p_value": 0.02},
        {"metric": ACC_METRIC, "p_value": 0.8},
        {"metric": LOSS_METRIC, "p_value": 0.04},
    ]

    _add_bh_q_values(statistics)

    assert statistics[0]["q_value"] == pytest.approx(0.03)
    assert statistics[1]["q_value"] == pytest.approx(0.03)
    assert statistics[2]["q_value"] == pytest.approx(0.8)
    assert statistics[3]["q_value"] == pytest.approx(0.04)
