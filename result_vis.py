#!/usr/bin/env python3
"""Compare completed test summaries from multiple artifact directories.

Input is one local YAML file with labelled absolute artifact roots, explicit
metric directions, a separate output directory, and a q-value threshold. The
workflow reads source artifacts only and writes normalized data, statistics,
charts, and a copyable Markdown plus PNG summary table under ``output_dir``.

Usage
-----
python result_vis.py <comparison.local.yaml>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot.results.comparison import (
    ManagedOutput,
    build_manifest_metadata,
    collect_comparison_result,
    create_table_display,
    write_result_data,
)
from plot.results.config import load_result_comparison_config
from plot.results.figure_visualizer import save_metric_figures
from plot.results.table_visualizer import save_markdown_table, save_table_image


FIGURES_DIRECTORY = "figures"
TABLE_MARKDOWN_FILENAME = "tables/performance.md"
TABLE_IMAGE_FILENAME = "tables/performance.png"


def main() -> None:
    """Run one YAML-driven cross-artifact result comparison."""
    parser = argparse.ArgumentParser(
        description="Compare completed test summaries from artifact roots.",
    )
    parser.add_argument(
        "comparison_config",
        type=Path,
        help="Path to a local YAML comparison specification.",
    )
    arguments = parser.parse_args()
    config = load_result_comparison_config(arguments.comparison_config)
    result = collect_comparison_result(config)
    output = ManagedOutput(config.output_dir)
    write_result_data(result, output)
    figures_dir = output.output_dir / FIGURES_DIRECTORY
    figure_paths = save_metric_figures(
        result.raw_rows,
        result.statistic_rows,
        [
            metric.name
            for metric in config.metrics
            if any(row["metric"] == metric.name for row in result.raw_rows)
        ],
        [artifact.label for artifact in config.artifacts],
        figures_dir,
        {
            dataset.name: dataset.display_name or dataset.name
            for dataset in config.datasets
        },
        {
            metric.name: metric.display_name or metric.name
            for metric in config.metrics
        },
        config.plot.show_individual_points,
        config.plot.naive_artifact,
    )
    for figure_path in figure_paths:
        output.path(
            figure_path.relative_to(output.output_dir).as_posix()
        )
    display = create_table_display(result, config)
    save_markdown_table(display, output.path(TABLE_MARKDOWN_FILENAME))
    save_table_image(display, output.path(TABLE_IMAGE_FILENAME))
    output.finalize(build_manifest_metadata(result, config))
    print(f"Saved comparison outputs to {output.output_dir}")


if __name__ == "__main__":
    main()
