"""Render prepared cross-artifact summary tables.

Inputs are display-ready headers, cells, and bold-cell coordinates. Outputs
are a portable Markdown table and a PNG image. This module intentionally does
not load artifacts, aggregate metrics, or perform statistical tests.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


MIN_FIGURE_WIDTH = 8.0
MIN_FIGURE_HEIGHT = 3.0
WIDTH_PER_COLUMN = 1.8
HEIGHT_PER_ROW = 0.42
TABLE_FONT_SIZE = 8.0
HEADER_COLOR = "#d9d9d9"
ROW_COLOR = "#f5f5f5"
FOOTNOTE = (
    "† Lower-ranked result was not statistically distinguishable from the "
    "best mean (BH-adjusted q ≥ configured threshold)."
)


@dataclass(frozen=True)
class TableDisplay:
    """Prepared values for cross-artifact table rendering.

    Parameters
    ----------
    headers : list[str]
        Ordered table column names.
    rows : list[list[str]]
        Display-ready cell values excluding the header row.
    bold_cells : frozenset[tuple[int, int]]
        Zero-based ``(row, column)`` body coordinates rendered in bold.
    """

    headers: list[str]
    rows: list[list[str]]
    bold_cells: frozenset[tuple[int, int]]


def save_markdown_table(display: TableDisplay, path: Path) -> None:
    """Save a copyable Markdown table with bold best-result cells.

    Parameters
    ----------
    display : TableDisplay
        Fully prepared table contents.
    path : pathlib.Path
        Destination Markdown file.
    """
    _validate_display(display)
    lines = [
        "| " + " | ".join(display.headers) + " |",
        "| " + " | ".join(["---"] * len(display.headers)) + " |",
    ]
    for row_index, row in enumerate(display.rows):
        formatted = []
        for column_index, value in enumerate(row):
            if (row_index, column_index) in display.bold_cells:
                formatted.append(f"**{value}**")
            else:
                formatted.append(value)
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend(["", FOOTNOTE])
    _atomic_write_text(path, "\n".join(lines) + "\n")


def save_table_image(display: TableDisplay, path: Path) -> None:
    """Render a prepared cross-artifact table to a PNG image.

    Parameters
    ----------
    display : TableDisplay
        Fully prepared table contents.
    path : pathlib.Path
        Destination PNG file.
    """
    _validate_display(display)
    column_count = len(display.headers)
    row_count = len(display.rows)
    width = max(MIN_FIGURE_WIDTH, WIDTH_PER_COLUMN * column_count)
    height = max(MIN_FIGURE_HEIGHT, HEIGHT_PER_ROW * (row_count + 4))
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    title = textwrap.fill("Cross-artifact test performance summary", 60)
    axis.set_title(title, fontsize=TABLE_FONT_SIZE + 3.0, pad=18.0)
    table = axis.table(
        cellText=display.rows,
        colLabels=display.headers,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_fontweight("bold")
        elif row_index % 2 == 0:
            cell.set_facecolor(ROW_COLOR)
        body_index = row_index - 1
        if (body_index, column_index) in display.bold_cells:
            cell.get_text().set_fontweight("bold")
    figure.text(
        0.01,
        0.015,
        textwrap.fill(FOOTNOTE, 125),
        fontsize=TABLE_FONT_SIZE - 1.0,
        ha="left",
        va="bottom",
    )
    _atomic_save_figure(figure, path)
    plt.close(figure)


def _validate_display(display: TableDisplay) -> None:
    """Validate table dimensions before writing a table artifact."""
    if not display.headers:
        raise ValueError("Expected non-empty table headers.")
    if not display.rows:
        raise ValueError("Expected at least one table row.")
    expected_columns = len(display.headers)
    for row_index, row in enumerate(display.rows):
        if len(row) != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} cells in table row "
                f"{row_index}, but got {len(row)}."
            )


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    """Atomically save one PNG figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=200, bbox_inches="tight")
    temporary_path.replace(path)
