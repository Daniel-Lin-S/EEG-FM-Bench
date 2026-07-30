"""Render prepared cross-artifact summary tables.

Inputs are display-ready headers, cells, and bold-cell coordinates. Outputs
are a portable Markdown table and a PNG image. This module intentionally does
not load artifacts, aggregate metrics, or perform statistical tests.
"""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MIN_FIGURE_WIDTH = 8.0
MIN_FIGURE_HEIGHT = 3.0
WIDTH_PER_COLUMN = 1.8
HEIGHT_PER_ROW = 0.42
TABLE_FONT_SIZE = 8.0
HEADER_COLOR = "#d9d9d9"
DATASET_COLOR = "#eaf2f8"
ROW_COLOR = "#f5f5f5"
GRID_COLOR = "#c7c7c7"
TABLE_BBOX = (0.0, 0.08, 1.0, 0.82)
HEADER_WRAP_WIDTH = 12
MIN_COLUMN_WEIGHT = 1.0
DATASET_WRAP_WIDTH = 16
METRIC_WRAP_WIDTH = 14
CHARACTERS_PER_COLUMN_WEIGHT = 12
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
    lines = ["<table>", "  <thead>", "    <tr>"]
    lines.extend(
        f"      <th>{html.escape(header)}</th>" for header in display.headers
    )
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for start_row, end_row in _dataset_row_groups(display.rows):
        for row_index in range(start_row, end_row):
            row = display.rows[row_index]
            lines.append("    <tr>")
            if row_index == start_row:
                rowspan = end_row - start_row
                lines.append(
                    "      <td rowspan=\""
                    f"{rowspan}\" align=\"center\" "
                    "style=\"vertical-align: middle;\">"
                    f"{html.escape(row[0])}</td>"
                )
            for column_index, value in enumerate(row[1:], start=1):
                lines.append(
                    "      <td align=\"center\">"
                    f"{_format_markdown_cell(display, row_index, column_index)}"
                    "</td>"
                )
            lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>", "", FOOTNOTE])
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
    header_lines = _header_line_count(display.headers)
    height = max(
        MIN_FIGURE_HEIGHT,
        HEIGHT_PER_ROW * (row_count + header_lines + 3),
    )
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    title = textwrap.fill("Cross-artifact test performance summary", 60)
    axis.set_title(title, fontsize=TABLE_FONT_SIZE + 3.0, pad=18.0)
    _draw_table(axis, display)
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


def _dataset_row_groups(rows: list[list[str]]) -> list[tuple[int, int]]:
    """Return contiguous row ranges sharing one dataset name."""
    groups: list[tuple[int, int]] = []
    start_row = 0
    while start_row < len(rows):
        dataset_name = rows[start_row][0]
        end_row = start_row + 1
        while end_row < len(rows) and rows[end_row][0] == dataset_name:
            end_row += 1
        groups.append((start_row, end_row))
        start_row = end_row
    return groups


def _format_markdown_cell(
    display: TableDisplay,
    row_index: int,
    column_index: int,
) -> str:
    """Escape and optionally emphasize one HTML-table value."""
    value = html.escape(display.rows[row_index][column_index])
    if (row_index, column_index) in display.bold_cells:
        return f"<strong>{value}</strong>"
    return value


def _draw_table(axis: plt.Axes, display: TableDisplay) -> None:
    """Draw a table with vertically merged, centered dataset cells."""
    left, bottom, _, table_height = TABLE_BBOX
    column_widths = _column_widths(display)
    header_lines = _header_line_count(display.headers)
    body_row_height = table_height / (len(display.rows) + header_lines)
    header_height = body_row_height * header_lines
    header_bottom = bottom + table_height - header_height
    x_positions = [left]
    for width in column_widths:
        x_positions.append(x_positions[-1] + width)

    for column_index, header in enumerate(display.headers):
        _draw_cell(
            axis,
            x_positions[column_index],
            header_bottom,
            column_widths[column_index],
            header_height,
            textwrap.fill(header, HEADER_WRAP_WIDTH),
            HEADER_COLOR,
            bold=True,
        )

    for start_row, end_row in _dataset_row_groups(display.rows):
        y_position = (
            bottom + table_height - header_height - end_row * body_row_height
        )
        _draw_cell(
            axis,
            x_positions[0],
            y_position,
            column_widths[0],
            (end_row - start_row) * body_row_height,
            textwrap.fill(display.rows[start_row][0], DATASET_WRAP_WIDTH),
            DATASET_COLOR,
        )
        for row_index in range(start_row, end_row):
            y_position = (
                bottom
                + table_height
                - header_height - (row_index + 1) * body_row_height
            )
            row_color = ROW_COLOR if row_index % 2 else "white"
            for column_index, value in enumerate(
                display.rows[row_index][1:],
                start=1,
            ):
                _draw_cell(
                    axis,
                    x_positions[column_index],
                    y_position,
                    column_widths[column_index],
                    body_row_height,
                    (
                        textwrap.fill(value, METRIC_WRAP_WIDTH)
                        if column_index == 1 else value
                    ),
                    row_color,
                    bold=(row_index, column_index) in display.bold_cells,
                )


def _column_widths(display: TableDisplay) -> list[float]:
    """Allocate table width from the longest header or body-cell value."""
    longest_values = [len(header) for header in display.headers]
    for row in display.rows:
        for column_index, value in enumerate(row):
            longest_values[column_index] = max(
                longest_values[column_index],
                len(value),
            )
    weights = [
        max(MIN_COLUMN_WEIGHT, value / CHARACTERS_PER_COLUMN_WEIGHT)
        for value in longest_values
    ]
    weight_total = sum(weights)
    return [TABLE_BBOX[2] * weight / weight_total for weight in weights]


def _header_line_count(headers: list[str]) -> int:
    """Return the number of lines required by the tallest table heading."""
    return max(
        len(textwrap.wrap(header, HEADER_WRAP_WIDTH))
        for header in headers
    )


def _draw_cell(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    width: float,
    height: float,
    value: str,
    color: str,
    bold: bool = False,
) -> None:
    """Draw one centered table cell in axes coordinates."""
    axis.add_patch(
        Rectangle(
            (x_position, y_position),
            width,
            height,
            facecolor=color,
            edgecolor=GRID_COLOR,
            linewidth=0.6,
        )
    )
    axis.text(
        x_position + width / 2.0,
        y_position + height / 2.0,
        value,
        ha="center",
        va="center",
        fontsize=TABLE_FONT_SIZE,
        fontweight="bold" if bold else "normal",
        wrap=True,
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
