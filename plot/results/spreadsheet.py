"""Read configured XLSX cells without an optional Excel package.

The reader exposes worksheets as header-keyed rows. It supports common XLSX
cell types and raises explicit errors for malformed workbook structures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


MAIN_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
NAMESPACES = {
    "main": MAIN_NAMESPACE,
    "rel": RELATIONSHIP_NAMESPACE,
    "package": PACKAGE_RELATIONSHIP_NAMESPACE,
}
CELL_REFERENCE_PATTERN = re.compile(r"([A-Z]+)")


def read_xlsx_rows(
    path: Path,
    sheet: str | int,
    header_row: int,
) -> list[dict[str, str]]:
    """Return non-empty XLSX rows keyed by their configured header row.

    Parameters
    ----------
    path : pathlib.Path
        Existing XLSX workbook path.
    sheet : str or int
        Zero-based sheet index or worksheet name.
    header_row : int
        One-based row number containing unique column names.

    Returns
    -------
    list[dict[str, str]]
        Data rows after ``header_row``. Empty spreadsheet rows are excluded.
    """
    if not path.is_file():
        raise ValueError(
            f"Spreadsheet source does not exist: {path.resolve()}."
        )
    try:
        with ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            worksheet_path = _resolve_worksheet_path(archive, sheet)
            cells_by_row = _read_worksheet_cells(
                archive,
                worksheet_path,
                shared_strings,
            )
    except BadZipFile as exc:
        raise ValueError(
            f"Expected an XLSX workbook at {path.resolve()}, but it is invalid."
        ) from exc
    headers = cells_by_row.get(header_row)
    if not headers:
        raise ValueError(
            f"Spreadsheet header row {header_row} is empty at {path.resolve()}."
        )
    header_by_column = {
        column: value.strip() or column
        for column, value in headers.items()
    }
    duplicates = _duplicate_values(header_by_column.values())
    if duplicates:
        raise ValueError(
            f"Spreadsheet header row {header_row} has duplicate columns at "
            f"{path.resolve()}: {duplicates}."
        )
    rows = []
    for row_number in sorted(cells_by_row):
        if row_number <= header_row:
            continue
        cells = cells_by_row[row_number]
        row = {
            header: cells.get(column, "").strip()
            for column, header in header_by_column.items()
        }
        if any(row.values()):
            rows.append(row)
    if not rows:
        raise ValueError(
            f"Spreadsheet has no data rows after header row {header_row} at "
            f"{path.resolve()}."
        )
    return rows


def _read_shared_strings(archive: ZipFile) -> list[str]:
    """Return all shared XLSX string values in index order."""
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    return [
        "".join(item.itertext())
        for item in root.findall("main:si", NAMESPACES)
    ]


def _resolve_worksheet_path(archive: ZipFile, sheet: str | int) -> str:
    """Resolve one configured workbook sheet to its archive member path."""
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.findall("main:sheets/main:sheet", NAMESPACES)
    if isinstance(sheet, int):
        if sheet < 0 or sheet >= len(sheets):
            raise ValueError(
                f"Spreadsheet sheet index {sheet} is outside 0 to "
                f"{len(sheets) - 1}."
            )
        selected = sheets[sheet]
    else:
        matches = [item for item in sheets if item.get("name") == sheet]
        if not matches:
            available = [item.get("name") for item in sheets]
            raise ValueError(
                f"Spreadsheet sheet {sheet!r} does not exist; available "
                f"sheets are {available}."
            )
        selected = matches[0]
    relationship_id = selected.get(f"{{{RELATIONSHIP_NAMESPACE}}}id")
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels"),
    )
    target = next(
        (
            item.get("Target")
            for item in relationships.findall(
                "package:Relationship",
                NAMESPACES,
            )
            if item.get("Id") == relationship_id
        ),
        None,
    )
    if target is None:
        raise ValueError(
            "Spreadsheet workbook has no relationship for its sheet."
        )
    return f"xl/{target.lstrip('/')}"


def _read_worksheet_cells(
    archive: ZipFile,
    worksheet_path: str,
    shared_strings: list[str],
) -> dict[int, dict[str, str]]:
    """Read a worksheet's stringified cells by row and Excel column label."""
    root = ElementTree.fromstring(archive.read(worksheet_path))
    cells_by_row: dict[int, dict[str, str]] = {}
    for row in root.findall(".//main:sheetData/main:row", NAMESPACES):
        row_number = int(row.get("r", "0"))
        if row_number <= 0:
            raise ValueError("Spreadsheet row has no positive row index.")
        cells = {}
        for cell in row.findall("main:c", NAMESPACES):
            reference = cell.get("r", "")
            match = CELL_REFERENCE_PATTERN.match(reference)
            if match is None:
                raise ValueError(
                    f"Spreadsheet cell has invalid reference {reference!r}."
                )
            cells[match.group(1)] = _read_cell_value(cell, shared_strings)
        cells_by_row[row_number] = cells
    return cells_by_row


def _read_cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> str:
    """Return one XLSX cell's stored value as text."""
    cell_type = cell.get("t")
    value = cell.findtext("main:v", default="", namespaces=NAMESPACES)
    if cell_type == "s":
        if not value.isdigit() or int(value) >= len(shared_strings):
            raise ValueError(
                "Spreadsheet cell has an invalid shared-string index."
            )
        return shared_strings[int(value)]
    if cell_type == "inlineStr":
        inline_string = cell.find("main:is", NAMESPACES)
        if inline_string is None:
            return ""
        return "".join(inline_string.itertext())
    if cell.find("main:f", NAMESPACES) is not None and not value:
        raise ValueError("Spreadsheet formula cell has no cached value.")
    return value


def _duplicate_values(values: Iterable[str]) -> list[str]:
    """Return sorted duplicate non-empty strings from an iterable."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
