"""Shared EEG electrode-position resolution utilities.

The preprocessing layer persists Cartesian positions in the same channel order as
the selected raw signal.  Source-provided MNE locations are always preferred;
configured standard montages are only a deterministic fallback.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import mne
import numpy as np
from mne.io import BaseRaw


_CHANNEL_ALIASES = {
    "A1": "M1",
    "A2": "M2",
    "T1": "T9",
    "T2": "T10",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def extract_complete_xyz(raw: BaseRaw) -> Optional[np.ndarray]:
    """Return complete source XYZ coordinates from ``raw.info``, if present."""
    positions = np.asarray(
        [channel["loc"][:3] for channel in raw.info["chs"]], dtype=np.float32
    )
    expected_shape = (len(raw.ch_names), 3)
    if positions.shape != expected_shape:
        raise ValueError(
            "Could not extract electrode positions with shape "
            f"{expected_shape}; got {positions.shape}."
        )
    if not np.isfinite(positions).all() or np.any(np.all(positions == 0.0, axis=1)):
        return None
    return np.ascontiguousarray(positions)


def resolve_electrode_positions(
    raw: BaseRaw,
    position_montage: Optional[str],
    standardized_channel_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Resolve complete selected-channel XYZ positions.

    Native ``raw.info`` coordinates take precedence.  When they are incomplete,
    a configured MNE template is mapped to the selected raw channel names (using
    standardized labels as aliases), attached to the raw object, and read back
    from ``raw.info``.  A configured montage must resolve every selected channel.
    """
    native_positions = extract_complete_xyz(raw)
    if native_positions is not None:
        return native_positions
    if not position_montage:
        return None
    if len(standardized_channel_names) != len(raw.ch_names):
        raise ValueError(
            "Configured position montage requires one standardized channel name per "
            f"selected raw channel; got {len(standardized_channel_names)} names for "
            f"{len(raw.ch_names)} channels."
        )

    try:
        template = mne.channels.make_standard_montage(position_montage)
    except Exception as exc:
        raise ValueError(
            f"Could not create configured position montage {position_montage!r}."
        ) from exc

    template_positions = template.get_positions()["ch_pos"]
    template_lookup = {
        _canonicalize_channel_name(name): np.asarray(position, dtype=np.float64)
        for name, position in template_positions.items()
    }
    channel_positions: dict[str, np.ndarray] = {}
    unresolved: list[str] = []
    used_template_names: set[str] = set()
    for raw_name, standardized_name in zip(raw.ch_names, standardized_channel_names):
        candidates = (_canonicalize_channel_name(raw_name), _canonicalize_channel_name(standardized_name))
        template_name = next((candidate for candidate in candidates if candidate in template_lookup), None)
        if template_name is None:
            unresolved.append(raw_name)
            continue
        if template_name in used_template_names:
            raise ValueError(
                "Configured position montage maps multiple selected channels to "
                f"{template_name!r}: {raw_name!r}."
            )
        used_template_names.add(template_name)
        channel_positions[raw_name] = template_lookup[template_name]

    if unresolved:
        raise ValueError(
            "Configured position montage "
            f"{position_montage!r} could not resolve selected channel(s): {unresolved}."
        )

    raw.set_montage(
        mne.channels.make_dig_montage(ch_pos=channel_positions, coord_frame="head"),
        on_missing="raise",
        verbose=False,
    )
    positions = extract_complete_xyz(raw)
    if positions is None:
        raise ValueError(
            f"Configured position montage {position_montage!r} did not produce "
            "complete finite XYZ positions."
        )
    return positions


def _canonicalize_channel_name(channel_name: str) -> str:
    """Normalize common EEG labels while retaining standard montage identifiers."""
    name = str(channel_name).strip().upper()
    name = re.sub(r"^EEG[ _-]*", "", name)
    name = re.sub(r"[-_ ]*(REF|LE|RE)$", "", name)
    if "-" in name:
        name = name.split("-", maxsplit=1)[0]
    name = re.sub(r"[^A-Z0-9]", "", name)
    return _CHANNEL_ALIASES.get(name, name)
