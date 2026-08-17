"""Temporary disk-backed arrays for classical feature extraction.

Inputs are array shapes and dtypes. Outputs are writable ``.npy`` memmaps in
one invocation-owned scratch directory, removed before completion is written.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import numpy as np

from baseline.feature_extractor.runtime import AddressSpaceGuard, GIBIBYTE


SCRATCH_FREE_RESERVE_BYTES = 16 * GIBIBYTE


class ScratchArray:
    """Own one temporary NumPy memmap and its allocated byte accounting."""

    def __init__(
        self,
        path: Path,
        array: np.memmap,
        owner: "ScratchSpace",
    ):
        self.path = path
        self._array: Optional[np.memmap] = array
        self.nbytes = int(array.nbytes)
        self._owner = owner

    @property
    def array(self) -> np.memmap:
        """Return the open memmap or fail after it has been released."""
        if self._array is None:
            raise RuntimeError(
                f"Scratch array at {self.path.resolve()} is already closed."
            )
        return self._array

    def close(self) -> None:
        """Flush, unmap, and remove this invocation-owned temporary file."""
        if self._array is None:
            return
        self._array.flush()
        memory_map = getattr(self._array, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
        self._array = None
        self.path.unlink(missing_ok=True)
        self._owner._release(self.nbytes)


class ScratchSpace:
    """Manage all temporary arrays for one dataset invocation."""

    def __init__(
        self,
        root: Path,
        prefix: str,
        memory_guard: AddressSpaceGuard,
    ):
        self.root = root.resolve()
        self.prefix = prefix
        self.memory_guard = memory_guard
        self.directory: Optional[Path] = None
        self._arrays: list[ScratchArray] = []
        self.current_bytes = 0
        self.peak_bytes = 0
        self.total_allocated_bytes = 0

    def __enter__(self) -> "ScratchSpace":
        """Create one invocation-local scratch directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(prefix=f"{self.prefix}-", dir=self.root)
        ).resolve()
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close arrays and remove only this invocation's directory."""
        del exception, traceback
        cleanup_error: Optional[BaseException] = None
        for scratch_array in reversed(self._arrays):
            try:
                scratch_array.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if self.directory is not None and self.directory.exists():
            try:
                shutil.rmtree(self.directory)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and exception_type is None:
            raise RuntimeError(
                "Failed to clean the invocation-local scratch directory."
            ) from cleanup_error

    def create_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype | type,
    ) -> ScratchArray:
        """Create a writable C-contiguous memmap after resource preflights."""
        if self.directory is None:
            raise RuntimeError("ScratchSpace must be entered before use.")
        normalized_dtype = np.dtype(dtype)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(
                f"Expected a non-empty positive array shape, but got {shape}."
            )
        requested_bytes = math.prod(shape) * normalized_dtype.itemsize
        disk_usage = shutil.disk_usage(self.root)
        required_disk = requested_bytes + SCRATCH_FREE_RESERVE_BYTES
        if disk_usage.free < required_disk:
            raise RuntimeError(
                f"Cannot allocate scratch array '{name}' with "
                f"{requested_bytes} bytes under {self.root}: "
                f"{disk_usage.free} bytes are free, but {required_disk} bytes "
                "are required including the scratch reserve."
            )
        self.memory_guard.require_additional(
            f"map scratch array {name}",
            requested_bytes,
        )
        path = self.directory / f"{name}.npy"
        try:
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=normalized_dtype,
                shape=shape,
                fortran_order=False,
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        scratch_array = ScratchArray(path, array, self)
        self._arrays.append(scratch_array)
        self.current_bytes += requested_bytes
        self.total_allocated_bytes += requested_bytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return scratch_array

    def _release(self, released_bytes: int) -> None:
        """Update live scratch accounting after one array is removed."""
        self.current_bytes -= released_bytes
        if self.current_bytes < 0:
            raise RuntimeError("Scratch byte accounting became negative.")
