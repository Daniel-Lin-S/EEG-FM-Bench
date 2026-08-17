"""Runtime memory and concurrency guards for classical EEG baselines.

The guards apply only to invocation-local process state. They neither modify
dataset sources nor persist machine-specific paths in result metadata.
"""

from __future__ import annotations

import fcntl
import json
import os
import resource
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Optional, Type


GIBIBYTE = 1 << 30


def _virtual_memory_bytes() -> int:
    """Return the current process virtual-memory size in bytes."""
    statm_path = Path("/proc/self/statm")
    try:
        pages = int(statm_path.read_text(encoding="utf-8").split()[0])
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot read process memory usage from {statm_path.resolve()}."
        ) from exc
    return pages * os.sysconf("SC_PAGE_SIZE")


def peak_resident_memory_bytes() -> int:
    """Return peak resident memory for the current Linux process."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class AddressSpaceGuard:
    """Apply and validate a per-process Linux address-space ceiling."""

    def __init__(self, limit_gib: float):
        self.limit_bytes = int(limit_gib * GIBIBYTE)
        self._original_limit: Optional[tuple[int, int]] = None
        self.effective_limit_bytes = self.limit_bytes

    def __enter__(self) -> "AddressSpaceGuard":
        """Lower the soft address-space limit for this invocation."""
        current = resource.getrlimit(resource.RLIMIT_AS)
        self._original_limit = current
        _, hard_limit = current
        if hard_limit != resource.RLIM_INFINITY:
            self.effective_limit_bytes = min(
                self.limit_bytes,
                int(hard_limit),
            )
        current_vms = _virtual_memory_bytes()
        if current_vms >= self.effective_limit_bytes:
            raise RuntimeError(
                "Cannot apply the feature-extractor memory limit: current "
                f"virtual memory is {current_vms} bytes, but the effective "
                f"limit is {self.effective_limit_bytes} bytes."
            )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.effective_limit_bytes, hard_limit),
        )
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Restore the pre-existing soft and hard resource limits."""
        del exception_type, exception, traceback
        if self._original_limit is not None:
            resource.setrlimit(resource.RLIMIT_AS, self._original_limit)

    def require_additional(
        self,
        phase: str,
        requested_bytes: int,
    ) -> None:
        """Require one planned allocation to fit below the active ceiling."""
        if requested_bytes < 0:
            raise ValueError(
                f"Expected non-negative requested bytes, but got "
                f"{requested_bytes}."
            )
        current_vms = _virtual_memory_bytes()
        projected = current_vms + requested_bytes
        if projected > self.effective_limit_bytes:
            raise MemoryError(
                f"Feature-extractor phase '{phase}' would require "
                f"approximately {requested_bytes} additional bytes; current "
                f"virtual memory is {current_vms} bytes and the configured "
                f"limit is {self.effective_limit_bytes} bytes."
            )


class ModelRunLock:
    """Reject a concurrent invocation of the same feature-extractor model."""

    def __init__(self, lock_root: str | Path, model_type: str):
        self.path = (
            Path(lock_root).resolve()
            / ".locks"
            / f"{model_type}.runtime.lock"
        )
        self.model_type = model_type
        self._file: Optional[BinaryIO] = None

    def __enter__(self) -> "ModelRunLock":
        """Acquire the model-wide advisory lock without waiting."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+b")
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            lock_file.seek(0)
            holder = lock_file.read().decode("utf-8", errors="replace").strip()
            lock_file.close()
            detail = f" Active holder: {holder}." if holder else ""
            raise RuntimeError(
                f"Another {self.model_type} invocation owns the runtime "
                f"lock at {self.path.resolve()}.{detail}"
            ) from exc
        payload = {
            "model_type": self.model_type,
            "pid": os.getpid(),
        }
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
        lock_file.flush()
        os.fsync(lock_file.fileno())
        self._file = lock_file
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Release the advisory lock."""
        del exception_type, exception, traceback
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None
