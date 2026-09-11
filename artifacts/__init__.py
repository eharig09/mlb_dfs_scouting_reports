"""Versioned artifact and cache I/O primitives.

Writers in this project are often read by another process (the dashboard) while the
pipeline is refreshing them.  Export the small, shared surface here so callers do not
grow their own subtly different temporary-file implementations.
"""

from .io import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_pickle,
    file_sha256,
    staged_path,
)

__all__ = [
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_pickle",
    "file_sha256",
    "staged_path",
]
