"""Crash-safe writes for files consumed across pipeline and dashboard processes."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def staged_path(path: str | os.PathLike[str], suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically promote it on success.

    Keeping the temporary file beside the target is important: ``os.replace`` is atomic
    only within one filesystem.  A failed writer leaves the previous target untouched and
    the temporary file is removed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        yield temporary_path
        # Flush writers that reopened the path before making the file visible to readers.
        # Windows requires a writable descriptor for ``fsync``; ``rb+`` works on every
        # supported platform without changing the bytes the writer produced.
        with temporary_path.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> Path:
    target = Path(path)
    with staged_path(target, suffix=".json.tmp") as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=indent, sort_keys=sort_keys)
            stream.flush()
            os.fsync(stream.fileno())
    return target


def atomic_write_pickle(path: str | os.PathLike[str], value: Any) -> Path:
    target = Path(path)
    with staged_path(target, suffix=".pickle.tmp") as temporary:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
    return target


def atomic_write_parquet(
    path: str | os.PathLike[str], frame: Any, *, index: bool = False
) -> Path:
    """Write a pandas-compatible frame without exposing a partial Parquet file."""
    target = Path(path)
    with staged_path(target, suffix=".parquet.tmp") as temporary:
        frame.to_parquet(temporary, index=index)
    return target


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()
