"""File writing and digesting shared by the build stages."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

DIGEST_CHUNK_BYTES = 1 << 20


def digest_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in chunks.

    Args:
        path: The file to digest.

    Returns:
        The digest as lowercase hexadecimal.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(DIGEST_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def atomic_path(path: Path) -> Generator[Path]:
    """Yield a temporary path, then move it onto the target path.

    The rename at the end is atomic, so the target is either absent or
    complete. A stage that stops part way leaves no partial output for a later
    run to read as finished.

    Args:
        path: The file to write.

    Yields:
        The temporary path to write to, same directory as the target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_atomic(path: Path, text: str) -> None:
    """Write text to a file through a temporary file in the same directory.

    Args:
        path: The file to write.
        text: The contents to write.
    """
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
