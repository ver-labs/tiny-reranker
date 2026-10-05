"""Write checkpoints and their metadata through same-directory temporary files."""

import os
import tempfile
from pathlib import Path

import torch


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_torch_save(payload, destination: Path, *, replace_existing: bool = False) -> None:
    """Verify a complete temporary checkpoint before atomically renaming it."""
    destination = Path(destination)
    if destination.exists() and not replace_existing:
        raise FileExistsError(destination)
    temporary = _temporary_path(destination)
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=True)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if destination.exists() and not replace_existing:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(content: str, destination: Path,
                      *, replace_existing: bool = False) -> None:
    """Write a report or manifest without exposing a partial final file."""
    destination = Path(destination)
    if destination.exists() and not replace_existing:
        raise FileExistsError(destination)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not replace_existing:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
