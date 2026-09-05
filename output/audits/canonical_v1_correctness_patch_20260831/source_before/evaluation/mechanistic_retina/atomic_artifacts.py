from __future__ import annotations

from pathlib import Path

import torch


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


__all__ = ["atomic_torch_save", "atomic_write_text"]
