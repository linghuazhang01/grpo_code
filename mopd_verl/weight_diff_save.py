"""Optional tensor diff persistence for weight-diff analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SavedDiffRow:
    teacher: str
    tensor: str
    path: str
    shape: tuple[int, ...]
    dtype: str


class DiffSaver:
    def __init__(self, output_dir: Path | None, dtype_name: str, torch_module: Any) -> None:
        self.output_dir = output_dir
        self.dtype_name = dtype_name
        self.torch = torch_module
        self._seen: dict[tuple[str, str], int] = {}
        self._manifest = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = (self.output_dir / "manifest.jsonl").open("w", encoding="utf-8")

    def close(self) -> None:
        if self._manifest is not None:
            self._manifest.close()
            self._manifest = None

    def save_tensor(self, teacher: str, tensor_name: str, teacher_tensor: Any, student_tensor: Any) -> None:
        if self.output_dir is None:
            return
        dtype = self._resolve_dtype(teacher_tensor)
        diff = teacher_tensor.to(dtype=dtype) - student_tensor.to(dtype=dtype)
        teacher_dir = self.output_dir / safe_path_part(teacher)
        teacher_dir.mkdir(parents=True, exist_ok=True)
        path = teacher_dir / f"{self._unique_name(teacher, tensor_name)}.pt"
        self.torch.save({"tensor": tensor_name, "diff": diff.detach().cpu()}, path)
        row = SavedDiffRow(
            teacher=teacher,
            tensor=tensor_name,
            path=str(path),
            shape=tuple(int(dim) for dim in diff.shape),
            dtype=str(diff.dtype).replace("torch.", ""),
        )
        assert self._manifest is not None
        self._manifest.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
        self._manifest.flush()

    def _resolve_dtype(self, tensor: Any) -> Any:
        if self.dtype_name == "source":
            return tensor.dtype
        dtype = getattr(self.torch, self.dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported diff save dtype: {self.dtype_name}")
        return dtype

    def _unique_name(self, teacher: str, tensor_name: str) -> str:
        safe_name = safe_path_part(tensor_name)
        key = (teacher, safe_name)
        count = self._seen.get(key, 0)
        self._seen[key] = count + 1
        return safe_name if count == 0 else f"{safe_name}.{count}"


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "tensor"
