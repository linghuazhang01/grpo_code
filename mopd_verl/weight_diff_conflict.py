"""Parameter-space teacher/student weight-diff conflict analysis."""

from __future__ import annotations

import argparse
import heapq
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from mopd_verl.weight_diff_io import WeightSource, require_torch
from mopd_verl.weight_diff_output import write_jsonl, write_markdown, write_sparse_diff_jsonl
from mopd_verl.weight_diff_save import DiffSaver
from mopd_verl.weight_diff_metrics import (
    Coordinate,
    PairDiffStats,
    TeacherDiffStats,
    pair_stats_from_sums,
    sparse_conflict_stats,
    teacher_stats_from_sums,
)

DEFAULT_CHUNK_SIZE = 1 << 20
DEFAULT_TOP_K = 100_000


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    path: Path


@dataclass
class TeacherAccumulator:
    path: str
    tensor_count: int = 0
    parameter_count: int = 0
    norm_sq: float = 0.0
    l1_norm: float = 0.0
    max_abs_diff: float = 0.0


@dataclass
class PairAccumulator:
    tensor_count: int = 0
    parameter_count: int = 0
    dot: float = 0.0


@dataclass(frozen=True)
class TensorPairRow:
    tensor: str
    teacher_a: str
    teacher_b: str
    parameter_count: int
    dot: float
    norm_a: float
    norm_b: float
    cosine: float | None
    conflict_strength: float | None


class TopKSupport:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._heap: list[tuple[float, str, int, float]] = []

    def offer(self, coord: Coordinate, value: float) -> None:
        if self.max_items <= 0 or value == 0.0:
            return
        item = (abs(value), coord[0], coord[1], value)
        if len(self._heap) < self.max_items:
            heapq.heappush(self._heap, item)
            return
        if item[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def offer_chunk(self, tensor_name: str, offset: int, diff: Any) -> None:
        if self.max_items <= 0 or diff.numel() == 0:
            return
        abs_diff = diff.abs()
        try:
            max_abs = float(abs_diff.max().item())
        except RuntimeError:
            return
        if max_abs == 0.0:
            return
        if len(self._heap) >= self.max_items:
            threshold = self._heap[0][0]
            if max_abs <= threshold:
                return
            indices = (abs_diff > threshold).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                return
            if indices.numel() > self.max_items:
                values, order = abs_diff[indices].topk(self.max_items)
                indices = indices[order]
            else:
                values = abs_diff[indices]
        else:
            keep = min(self.max_items, int(abs_diff.numel()))
            values, indices = abs_diff.topk(keep)
        for abs_value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            idx = int(index)
            signed = float(diff[idx].item())
            self.offer((tensor_name, offset + idx), signed if signed != 0.0 else float(abs_value))

    def as_mapping(self) -> dict[Coordinate, float]:
        return {(tensor, index): value for _abs_value, tensor, index, value in self._heap}


def parse_teacher_specs(raw_values: Sequence[str]) -> list[TeacherSpec]:
    specs: list[TeacherSpec] = []
    seen: set[str] = set()
    for raw in raw_values:
        if "=" in raw:
            name, raw_path = raw.split("=", 1)
        else:
            raw_path = raw
            name = Path(raw).name
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"Invalid or duplicate teacher name: {raw}")
        seen.add(name)
        specs.append(TeacherSpec(name=name, path=Path(raw_path)))
    if len(specs) < 2:
        raise ValueError("At least two --teacher entries are required")
    return specs


def resolve_tensor_names(
    student: WeightSource,
    teachers: Mapping[str, WeightSource],
    include_regex: str | None,
    exclude_regex: str | None,
    allow_missing: bool,
) -> list[str]:
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected = {
        name
        for name in student.tensor_names()
        if (include is None or include.search(name)) and (exclude is None or not exclude.search(name))
    }
    common = set(selected)
    missing: dict[str, list[str]] = {}
    for teacher_name, source in teachers.items():
        teacher_names = source.tensor_names()
        missing_names = sorted(selected - teacher_names)
        if missing_names:
            missing[teacher_name] = missing_names[:10]
        common &= teacher_names
    shape_mismatches = []
    for name in sorted(common):
        student_shape = student.shape(name)
        for teacher_name, source in teachers.items():
            if source.shape(name) != student_shape:
                shape_mismatches.append((name, teacher_name, student_shape, source.shape(name)))
    if (missing or shape_mismatches) and not allow_missing:
        details = {"missing_examples": missing, "shape_mismatch_examples": shape_mismatches[:10]}
        raise ValueError("Checkpoints are not structurally compatible. Use --allow-missing to skip. " + json.dumps(details))
    common -= {name for name, _teacher, _shape_a, _shape_b in shape_mismatches}
    if not common:
        raise ValueError("No common tensors remain after filtering")
    return sorted(common)


def analyze_weight_diffs(args: argparse.Namespace) -> tuple[list[TeacherDiffStats], list[PairDiffStats], list[TensorPairRow]]:
    torch = require_torch()
    teacher_specs = parse_teacher_specs(args.teacher)
    student = WeightSource(args.student, args.device)
    teachers = {spec.name: WeightSource(spec.path, args.device) for spec in teacher_specs}
    tensor_names = resolve_tensor_names(
        student,
        teachers,
        args.include_regex,
        args.exclude_regex,
        args.allow_missing,
    )
    if args.limit_tensors is not None:
        tensor_names = tensor_names[: args.limit_tensors]

    teacher_acc = {
        spec.name: TeacherAccumulator(path=str(spec.path.expanduser()))
        for spec in teacher_specs
    }
    pair_names = list(combinations([spec.name for spec in teacher_specs], 2))
    pair_acc = {pair: PairAccumulator() for pair in pair_names}
    topk = {spec.name: TopKSupport(args.top_k) for spec in teacher_specs}
    diff_saver = DiffSaver(args.diff_output_dir, args.diff_save_dtype, torch)
    tensor_rows: list[TensorPairRow] = []

    try:
        for ordinal, tensor_name in enumerate(tensor_names, start=1):
            student_tensor = student.load_tensor(tensor_name)
            teacher_tensors = {name: source.load_tensor(tensor_name) for name, source in teachers.items()}
            if args.skip_non_floating and (
                not torch.is_floating_point(student_tensor)
                or any(not torch.is_floating_point(tensor) for tensor in teacher_tensors.values())
            ):
                continue
            tensor_count = int(student_tensor.numel())
            tensor_norm_sq = {name: 0.0 for name in teachers}
            tensor_dot = {pair: 0.0 for pair in pair_names}
            flat_student = student_tensor.reshape(-1)
            flat_teachers = {name: tensor.reshape(-1) for name, tensor in teacher_tensors.items()}
            for name, tensor in teacher_tensors.items():
                diff_saver.save_tensor(name, tensor_name, tensor, student_tensor)
            for start in range(0, tensor_count, args.chunk_size_elements):
                end = min(start + args.chunk_size_elements, tensor_count)
                student_chunk = flat_student[start:end].to(dtype=torch.float64)
                diffs: dict[str, Any] = {}
                for name, flat_teacher in flat_teachers.items():
                    diff = flat_teacher[start:end].to(dtype=torch.float64) - student_chunk
                    diffs[name] = diff
                    norm_sq = float((diff * diff).sum().item())
                    abs_sum = float(diff.abs().sum().item())
                    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
                    tensor_norm_sq[name] += norm_sq
                    acc = teacher_acc[name]
                    acc.norm_sq += norm_sq
                    acc.l1_norm += abs_sum
                    acc.max_abs_diff = max(acc.max_abs_diff, max_abs)
                    topk[name].offer_chunk(tensor_name, start, diff)
                for left, right in pair_names:
                    value = float((diffs[left] * diffs[right]).sum().item())
                    tensor_dot[(left, right)] += value
                    pair_acc[(left, right)].dot += value
            for name in teachers:
                teacher_acc[name].tensor_count += 1
                teacher_acc[name].parameter_count += tensor_count
            for pair in pair_names:
                pair_acc[pair].tensor_count += 1
                pair_acc[pair].parameter_count += tensor_count
                if args.layer_jsonl is not None:
                    stats = pair_stats_from_sums(
                        teacher_a=pair[0],
                        teacher_b=pair[1],
                        tensor_count=1,
                        parameter_count=tensor_count,
                        dot=tensor_dot[pair],
                        norm_sq_a=tensor_norm_sq[pair[0]],
                        norm_sq_b=tensor_norm_sq[pair[1]],
                    )
                    tensor_rows.append(
                        TensorPairRow(
                            tensor=tensor_name,
                            teacher_a=pair[0],
                            teacher_b=pair[1],
                            parameter_count=tensor_count,
                            dot=stats.dot,
                            norm_a=stats.norm_a,
                            norm_b=stats.norm_b,
                            cosine=stats.cosine,
                            conflict_strength=stats.conflict_strength,
                        )
                    )
            if args.progress_every > 0 and ordinal % args.progress_every == 0:
                print(json.dumps({"processed_tensors": ordinal, "total_tensors": len(tensor_names)}), flush=True)
    finally:
        diff_saver.close()

    supports = {name: topk[name].as_mapping() for name in teachers}
    if args.sparse_diff_jsonl is not None:
        write_sparse_diff_jsonl(args.sparse_diff_jsonl, supports)
    teacher_rows = [
        teacher_stats_from_sums(
            teacher=name,
            path=teacher_acc[name].path,
            tensor_count=teacher_acc[name].tensor_count,
            parameter_count=teacher_acc[name].parameter_count,
            norm_sq=teacher_acc[name].norm_sq,
            l1_norm=teacher_acc[name].l1_norm,
            max_abs_diff=teacher_acc[name].max_abs_diff,
        )
        for name in teachers
    ]
    pair_rows = [
        pair_stats_from_sums(
            teacher_a=left,
            teacher_b=right,
            tensor_count=pair_acc[(left, right)].tensor_count,
            parameter_count=pair_acc[(left, right)].parameter_count,
            dot=pair_acc[(left, right)].dot,
            norm_sq_a=teacher_acc[left].norm_sq,
            norm_sq_b=teacher_acc[right].norm_sq,
            sparse=sparse_conflict_stats(supports[left], supports[right]) if args.top_k > 0 else None,
        )
        for left, right in pair_names
    ]
    return teacher_rows, pair_rows, tensor_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True, help="Student/base checkpoint path")
    parser.add_argument("--teacher", action="append", required=True, help="Teacher as name=checkpoint_path")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Pair-level JSONL output")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown summary output")
    parser.add_argument("--layer-jsonl", type=Path, default=None, help="Optional per-tensor pair JSONL output")
    parser.add_argument("--teacher-jsonl", type=Path, default=None, help="Optional teacher diff norm JSONL output")
    parser.add_argument("--diff-output-dir", type=Path, default=None, help="Optional directory for full per-tensor diff .pt files")
    parser.add_argument("--diff-save-dtype", default="float32", choices=("float32", "float16", "bfloat16", "source"))
    parser.add_argument("--sparse-diff-jsonl", type=Path, default=None, help="Optional JSONL for top-|delta| diff coordinates")
    parser.add_argument("--device", default="cpu", help="torch device for tensor loading/reduction")
    parser.add_argument("--chunk-size-elements", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-|delta| coordinates per teacher; 0 disables sparse metrics")
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing or shape-mismatched tensors")
    parser.add_argument("--skip-non-floating", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-tensors", type=int, default=None, help="Debug only: process first N matched tensors")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args(argv)
    if args.chunk_size_elements <= 0:
        parser.error("--chunk-size-elements must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.limit_tensors is not None and args.limit_tensors <= 0:
        parser.error("--limit-tensors must be positive when set")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    teacher_rows, pair_rows, tensor_rows = analyze_weight_diffs(args)
    write_jsonl(args.output_jsonl, pair_rows)
    if args.teacher_jsonl is not None:
        write_jsonl(args.teacher_jsonl, teacher_rows)
    if args.layer_jsonl is not None:
        write_jsonl(args.layer_jsonl, tensor_rows)
    write_markdown(args.output_md, teacher_rows, pair_rows)
    summary = {
        "teachers": len(teacher_rows),
        "pairs": len(pair_rows),
        "output_jsonl": str(args.output_jsonl),
        "output_md": str(args.output_md),
        "layer_jsonl": None if args.layer_jsonl is None else str(args.layer_jsonl),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
