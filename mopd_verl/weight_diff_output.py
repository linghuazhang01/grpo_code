"""Output writers for weight-diff conflict analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from mopd_verl.weight_diff_metrics import PairDiffStats, TeacherDiffStats


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_markdown(path: Path, teacher_rows: Sequence[TeacherDiffStats], pair_rows: Sequence[PairDiffStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Weight Diff Conflict Summary",
        "",
        "Teacher/student diff is defined as `teacher_weight - student_weight` over matched floating tensors.",
        "",
        "## Diff Norms",
        "",
        "| Teacher | Tensors | Parameters | L2 Norm | L1 Norm | Max Abs Diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in teacher_rows:
        lines.append(
            f"| {row.teacher} | {row.tensor_count} | {row.parameter_count} | "
            f"{row.norm:.6g} | {row.l1_norm:.6g} | {row.max_abs_diff:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Pair Conflicts",
            "",
            "| Pair | Cosine | Conflict Strength | Negative Dot | Sparse Cosine | Sparse Conflict | Top-K Overlap | Jaccard |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        sparse = row.sparse
        lines.append(
            "| {pair} | {cos} | {conflict} | {neg:.6g} | {sparse_cos} | {sparse_conflict} | {overlap} | {jaccard} |".format(
                pair=f"{row.teacher_a}/{row.teacher_b}",
                cos=fmt(row.cosine),
                conflict=fmt(row.conflict_strength),
                neg=row.negative_dot,
                sparse_cos=fmt(None if sparse is None else sparse.sparse_cosine),
                sparse_conflict=fmt(None if sparse is None else sparse.sparse_conflict_strength),
                overlap="NA" if sparse is None else sparse.overlap_size,
                jaccard="NA" if sparse is None else f"{sparse.support_jaccard:.6g}",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `cosine < 0` means the two teacher/student update directions oppose each other globally.",
            "- `conflict_strength = max(0, -cosine)` is the normalized global conflict score.",
            "- Sparse metrics are computed on the overlap of each diff's top-|delta| coordinates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6g}"


def write_sparse_diff_jsonl(path: Path, supports: dict[str, dict[tuple[str, int], float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for teacher, values in sorted(supports.items()):
            rows = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
            for (tensor, flat_index), diff in rows:
                handle.write(
                    json.dumps(
                        {
                            "teacher": teacher,
                            "tensor": tensor,
                            "flat_index": flat_index,
                            "diff": diff,
                            "abs_diff": abs(diff),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
