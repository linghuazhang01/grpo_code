"""Metrics for parameter-space teacher/student weight-diff conflicts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

Coordinate = tuple[str, int]


@dataclass(frozen=True)
class SparseConflictStats:
    top_k_a: int
    top_k_b: int
    overlap_size: int
    support_jaccard: float
    sparse_dot: float
    sparse_cosine: float | None
    sparse_conflict_strength: float | None
    sparse_negative_dot: float
    sparse_abs_overlap_mass: float


@dataclass(frozen=True)
class PairDiffStats:
    teacher_a: str
    teacher_b: str
    tensor_count: int
    parameter_count: int
    dot: float
    norm_a: float
    norm_b: float
    cosine: float | None
    alignment_strength: float | None
    conflict_strength: float | None
    negative_dot: float
    negative_dot_per_param: float | None
    projection_a_on_b: float | None
    projection_b_on_a: float | None
    sparse: SparseConflictStats | None = None


@dataclass(frozen=True)
class TeacherDiffStats:
    teacher: str
    path: str
    tensor_count: int
    parameter_count: int
    norm: float
    l1_norm: float
    max_abs_diff: float


def cosine_from_sums(dot: float, norm_sq_a: float, norm_sq_b: float) -> float | None:
    if norm_sq_a <= 0.0 or norm_sq_b <= 0.0:
        return None
    return dot / math.sqrt(norm_sq_a * norm_sq_b)


def pair_stats_from_sums(
    teacher_a: str,
    teacher_b: str,
    tensor_count: int,
    parameter_count: int,
    dot: float,
    norm_sq_a: float,
    norm_sq_b: float,
    sparse: SparseConflictStats | None = None,
) -> PairDiffStats:
    cosine = cosine_from_sums(dot, norm_sq_a, norm_sq_b)
    norm_a = math.sqrt(max(norm_sq_a, 0.0))
    norm_b = math.sqrt(max(norm_sq_b, 0.0))
    negative_dot = max(0.0, -dot)
    return PairDiffStats(
        teacher_a=teacher_a,
        teacher_b=teacher_b,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        dot=dot,
        norm_a=norm_a,
        norm_b=norm_b,
        cosine=cosine,
        alignment_strength=None if cosine is None else max(0.0, cosine),
        conflict_strength=None if cosine is None else max(0.0, -cosine),
        negative_dot=negative_dot,
        negative_dot_per_param=None if parameter_count <= 0 else negative_dot / parameter_count,
        projection_a_on_b=None if norm_b == 0.0 else dot / norm_b,
        projection_b_on_a=None if norm_a == 0.0 else dot / norm_a,
        sparse=sparse,
    )


def sparse_conflict_stats(
    left: Mapping[Coordinate, float],
    right: Mapping[Coordinate, float],
) -> SparseConflictStats:
    support_left = set(left)
    support_right = set(right)
    overlap = support_left & support_right
    union = support_left | support_right
    sparse_dot = sum(left[coord] * right[coord] for coord in overlap)
    norm_sq_left = sum(left[coord] * left[coord] for coord in overlap)
    norm_sq_right = sum(right[coord] * right[coord] for coord in overlap)
    sparse_cosine = cosine_from_sums(sparse_dot, norm_sq_left, norm_sq_right)
    return SparseConflictStats(
        top_k_a=len(left),
        top_k_b=len(right),
        overlap_size=len(overlap),
        support_jaccard=0.0 if not union else len(overlap) / len(union),
        sparse_dot=sparse_dot,
        sparse_cosine=sparse_cosine,
        sparse_conflict_strength=None if sparse_cosine is None else max(0.0, -sparse_cosine),
        sparse_negative_dot=max(0.0, -sparse_dot),
        sparse_abs_overlap_mass=sum(min(abs(left[coord]), abs(right[coord])) for coord in overlap),
    )


def teacher_stats_from_sums(
    teacher: str,
    path: str,
    tensor_count: int,
    parameter_count: int,
    norm_sq: float,
    l1_norm: float,
    max_abs_diff: float,
) -> TeacherDiffStats:
    return TeacherDiffStats(
        teacher=teacher,
        path=path,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        norm=math.sqrt(max(norm_sq, 0.0)),
        l1_norm=l1_norm,
        max_abs_diff=max_abs_diff,
    )
