"""SOPC-lite postprocessing for existing MOPD audit token-gap vectors."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_TOP_MASS_FRACTION = 0.10
DEFAULT_RANDOM_TRIALS = 32


@dataclass(frozen=True)
class GapVector:
    step: int
    domain: str
    values: tuple[float, ...]
    token_count: float | None
    coordinate_space: str = "occurrence"
    source_field: str = "gap_signed_vector_domain"
    row_count: int = 1


@dataclass(frozen=True)
class CollisionRow:
    step: int
    domain_a: str
    domain_b: str
    vector_size: int
    top_mass_fraction: float
    support_a_size: int
    support_b_size: int
    overlap_size: int
    support_jaccard: float
    full_cosine: float | None
    sparse_cosine: float | None
    sparse_signed_dot: float
    sparse_negative_collision: float
    sparse_abs_overlap_mass: float
    random_negative_collision_mean: float | None
    random_negative_collision_std: float | None


@dataclass(frozen=True)
class SkippedPair:
    step: int
    domain_a: str
    domain_b: str
    reason: str
    vector_size_a: int
    vector_size_b: int
    coordinate_space_a: str
    coordinate_space_b: str


@dataclass(frozen=True)
class CollisionResult:
    rows: tuple[CollisionRow, ...]
    skipped_pairs: tuple[SkippedPair, ...]


def read_gap_vectors(path: Path) -> list[GapVector]:
    rows: list[GapVector] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            raw_values, source_field, coordinate_space = _select_gap_values(payload)
            if not isinstance(raw_values, list):
                raise ValueError(f"{path}:{line_number} has no gap vector")
            rows.append(
                GapVector(
                    step=int(payload.get("step", 0)),
                    domain=str(payload.get("domain", "unknown")),
                    values=tuple(float(value) for value in raw_values),
                    token_count=_optional_float(payload.get("observed_token_count", payload.get("token_count"))),
                    coordinate_space=coordinate_space,
                    source_field=source_field,
                )
            )
    return rows


def compute_collision_rows(
    vectors: Sequence[GapVector],
    top_mass_fraction: float = DEFAULT_TOP_MASS_FRACTION,
    random_trials: int = DEFAULT_RANDOM_TRIALS,
    seed: int = 0,
) -> list[CollisionRow]:
    return list(
        compute_collision_result(
            vectors,
            top_mass_fraction=top_mass_fraction,
            random_trials=random_trials,
            seed=seed,
        ).rows
    )


def compute_collision_result(
    vectors: Sequence[GapVector],
    top_mass_fraction: float = DEFAULT_TOP_MASS_FRACTION,
    random_trials: int = DEFAULT_RANDOM_TRIALS,
    seed: int = 0,
) -> CollisionResult:
    grouped: dict[int, list[GapVector]] = {}
    for vector in aggregate_gap_vectors(vectors):
        grouped.setdefault(vector.step, []).append(vector)

    rows: list[CollisionRow] = []
    skipped_pairs: list[SkippedPair] = []
    rng = random.Random(seed)
    for step, step_vectors in sorted(grouped.items()):
        for left, right in combinations(sorted(step_vectors, key=lambda item: item.domain), 2):
            reason = incompatible_reason(left, right)
            if reason is not None:
                skipped_pairs.append(
                    SkippedPair(
                        step=step,
                        domain_a=left.domain,
                        domain_b=right.domain,
                        reason=reason,
                        vector_size_a=len(left.values),
                        vector_size_b=len(right.values),
                        coordinate_space_a=left.coordinate_space,
                        coordinate_space_b=right.coordinate_space,
                    )
                )
                continue
            rows.append(
                compute_pair_collision(
                    left,
                    right,
                    top_mass_fraction=top_mass_fraction,
                    random_trials=random_trials,
                    rng=rng,
                )
            )
    return CollisionResult(rows=tuple(rows), skipped_pairs=tuple(skipped_pairs))


def aggregate_gap_vectors(vectors: Sequence[GapVector]) -> list[GapVector]:
    groups: dict[tuple[int, str, str, str], list[GapVector]] = {}
    for vector in vectors:
        key = (vector.step, vector.domain, vector.coordinate_space, vector.source_field)
        groups.setdefault(key, []).append(vector)

    aggregated: list[GapVector] = []
    for group_vectors in groups.values():
        first = group_vectors[0]
        if len(group_vectors) == 1:
            aggregated.append(first)
            continue
        lengths = {len(vector.values) for vector in group_vectors}
        if len(lengths) != 1:
            raise ValueError(f"Cannot aggregate {first.step}/{first.domain}: mixed vector lengths {sorted(lengths)}")
        if all(vector.values == first.values for vector in group_vectors):
            values = first.values
        else:
            values = tuple(
                sum(vector.values[idx] for vector in group_vectors) / len(group_vectors)
                for idx in range(len(first.values))
            )
        token_counts = [vector.token_count for vector in group_vectors if vector.token_count is not None]
        aggregated.append(
            GapVector(
                step=first.step,
                domain=first.domain,
                values=values,
                token_count=mean(token_counts),
                coordinate_space=first.coordinate_space,
                source_field=first.source_field,
                row_count=len(group_vectors),
            )
        )
    return aggregated


def incompatible_reason(left: GapVector, right: GapVector) -> str | None:
    if left.step != right.step:
        return "step_mismatch"
    if left.coordinate_space != right.coordinate_space:
        return "coordinate_space_mismatch"
    if len(left.values) != len(right.values):
        return "vector_size_mismatch"
    return None


def compute_pair_collision(
    left: GapVector,
    right: GapVector,
    top_mass_fraction: float,
    random_trials: int,
    rng: random.Random,
) -> CollisionRow:
    if len(left.values) != len(right.values):
        raise ValueError(
            f"Vector size mismatch for step {left.step}: "
            f"{left.domain}={len(left.values)} vs {right.domain}={len(right.values)}"
        )
    if left.step != right.step:
        raise ValueError(f"Step mismatch: {left.step} vs {right.step}")

    support_left = top_mass_support(left.values, top_mass_fraction)
    support_right = top_mass_support(right.values, top_mass_fraction)
    overlap = support_left & support_right
    union = support_left | support_right
    sparse_dot = dot_on_indices(left.values, right.values, overlap)
    random_values = random_negative_collisions(
        left.values,
        right.values,
        sample_size=len(overlap),
        trials=random_trials,
        rng=rng,
    )
    return CollisionRow(
        step=left.step,
        domain_a=left.domain,
        domain_b=right.domain,
        vector_size=len(left.values),
        top_mass_fraction=top_mass_fraction,
        support_a_size=len(support_left),
        support_b_size=len(support_right),
        overlap_size=len(overlap),
        support_jaccard=0.0 if not union else len(overlap) / len(union),
        full_cosine=cosine(left.values, right.values),
        sparse_cosine=cosine_on_indices(left.values, right.values, overlap),
        sparse_signed_dot=sparse_dot,
        sparse_negative_collision=max(0.0, -sparse_dot),
        sparse_abs_overlap_mass=sum(min(abs(left.values[idx]), abs(right.values[idx])) for idx in overlap),
        random_negative_collision_mean=mean(random_values),
        random_negative_collision_std=std(random_values),
    )


def top_mass_support(values: Sequence[float], fraction: float) -> set[int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top_mass_fraction must be in (0, 1]")
    abs_values = [(idx, abs(value)) for idx, value in enumerate(values) if value != 0.0]
    total = sum(value for _, value in abs_values)
    if total <= 0.0:
        return set()
    threshold = total * fraction
    support: set[int] = set()
    running = 0.0
    for idx, value in sorted(abs_values, key=lambda item: item[1], reverse=True):
        support.add(idx)
        running += value
        if running >= threshold:
            break
    return support


def dot_on_indices(left: Sequence[float], right: Sequence[float], indices: Iterable[int]) -> float:
    return sum(left[idx] * right[idx] for idx in indices)


def cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    indices = range(len(left))
    return cosine_on_indices(left, right, indices)


def cosine_on_indices(left: Sequence[float], right: Sequence[float], indices: Iterable[int]) -> float | None:
    index_tuple = tuple(indices)
    if not index_tuple:
        return None
    dot = dot_on_indices(left, right, index_tuple)
    left_norm = math.sqrt(sum(left[idx] * left[idx] for idx in index_tuple))
    right_norm = math.sqrt(sum(right[idx] * right[idx] for idx in index_tuple))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (left_norm * right_norm)


def random_negative_collisions(
    left: Sequence[float],
    right: Sequence[float],
    sample_size: int,
    trials: int,
    rng: random.Random,
) -> list[float]:
    if sample_size <= 0 or trials <= 0:
        return []
    sample_size = min(sample_size, len(left))
    indices = list(range(len(left)))
    values = []
    for _ in range(trials):
        sample = rng.sample(indices, sample_size)
        values.append(max(0.0, -dot_on_indices(left, right, sample)))
    return values


def write_jsonl(path: Path, rows: Sequence[CollisionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_markdown(
    path: Path,
    rows: Sequence[CollisionRow],
    source: Path,
    skipped_pairs: Sequence[SkippedPair] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SOPC-Lite Audit Summary",
        "",
        f"Source: `{source}`",
        "",
        "| Step | Pair | Sparse Neg. Collision | Sparse Cosine | Full Cosine | Overlap | Jaccard | Random Neg. Mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = f"{row.domain_a}/{row.domain_b}"
        lines.append(
            "| {step} | {pair} | {neg:.6g} | {sparse_cos} | {full_cos} | {overlap} | {jaccard:.4f} | {random_mean} |".format(
                step=row.step,
                pair=pair,
                neg=row.sparse_negative_collision,
                sparse_cos=_fmt_optional(row.sparse_cosine),
                full_cos=_fmt_optional(row.full_cosine),
                overlap=row.overlap_size,
                jaccard=row.support_jaccard,
                random_mean=_fmt_optional(row.random_negative_collision_mean),
            )
        )
    if skipped_pairs:
        lines.extend(
            [
                "",
                "Skipped pairs:",
                "",
                "| Step | Pair | Reason | Shape A | Shape B | Space A | Space B |",
                "|---:|---|---|---:|---:|---|---|",
            ]
        )
        for skipped in skipped_pairs:
            pair = f"{skipped.domain_a}/{skipped.domain_b}"
            lines.append(
                "| {step} | {pair} | {reason} | {shape_a} | {shape_b} | {space_a} | {space_b} |".format(
                    step=skipped.step,
                    pair=pair,
                    reason=skipped.reason,
                    shape_a=skipped.vector_size_a,
                    shape_b=skipped.vector_size_b,
                    space_a=skipped.coordinate_space_a,
                    space_b=skipped.coordinate_space_b,
                )
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This is **SOPC-lite**, based on signed token-gap vectors rather than parameter-space OPD residual gradients.",
            "- Cross-domain dot products require shared coordinates; vocab vectors are comparable, occurrence vectors usually are not.",
            "- A high sparse negative collision with low random negative collision is a candidate failure case to inspect.",
            "- This output is suitable for R003 pre-screening, not for a final 5.0 claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6g}"


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    avg = mean(values)
    assert avg is not None
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _select_gap_values(payload: dict[str, Any]) -> tuple[Any, str, str]:
    candidates = [
        ("gap_signed_sum_vector_vocab", "vocab"),
        ("gap_signed_mean_vector_vocab", "vocab"),
        ("gap_signed_vector_domain", "occurrence"),
        ("gap_vector_domain", "occurrence"),
        ("gap_abs_sum_vector_vocab", "vocab_abs"),
        ("gap_abs_mean_vector_vocab", "vocab_abs"),
        ("gap_abs_vector_domain", "occurrence_abs"),
    ]
    for field, coordinate_space in candidates:
        values = payload.get(field)
        if isinstance(values, list):
            return values, field, coordinate_space
    return None, "missing", "missing"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token_gap_vectors", type=Path, help="Path to token_gap_vectors.jsonl")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output collision JSONL path")
    parser.add_argument("--output-md", type=Path, required=True, help="Output markdown summary path")
    parser.add_argument("--top-mass-fraction", type=float, default=DEFAULT_TOP_MASS_FRACTION)
    parser.add_argument("--random-trials", type=int, default=DEFAULT_RANDOM_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    vectors = read_gap_vectors(args.token_gap_vectors)
    result = compute_collision_result(
        vectors,
        top_mass_fraction=args.top_mass_fraction,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    write_jsonl(args.output_jsonl, result.rows)
    write_markdown(args.output_md, result.rows, args.token_gap_vectors, result.skipped_pairs)
    print(
        json.dumps(
            {
                "input_vectors": len(vectors),
                "pairs": len(result.rows),
                "skipped_pairs": len(result.skipped_pairs),
                "output_jsonl": str(args.output_jsonl),
                "output_md": str(args.output_md),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
