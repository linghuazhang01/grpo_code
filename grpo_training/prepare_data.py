"""Prepare parquet files for standalone GRPO teacher training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from grpo.data.m2rl import SUPPORTED_RM_TYPES, m2rl_to_verl_parquet

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

VALID_TEACHERS = {"if", "science"}


@dataclass(frozen=True)
class TeacherValidation:
    counts: dict[str, int]
    invalid_rows: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_rows


@dataclass(frozen=True)
class SampleIdValidation:
    duplicate_count: int
    invalid_rows: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return self.duplicate_count == 0 and not self.invalid_rows


def _normalize_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"Unsupported extra_info value: {value!r}")


def _iter_extra_info(frame: pd.DataFrame):
    if "extra_info" not in frame.columns:
        for index in frame.index:
            yield int(index), {}
        return
    for index, value in frame["extra_info"].items():
        try:
            yield int(index), _normalize_extra_info(value)
        except ValueError:
            yield int(index), {}


def validate_teacher_labels(path: str | Path) -> TeacherValidation:
    frame = pd.read_parquet(Path(path))
    counts = {teacher: 0 for teacher in sorted(VALID_TEACHERS)}
    invalid_rows: list[dict[str, Any]] = []

    if "extra_info" not in frame.columns:
        return TeacherValidation(
            counts=counts,
            invalid_rows=[
                {"index": int(index), "teacher": None, "reason": "missing extra_info"}
                for index in frame.index
            ],
        )

    for index, value in frame["extra_info"].items():
        try:
            teacher = _normalize_extra_info(value).get("opd_teacher")
        except ValueError as exc:
            invalid_rows.append({"index": int(index), "teacher": None, "reason": str(exc)})
            continue
        if teacher in counts:
            counts[teacher] += 1
        else:
            invalid_rows.append({"index": int(index), "teacher": teacher, "reason": "invalid or missing opd_teacher"})

    return TeacherValidation(counts=counts, invalid_rows=invalid_rows)


def validate_sample_ids(path: str | Path) -> SampleIdValidation:
    frame = pd.read_parquet(Path(path))
    invalid_rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    duplicate_count = 0

    for index, extra_info in _iter_extra_info(frame):
        sample_id = extra_info.get("sample_id")
        teacher = extra_info.get("opd_teacher")
        domain = extra_info.get("domain")
        if not sample_id:
            invalid_rows.append({"index": index, "sample_id": sample_id, "reason": "missing sample_id"})
            continue
        sample_id = str(sample_id)
        if sample_id in seen:
            duplicate_count += 1
            invalid_rows.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason": f"duplicate sample_id first seen at row {seen[sample_id]}",
                }
            )
        else:
            seen[sample_id] = index
        if teacher in VALID_TEACHERS and domain != teacher:
            invalid_rows.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason": f"domain {domain!r} does not match opd_teacher {teacher!r}",
                }
            )

    return SampleIdValidation(duplicate_count=duplicate_count, invalid_rows=invalid_rows)


def _validation_payload(path: str | Path, count: int | None = None) -> dict[str, Any]:
    teacher_validation = validate_teacher_labels(path)
    sample_validation = validate_sample_ids(path)
    payload: dict[str, Any] = {
        "counts": teacher_validation.counts,
        "invalid_rows": teacher_validation.invalid_rows[:20],
        "sample_id_duplicate_count": sample_validation.duplicate_count,
        "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
    }
    if count is not None:
        payload["count"] = count
    return payload


def _is_valid_payload(payload: Mapping[str, Any]) -> bool:
    return (
        not payload["invalid_rows"]
        and int(payload["sample_id_duplicate_count"]) == 0
        and not payload["sample_id_invalid_rows"]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate GRPO parquet metadata.")
    inspect_parser.add_argument("path", help="Parquet file to inspect.")

    m2rl_parser = subparsers.add_parser(
        "prepare-m2rl",
        help="Convert M2RL IFBench/Science parquet, JSON, or JSONL to verl schema.",
    )
    m2rl_parser.add_argument("--input", required=True, help="Input .parquet, .json, or .jsonl file.")
    m2rl_parser.add_argument("--output", required=True, help="Output verl parquet file.")
    m2rl_parser.add_argument("--rm-type", required=True, choices=sorted(SUPPORTED_RM_TYPES))
    m2rl_parser.add_argument("--split", default="train", help="Split name to record in extra_info.")
    m2rl_parser.add_argument("--domain", default=None, choices=["if", "science"], help="Domain/teacher label.")
    m2rl_parser.add_argument("--teacher", default=None, choices=["if", "science"], help="Alias for --domain.")
    m2rl_parser.add_argument("--data-source", default=None, help="Data source tag for reward dispatch.")
    m2rl_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "inspect":
        payload = _validation_payload(args.path)
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if _is_valid_payload(payload) else 1

    if args.command == "prepare-m2rl":
        domain = args.domain or args.teacher or ("if" if args.rm_type == "ifbench" else "science")
        report = m2rl_to_verl_parquet(
            args.input,
            args.output,
            rm_type=args.rm_type,
            split=args.split,
            domain=domain,
            data_source=args.data_source,
            max_samples=args.max_samples,
        )
        if not report.is_valid:
            sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
            return 1
        payload = _validation_payload(args.output, count=report.count)
        payload["m2rl_schema"] = report.to_dict()
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if _is_valid_payload(payload) else 1

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
