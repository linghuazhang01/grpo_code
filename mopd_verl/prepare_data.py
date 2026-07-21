"""Prepare verl parquet files with domain-specific MOPD teacher routing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from grpo.data.m2rl import SUPPORTED_RM_TYPES, m2rl_to_verl_parquet
from mopd_verl.general_reasoner_data import (
    DEFAULT_DATASET_NAME as GENERAL_REASONER_DATASET_NAME,
    general_reasoner_to_verl_parquet,
    prepare_general_reasoner_hf_dataset,
)
from mopd_verl.searchqa_data import searchqa_to_verl_parquet

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

VALID_TEACHERS = {"math", "code", "reasoning", "search", "tool", "if", "science"}


def _load_optional_module(module_name: str, command: str) -> ModuleType:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name
        target_is_missing = missing_name == module_name or (
            missing_name is not None and module_name.startswith(f"{missing_name}.")
        )
        if not target_is_missing:
            raise
        raise RuntimeError(
            f"{command} requires the optional {module_name} module."
        ) from exc


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
        return not self.invalid_rows and self.duplicate_count == 0


def _normalize_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"Unsupported extra_info value: {value!r}")


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def _stable_sample_id(row: Mapping[str, Any], teacher: str, row_position: int, extra_info: Mapping[str, Any]) -> str:
    data_source = str(row.get("data_source", "unknown")).replace("/", "_")
    index = extra_info.get("index", row_position)
    return f"{teacher}:{data_source}:{index}"


def add_teacher_column(frame: pd.DataFrame, teacher: str) -> pd.DataFrame:
    if teacher not in VALID_TEACHERS:
        raise ValueError(f"teacher must be one of {sorted(VALID_TEACHERS)}, got {teacher!r}")

    result = frame.copy(deep=True)
    if "extra_info" not in result.columns:
        result["extra_info"] = [{} for _ in range(len(result))]

    extra_info = []
    for row_position, (_, row) in enumerate(result.iterrows()):
        value = row.get("extra_info")
        normalized = _normalize_extra_info(value)
        normalized["opd_teacher"] = teacher
        normalized["domain"] = teacher
        normalized["source_domain"] = teacher
        normalized.setdefault("sample_id", _stable_sample_id(row, teacher, row_position, normalized))
        extra_info.append(normalized)

    result["extra_info"] = extra_info
    return result


def merge_teacher_data(math_path: str | Path, code_path: str | Path, output_path: str | Path) -> None:
    math_frame = add_teacher_column(read_parquet(math_path), "math")
    code_frame = add_teacher_column(read_parquet(code_path), "code")
    merged = pd.concat([math_frame, code_frame], ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)


def teacher_counts(path: str | Path) -> dict[str, int]:
    return validate_teacher_labels(path).counts


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
    frame = read_parquet(path)
    counts = {teacher: 0 for teacher in sorted(VALID_TEACHERS)}
    invalid_rows: list[dict[str, Any]] = []

    if "extra_info" not in frame.columns:
        return TeacherValidation(
            counts=counts,
            invalid_rows=[{"index": int(index), "teacher": None, "reason": "missing extra_info"} for index in frame.index],
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
    frame = read_parquet(path)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge math/code parquet files and add opd_teacher.")
    merge_parser.add_argument("--math-train", required=True, help="Math-domain training parquet.")
    merge_parser.add_argument("--code-train", required=True, help="Code-domain training parquet.")
    merge_parser.add_argument("--output", required=True, help="Output merged parquet.")

    inspect_parser = subparsers.add_parser("inspect", help="Count opd_teacher labels in a parquet file.")
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

    paper_eval_parser = subparsers.add_parser(
        "prepare-paper-eval",
        help="Convert paper math eval JSONL files into verl validation parquet files.",
    )
    paper_eval_parser.add_argument("--gopd-dir", required=True, help="Path to the G-OPD checkout root.")
    paper_eval_parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for generated parquets. Defaults to <gopd-dir>/eval/domains.",
    )
    searchqa_parser = subparsers.add_parser(
        "prepare-searchqa",
        help="Convert SearchQA/Search-R1-style parquet or JSONL into verl parquet format.",
    )
    searchqa_parser.add_argument("--input", required=True, help="Input .parquet or .jsonl file.")
    searchqa_parser.add_argument("--output", required=True, help="Output verl parquet file.")
    searchqa_parser.add_argument("--split", default="train", help="Split name to record in extra_info.")
    searchqa_parser.add_argument(
        "--data-source",
        default=None,
        help="Default base data source, e.g. nq or hotpotqa. Existing row data_source values take precedence.",
    )
    searchqa_parser.add_argument("--teacher", default="search", choices=sorted(VALID_TEACHERS), help="OPD teacher label.")
    searchqa_parser.add_argument(
        "--data-source-prefix",
        default="searchR1",
        help="Prefix used for reward dispatch, e.g. searchR1 produces searchR1_nq.",
    )
    general_reasoner_parser = subparsers.add_parser(
        "prepare-general-reasoner",
        help="Convert General-Reasoner/WebInstruct JSONL, JSON, or parquet into verl parquet format.",
    )
    general_reasoner_parser.add_argument("--input", required=True, help="Input .parquet, .json, or .jsonl file.")
    general_reasoner_parser.add_argument("--output", required=True, help="Output verl parquet file.")
    general_reasoner_parser.add_argument("--split", default="train", help="Split name to record in extra_info.")
    general_reasoner_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests.")

    general_reasoner_hf_parser = subparsers.add_parser(
        "prepare-general-reasoner-hf",
        help="Download TIGER-Lab/WebInstruct-verified and write train/test verl parquet files.",
    )
    general_reasoner_hf_parser.add_argument("--dataset-name", default=GENERAL_REASONER_DATASET_NAME)
    general_reasoner_hf_parser.add_argument(
        "--output-dir",
        default="data/GeneralReasoner/WebInstructVerified",
        help="Directory for train.parquet/test.parquet outputs.",
    )
    general_reasoner_hf_parser.add_argument(
        "--test-max-samples",
        type=int,
        default=100,
        help="Cap test/validation examples for validation cost control. Use -1 for all.",
    )
    toolrl_parser = subparsers.add_parser(
        "prepare-toolrl",
        help="Convert ToolRL/RLLA parquet into the shared verl parquet format.",
    )
    toolrl_parser.add_argument("--input", required=True, help="Input ToolRL .parquet file.")
    toolrl_parser.add_argument("--output", required=True, help="Output verl parquet file.")
    toolrl_parser.add_argument("--split", default="train", help="Split name to record in extra_info.")
    toolrl_parser.add_argument("--teacher", default="tool", choices=sorted(VALID_TEACHERS), help="OPD teacher label.")
    toolrl_parser.add_argument("--data-source", default="toolrl_rlla", help="Data source tag for reward logging.")
    toolrl_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
        validation = validate_teacher_labels(args.output)
        sample_validation = validate_sample_ids(args.output)
        payload = {
            "count": report.count,
            "counts": validation.counts,
            "invalid_rows": validation.invalid_rows[:20],
            "sample_id_duplicate_count": sample_validation.duplicate_count,
            "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
            "m2rl_schema": report.to_dict(),
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if validation.is_valid and sample_validation.is_valid else 1
    if args.command == "prepare-paper-eval":
        paper_eval_module = _load_optional_module(
            "eval.data_prep.paper_eval", "prepare-paper-eval"
        )
        counts = paper_eval_module.prepare_paper_eval_data(args.gopd_dir, args.output_root)
        sys.stdout.write(json.dumps({"counts": counts}, sort_keys=True) + "\n")
        return 0
    if args.command == "prepare-searchqa":
        count = searchqa_to_verl_parquet(
            args.input,
            args.output,
            split=args.split,
            data_source=args.data_source,
            teacher=args.teacher,
            data_source_prefix=args.data_source_prefix,
        )
        validation = validate_teacher_labels(args.output)
        sample_validation = validate_sample_ids(args.output)
        payload = {
            "count": count,
            "counts": validation.counts,
            "invalid_rows": validation.invalid_rows[:20],
            "sample_id_duplicate_count": sample_validation.duplicate_count,
            "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if validation.is_valid and sample_validation.is_valid else 1
    if args.command == "prepare-general-reasoner":
        count = general_reasoner_to_verl_parquet(
            args.input,
            args.output,
            split=args.split,
            max_samples=args.max_samples,
        )
        validation = validate_teacher_labels(args.output)
        sample_validation = validate_sample_ids(args.output)
        payload = {
            "count": count,
            "counts": validation.counts,
            "invalid_rows": validation.invalid_rows[:20],
            "sample_id_duplicate_count": sample_validation.duplicate_count,
            "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if validation.is_valid and sample_validation.is_valid else 1
    if args.command == "prepare-general-reasoner-hf":
        counts = prepare_general_reasoner_hf_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            test_max_samples=args.test_max_samples,
        )
        payload = {"counts": counts, "output_dir": args.output_dir}
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0
    if args.command == "prepare-toolrl":
        toolrl_module = _load_optional_module("grpo.data.toolrl", "prepare-toolrl")
        count = toolrl_module.toolrl_to_verl_parquet(
            args.input,
            args.output,
            split=args.split,
            teacher=args.teacher,
            data_source=args.data_source,
            max_samples=args.max_samples,
        )
        validation = validate_teacher_labels(args.output)
        sample_validation = validate_sample_ids(args.output)
        payload = {
            "count": count,
            "counts": validation.counts,
            "invalid_rows": validation.invalid_rows[:20],
            "sample_id_duplicate_count": sample_validation.duplicate_count,
            "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0 if validation.is_valid and sample_validation.is_valid else 1

    if args.command == "merge":
        merge_teacher_data(args.math_train, args.code_train, args.output)
        validation = validate_teacher_labels(args.output)
    else:
        validation = validate_teacher_labels(args.path)

    sample_validation = validate_sample_ids(args.output if args.command == "merge" else args.path)
    payload = {
        "counts": validation.counts,
        "invalid_rows": validation.invalid_rows[:20],
        "sample_id_duplicate_count": sample_validation.duplicate_count,
        "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0 if validation.is_valid and sample_validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
