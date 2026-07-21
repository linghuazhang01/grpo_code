"""Convert and validate M2RL IFBench/Science GRPO data for the local verl launcher."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_RM_TYPES = {"ifbench", "gpqa"}


@dataclass(frozen=True)
class M2RLSchemaReport:
    count: int
    rm_type: str
    invalid_rows: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_rows

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "rm_type": self.rm_type, "invalid_rows": self.invalid_rows}


def _load_json(value: str) -> Any:
    return json.loads(value)


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        parsed = _load_json(stripped)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalize_valid_letters(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return _normalize_valid_letters(_load_json(stripped))
        except json.JSONDecodeError:
            entries = (
                list(stripped)
                if re.fullmatch(r"[A-Za-z]+", stripped)
                else re.split(r"[\s,;/|]+", stripped)
            )
    elif isinstance(value, Mapping):
        entries = list(value.keys())
    elif isinstance(value, Sequence):
        entries = list(value)
    else:
        try:
            entries = list(value)
        except TypeError:
            item_method = getattr(value, "item", None)
            if not callable(item_method):
                entries = [value]
            else:
                scalar = item_method()
                return [] if scalar is value else _normalize_valid_letters(scalar)

    normalized: list[str] = []
    for entry in entries:
        text = str(entry).strip().upper()
        if len(text) == 1 and text in string.ascii_uppercase and text not in normalized:
            normalized.append(text)
    return normalized


def _choice_count(value: Any) -> int:
    if isinstance(value, str):
        try:
            value = _load_json(value)
        except json.JSONDecodeError:
            return 0
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Iterable):
        try:
            return len(value)
        except TypeError:
            return len(list(value))
    return 0


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str, Mapping)):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "user")
            content = item.get("content")
            if content is None:
                continue
            messages.append({"role": role, "content": str(content)})
        if messages:
            return messages
    raise ValueError("row is missing a usable prompt/messages value")


def _prompt_value(row: Mapping[str, Any]) -> Any:
    for key in ("prompt", "messages", "question", "input"):
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError("row is missing prompt/messages/question/input")


def _prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    user_messages = [str(item.get("content", "")) for item in messages if item.get("role") == "user"]
    if user_messages:
        return user_messages[-1]
    return "\n".join(str(item.get("content", "")) for item in messages)


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def _metadata_from_row(row: Mapping[str, Any], rm_type: str, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    metadata = _normalize_mapping(row.get("metadata"))
    extra_info = _normalize_mapping(row.get("extra_info"))
    metadata.update(extra_info)
    metadata["rm_type"] = rm_type

    for key in (
        "instruction_id_list",
        "kwargs",
        "prompt_text",
        "record_id",
        "key",
        "choices",
        "valid_letters",
        "correct_letter",
        "correct_answer",
        "answer_text",
    ):
        if key in row and row[key] is not None:
            metadata.setdefault(key, row[key])

    metadata.setdefault("prompt_text", _prompt_text(messages))
    return metadata


def _label_from_row(row: Mapping[str, Any]) -> Any:
    reward_model = _normalize_mapping(row.get("reward_model"))
    if "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    return _first_present(
        row,
        (
            "label",
            "ground_truth",
            "answer",
            "target",
            "correct_letter",
            "correct_answer",
            "answer_text",
        ),
    )


def _sample_id(row: Mapping[str, Any], row_position: int, rm_type: str, domain: str) -> str:
    metadata = _normalize_mapping(row.get("metadata"))
    row_id = _first_present(row, ("id", "key", "record_id")) or metadata.get("id") or metadata.get("key")
    if row_id is None:
        row_id = row_position
    return f"{domain}:{rm_type}:{row_id}"


def m2rl_frame_to_verl(
    frame: pd.DataFrame,
    *,
    rm_type: str,
    split: str,
    domain: str,
    data_source: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Normalize an M2RL-style dataframe into the verl parquet schema."""

    if rm_type not in SUPPORTED_RM_TYPES:
        raise ValueError(f"rm_type must be one of {sorted(SUPPORTED_RM_TYPES)}, got {rm_type!r}")
    if max_samples is not None:
        frame = frame.head(max_samples)

    rows: list[dict[str, Any]] = []
    source = data_source or f"m2rl_{rm_type}"
    for row_position, (_, row) in enumerate(frame.iterrows()):
        row_dict = dict(row)
        messages = _normalize_messages(_prompt_value(row_dict))
        label = _label_from_row(row_dict)
        metadata = _metadata_from_row(row_dict, rm_type, messages)
        metadata.update(
            {
                "opd_teacher": domain,
                "domain": domain,
                "source_domain": domain,
                "split": split,
                "sample_id": metadata.get("sample_id") or _sample_id(row_dict, row_position, rm_type, domain),
            }
        )
        rows.append(
            {
                "data_source": source,
                "prompt": messages,
                "ability": domain,
                "reward_model": {"style": "rule", "ground_truth": "" if label is None else label},
                "extra_info": metadata,
            }
        )
    return pd.DataFrame(rows)


def _row_invalid_reasons(row: Mapping[str, Any], rm_type: str) -> list[str]:
    reasons: list[str] = []
    try:
        messages = _normalize_messages(_prompt_value(row))
    except ValueError as exc:
        return [str(exc)]
    metadata = _metadata_from_row(row, rm_type, messages)
    label = _label_from_row(row)

    if rm_type == "ifbench":
        instruction_ids = metadata.get("instruction_id_list")
        if isinstance(instruction_ids, str):
            instruction_ids = [instruction_ids]
        elif isinstance(instruction_ids, Iterable):
            instruction_ids = [item for item in instruction_ids if item is not None]
        else:
            instruction_ids = []
        if len(instruction_ids) == 0:
            reasons.append("missing IFBench instruction_id_list metadata")
        if not metadata.get("prompt_text"):
            reasons.append("missing IFBench prompt_text metadata")
    elif rm_type == "gpqa":
        choices = metadata.get("choices")
        correct_letter = metadata.get("correct_letter")
        has_label = label is not None
        if not correct_letter and not has_label:
            reasons.append("missing GPQA correct_letter or label/answer")
        if choices is None and not correct_letter and not (isinstance(label, str) and len(label.strip()) == 1):
            reasons.append("missing GPQA choices for non-letter label")
        valid_letters = _normalize_valid_letters(metadata.get("valid_letters"))
        if not valid_letters:
            valid_letters = list(string.ascii_uppercase[: _choice_count(choices)])
        if isinstance(correct_letter, str):
            normalized_correct_letter = correct_letter.strip().upper()
        elif isinstance(label, str) and len(label.strip()) == 1:
            normalized_correct_letter = label.strip().upper()
        else:
            normalized_correct_letter = ""
        if valid_letters and normalized_correct_letter and normalized_correct_letter not in valid_letters:
            reasons.append(
                "GPQA correct_letter is not included in valid_letters: "
                f"{normalized_correct_letter!r} not in {valid_letters!r}"
            )
    else:
        reasons.append(f"unsupported rm_type {rm_type!r}")
    return reasons


def validate_m2rl_frame(frame: pd.DataFrame, *, rm_type: str) -> M2RLSchemaReport:
    invalid_rows: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(frame.iterrows()):
        reasons = _row_invalid_reasons(dict(row), rm_type)
        if reasons:
            invalid_rows.append({"index": row_position, "reasons": reasons})
    return M2RLSchemaReport(count=len(frame), rm_type=rm_type, invalid_rows=invalid_rows)


def read_frame(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix == ".jsonl":
        return pd.read_json(input_path, lines=True)
    if input_path.suffix == ".json":
        return pd.read_json(input_path)
    raise ValueError(f"Unsupported input format: {input_path}")


def validate_m2rl_parquet(path: str | Path, *, rm_type: str) -> M2RLSchemaReport:
    return validate_m2rl_frame(read_frame(path), rm_type=rm_type)


def m2rl_to_verl_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    rm_type: str,
    split: str,
    domain: str,
    data_source: str | None = None,
    max_samples: int | None = None,
) -> M2RLSchemaReport:
    frame = read_frame(input_path)
    report = validate_m2rl_frame(frame.head(max_samples) if max_samples is not None else frame, rm_type=rm_type)
    if not report.is_valid:
        return report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = m2rl_frame_to_verl(
        frame,
        rm_type=rm_type,
        split=split,
        domain=domain,
        data_source=data_source,
        max_samples=max_samples,
    )
    normalized.to_parquet(output, index=False)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert M2RL-style parquet/json/jsonl into verl parquet.")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--rm-type", required=True, choices=sorted(SUPPORTED_RM_TYPES))
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--domain", default=None)
    prepare.add_argument("--data-source", default=None)
    prepare.add_argument("--max-samples", type=int, default=None)

    validate = subparsers.add_parser("validate", help="Validate an M2RL-style file before training.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--rm-type", required=True, choices=sorted(SUPPORTED_RM_TYPES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "validate":
        report = validate_m2rl_parquet(args.input, rm_type=args.rm_type)
    else:
        domain = args.domain or ("if" if args.rm_type == "ifbench" else "science")
        report = m2rl_to_verl_parquet(
            args.input,
            args.output,
            rm_type=args.rm_type,
            split=args.split,
            domain=domain,
            data_source=args.data_source,
            max_samples=args.max_samples,
        )
    sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
