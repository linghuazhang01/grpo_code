#!/usr/bin/env python3
"""Prepare Nemotron-3 Nano RL blend data for local M2RL GRPO recipes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from grpo.data.m2rl import m2rl_frame_to_verl


DEFAULT_INPUT = Path("data/raw/nemotron-3-nano-rl-training-blend/train.jsonl")
DEFAULT_SPLIT_DIR = Path("data/nemotron_rl/splits")
DEFAULT_MANIFEST = Path("data/nemotron_rl/manifest.json")
DEFAULT_IF_OUTPUT = Path("data/M2RL/if/train.parquet")
DEFAULT_SCIENCE_OUTPUT = Path("data/M2RL/science/train.parquet")

IF_CATEGORY = "nano_v3_sft_profiled_instruction_following"
SCIENCE_CATEGORY = "nano_v3_sft_profiled_stem_mcqa"
STRUCTURED_OUTPUTS_CATEGORY = "nano_v3_sft_profiled_structured_outputs"


@dataclass(frozen=True)
class SplitOutputs:
    manifest_path: Path
    raw_split_paths: dict[str, Path]
    verl_paths: dict[str, Path]


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _category(row: Mapping[str, Any]) -> str:
    category = row.get("category") or row.get("dataset")
    if category:
        return str(category)
    if row.get("instruction_id_list") and row.get("kwargs"):
        return "nvidia/Nemotron-RL-instruction_following"
    if row.get("expected_answer") and row.get("options"):
        return "nvidia/Nemotron-RL-knowledge-mcqa"
    return "unknown"


def _domain_from_row(row: Mapping[str, Any]) -> str:
    category = _category(row)
    lowered = category.lower()
    if row.get("instruction_id_list") and row.get("kwargs"):
        return "if"
    if row.get("expected_answer") and row.get("options"):
        return "science"
    if category == IF_CATEGORY:
        return "if"
    if category == SCIENCE_CATEGORY:
        return "science"
    if category == STRUCTURED_OUTPUTS_CATEGORY:
        return "structured_outputs"
    if "comp_coding" in lowered or "coding" in lowered:
        return "coding"
    if "dapo" in lowered or "skywork" in lowered or "math" in lowered:
        return "math"
    if "workbench" in lowered or "agent" in lowered:
        return "agent"
    return "unknown"


def _messages_from_responses_create_params(row: Mapping[str, Any]) -> list[dict[str, str]]:
    params = row.get("responses_create_params")
    if not isinstance(params, Mapping):
        return []
    raw_input = params.get("input")
    if not isinstance(raw_input, Sequence) or isinstance(raw_input, (str, bytes, bytearray)):
        return []

    messages: list[dict[str, str]] = []
    for item in raw_input:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if content is None:
            continue
        messages.append({"role": str(item.get("role") or "user"), "content": str(content)})
    return messages


def _prompt_from_row(row: Mapping[str, Any]) -> str | list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    messages = _messages_from_responses_create_params(row)
    if messages:
        return messages
    raise ValueError(f"row {row.get('id')} is missing prompt text/messages")


def _last_user_text(prompt: str | Sequence[Mapping[str, str]]) -> str:
    if isinstance(prompt, str):
        return prompt
    user_messages = [str(item.get("content", "")) for item in prompt if item.get("role") == "user"]
    if user_messages:
        return user_messages[-1]
    return "\n".join(str(item.get("content", "")) for item in prompt)


def _metadata_base(row: Mapping[str, Any], domain: str, row_index: int) -> dict[str, Any]:
    return {
        "record_id": row.get("id"),
        "uuid": row.get("uuid"),
        "hash_id": row.get("hash_id"),
        "source_row_index": row_index,
        "dataset": row.get("dataset"),
        "source": row.get("source"),
        "original_category": _category(row),
        "original_domain": domain,
        "pass_rate": row.get("pass_rate"),
        "pass_rate_total": row.get("pass_rate_total"),
        "pass_rate_passed": row.get("pass_rate_passed"),
    }


def _unique_record_id(value: Any, row_index: int) -> str:
    base = "row" if value is None else str(value)
    return f"{base}:{row_index}"


def _if_row(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    prompt = _prompt_from_row(row)
    metadata = _metadata_base(row, "if", row_index)
    prompt_text = row.get("prompt")
    metadata.update(
        {
            "instruction_id_list": row.get("instruction_id_list"),
            "kwargs": row.get("kwargs"),
            "prompt_text": str(prompt_text or _last_user_text(prompt)),
        }
    )
    return {"prompt": prompt, "label": "", "record_id": _unique_record_id(row.get("id"), row_index), "metadata": metadata}


def _choice_label(choice: Mapping[str, Any]) -> str | None:
    for key in choice:
        if choice.get(key) is not None:
            return str(key)
    return None


def _choice_text(choice: Mapping[str, Any]) -> str | None:
    for value in choice.values():
        if value is not None:
            return str(value)
    return None


def _choices_from_options(raw_options: Any) -> tuple[list[str], list[str]]:
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes, bytearray)):
        return [], []

    pairs: list[tuple[str, str]] = []
    for option in raw_options:
        if not isinstance(option, Mapping):
            continue
        label = _choice_label(option)
        text = _choice_text(option)
        if label is None or text is None:
            continue
        pairs.append((label.upper(), text))

    pairs.sort(key=lambda item: item[0])
    return [label for label, _ in pairs], [text for _, text in pairs]


def _science_row(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    prompt = _prompt_from_row(row)
    labels, choices = _choices_from_options(row.get("options"))
    correct_letter = str(row.get("expected_answer") or "").strip().upper()
    if not labels:
        raise ValueError("science row is missing usable option labels")
    if correct_letter not in labels:
        raise ValueError(
            f"science expected_answer {correct_letter!r} is not present in option labels {labels!r}"
        )
    metadata = _metadata_base(row, "science", row_index)
    metadata.update(
        {
            "choices": choices,
            "valid_letters": labels,
            "correct_letter": correct_letter,
            "template_metadata": row.get("template_metadata"),
            "verifier_metadata": row.get("verifier_metadata"),
        }
    )
    return {
        "prompt": prompt,
        "label": correct_letter,
        "record_id": _unique_record_id(row.get("uuid") or row.get("id"), row_index),
        "metadata": metadata,
    }


def _open_split_handles(split_dir: Path, domains: Sequence[str]) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        domain: (split_dir / f"{domain}.jsonl").open("w", encoding="utf-8")
        for domain in domains
    }


def _close_handles(handles: Mapping[str, Any]) -> None:
    for handle in handles.values():
        handle.close()


def prepare_nemotron_rl_data(
    input_path: Path,
    split_dir: Path,
    manifest_path: Path,
    if_output_path: Path,
    science_output_path: Path,
    *,
    write_raw_splits: bool,
    if_max_samples: int | None,
) -> SplitOutputs:
    domains = ("math", "coding", "science", "if", "agent", "structured_outputs", "unknown")
    handles = _open_split_handles(split_dir, domains) if write_raw_splits else {}
    domain_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    if_rows: list[dict[str, Any]] = []
    science_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    try:
        for row_index, row in enumerate(_iter_jsonl(input_path)):
            domain = _domain_from_row(row)
            category = _category(row)
            domain_counts[domain] += 1
            category_counts[category] += 1

            if write_raw_splits:
                handles[domain].write(json.dumps(row, ensure_ascii=False) + "\n")

            try:
                if domain == "if":
                    if if_max_samples is None or len(if_rows) < if_max_samples:
                        if_rows.append(_if_row(row, row_index))
                elif domain == "science":
                    science_rows.append(_science_row(row, row_index))
            except ValueError as exc:
                invalid_rows.append({"row_index": row_index, "domain": domain, "reason": str(exc)})
    finally:
        _close_handles(handles)

    if write_raw_splits:
        for handle_path in (split_dir / f"{domain}.jsonl" for domain in domains):
            if handle_path.exists() and handle_path.stat().st_size == 0:
                handle_path.unlink()

    verl_paths: dict[str, Path] = {}
    if if_rows:
        if_output_path.parent.mkdir(parents=True, exist_ok=True)
        if_frame = pd.DataFrame(if_rows)
        if_verl = m2rl_frame_to_verl(if_frame, rm_type="ifbench", split="train", domain="if")
        if_verl.to_parquet(if_output_path, index=False)
        verl_paths["if"] = if_output_path

    if science_rows:
        science_output_path.parent.mkdir(parents=True, exist_ok=True)
        science_frame = pd.DataFrame(science_rows)
        science_verl = m2rl_frame_to_verl(science_frame, rm_type="gpqa", split="train", domain="science")
        science_verl.to_parquet(science_output_path, index=False)
        verl_paths["science"] = science_output_path

    raw_split_paths = {
        domain: split_dir / f"{domain}.jsonl"
        for domain in domains
        if write_raw_splits and (split_dir / f"{domain}.jsonl").exists()
    }
    manifest = {
        "input_path": str(input_path),
        "domain_counts": dict(sorted(domain_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "raw_split_paths": {key: str(path) for key, path in sorted(raw_split_paths.items())},
        "verl_paths": {key: str(path) for key, path in sorted(verl_paths.items())},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SplitOutputs(manifest_path=manifest_path, raw_split_paths=raw_split_paths, verl_paths=verl_paths)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Downloaded Nemotron RL blend JSONL.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR, help="Directory for raw domain JSONL splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output manifest path.")
    parser.add_argument("--if-output", type=Path, default=DEFAULT_IF_OUTPUT, help="Output verl IF train parquet.")
    parser.add_argument(
        "--science-output",
        type=Path,
        default=DEFAULT_SCIENCE_OUTPUT,
        help="Output verl science train parquet.",
    )
    parser.add_argument("--write-raw-splits", action="store_true", help="Write raw JSONL split files for all domains.")
    parser.add_argument(
        "--if-max-samples",
        type=int,
        default=None,
        help="Optional cap for IF rows written to the verl train parquet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = prepare_nemotron_rl_data(
        args.input,
        args.split_dir,
        args.manifest,
        args.if_output,
        args.science_output,
        write_raw_splits=args.write_raw_splits,
        if_max_samples=args.if_max_samples,
    )
    print(
        json.dumps(
            {
                "manifest": str(outputs.manifest_path),
                "raw_split_paths": {key: str(path) for key, path in sorted(outputs.raw_split_paths.items())},
                "verl_paths": {key: str(path) for key, path in sorted(outputs.verl_paths.items())},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
