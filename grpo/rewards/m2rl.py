"""M2RL-style IFBench and GPQA reward functions for verl GRPO."""

from __future__ import annotations

import importlib
import json
import os
import re
import string
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])
IFBENCH_PASS_SCORE = 1.0
IFBENCH_FAIL_SCORE = 0.0


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalize_instruction_ids(raw_ids: Any) -> list[str]:
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, Sequence):
        return []
    output: list[str] = []
    for entry in raw_ids:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            output.append(text)
    return output


def _coerce_kwargs_list(raw_kwargs: Any, num_instructions: int) -> list[dict[str, Any]]:
    if isinstance(raw_kwargs, str):
        try:
            raw_kwargs = json.loads(raw_kwargs)
        except json.JSONDecodeError:
            raw_kwargs = None

    if isinstance(raw_kwargs, list):
        processed = [dict(item) if isinstance(item, Mapping) else {} for item in raw_kwargs]
    elif isinstance(raw_kwargs, Mapping):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    return [{key: value for key, value in item.items() if value is not None} for item in processed]


def _candidate_ifbench_paths() -> list[Path]:
    candidates: list[Path] = []
    for name in ("IFBENCH_REPO", "M2RL_IFBENCH_REPO"):
        value = os.getenv(name)
        if value:
            candidates.append(Path(value).expanduser())

    current = Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            current / "IFBench",
            current.parent / "IFBench",
            repo_root / "IFBench",
            repo_root / "temp" / "IFBench",
            repo_root.parent / "IFBench",
        ]
    )
    return candidates


def _configure_ifbench_data_path(path: Path) -> None:
    nltk_data_dir = path / ".nltk_data"
    if not nltk_data_dir.exists():
        return

    existing = os.getenv("NLTK_DATA", "")
    paths = [item for item in existing.split(os.pathsep) if item]
    data_path = str(nltk_data_dir)
    if data_path not in paths:
        os.environ["NLTK_DATA"] = os.pathsep.join([data_path, *paths])


def _ensure_ifbench_importable() -> Any:
    for path in _candidate_ifbench_paths():
        if (path / "evaluation_lib.py").exists():
            _configure_ifbench_data_path(path)

    try:
        return importlib.import_module("evaluation_lib")
    except ImportError:
        pass

    for path in _candidate_ifbench_paths():
        if (path / "evaluation_lib.py").exists():
            _configure_ifbench_data_path(path)
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return importlib.import_module("evaluation_lib")

    if os.getenv("M2RL_ALLOW_IFBENCH_AUTO_CLONE", "0") == "1":
        target = Path(os.getenv("M2RL_IFBENCH_REPO", str(Path.cwd() / "IFBench"))).expanduser()
        if not target.exists():
            subprocess.run(["git", "clone", "https://github.com/allenai/IFBench.git", str(target)], check=True)
        target_str = str(target)
        if target_str not in sys.path:
            sys.path.insert(0, target_str)
        _configure_ifbench_data_path(target)
        return importlib.import_module("evaluation_lib")

    raise RuntimeError(
        "IFBench reward requires allenai/IFBench. Set IFBENCH_REPO to a local clone, "
        "or set M2RL_ALLOW_IFBENCH_AUTO_CLONE=1 to allow cloning during startup."
    )


def compute_ifbench_reward(response: str, metadata: Mapping[str, Any] | None = None) -> float:
    """Score a response with official IFBench strict instruction-following rules."""

    if not response or metadata is None:
        return IFBENCH_FAIL_SCORE

    evaluation_lib = _ensure_ifbench_importable()
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return IFBENCH_FAIL_SCORE

    prompt_text = str(metadata.get("prompt_text") or metadata.get("prompt") or "")
    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))
    raw_record_id = metadata.get("record_id") or metadata.get("key") or 0
    try:
        record_id = int(raw_record_id)
    except (TypeError, ValueError):
        record_id = str(raw_record_id)
    input_example = evaluation_lib.InputExample(
        key=record_id,
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )
    result = evaluation_lib.test_instruction_following_strict(input_example, {prompt_text: response})
    return IFBENCH_PASS_SCORE if result.follow_all_instructions else IFBENCH_FAIL_SCORE


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _letter_from_match(match: re.Match[str], valid_letters: set[str]) -> str | None:
    for group_index in range(1, len(match.groups()) + 1):
        value = match.group(group_index)
        if value is None:
            continue
        letter = value.strip().upper()
        if len(letter) != 1 or letter not in valid_letters:
            continue
        start, end = match.span(group_index)
        previous_char = match.string[start - 1] if start > 0 else ""
        next_char = match.string[end] if end < len(match.string) else ""
        if previous_char.isalnum() or previous_char == "_":
            continue
        if next_char.isalnum() or next_char == "_":
            continue
        return letter
    return None


def _extract_letter_from_response(
    response: str,
    valid_letters: Iterable[str],
    output_regex: str | None = None,
) -> str | None:
    if not response:
        return None
    text = _strip_chain_of_thought(response)
    valid_set = {letter.upper() for letter in valid_letters}
    if output_regex:
        try:
            template_matches = re.finditer(output_regex, text, flags=re.IGNORECASE)
            template_candidates = [
                (match.start(), letter)
                for match in template_matches
                if (letter := _letter_from_match(match, valid_set)) is not None
            ]
        except re.error:
            template_candidates = []
        if template_candidates:
            return max(template_candidates, key=lambda item: item[0])[1]

    patterns = [
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"<final_answer>\s*([A-Z])\s*</final_answer>",
        r"(?:final\s*)?(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-Z])(?![A-Z0-9_/])",
        r"\b([A-Z])\b\s+is\s+(?:the\s+)?correct(?:\s+(?:answer|option|choice))?",
    ]
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            letter = _letter_from_match(match, valid_set)
            if letter is not None:
                candidates.append((match.start(), letter))

    fallback_text = re.sub(
        r"\b[A-Z](?:\s*/\s*[A-Z])+\b",
        lambda match: " " * len(match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    for match in re.finditer(r"\b([A-Z])\b", fallback_text):
        letter = match.group(1).upper()
        if letter in valid_set:
            candidates.append((match.start(), letter))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _choices_from_metadata(metadata: Mapping[str, Any]) -> list[Any] | None:
    choices = metadata.get("choices")
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError:
            choices = None
    if isinstance(choices, Mapping):
        return list(choices.values())
    if choices is not None:
        return list(choices)
    return None


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
            return _normalize_valid_letters(json.loads(stripped))
        except json.JSONDecodeError:
            if re.fullmatch(r"[A-Za-z]+", stripped):
                entries: list[Any] = list(stripped)
            else:
                entries = re.split(r"[\s,;/|]+", stripped)
    elif isinstance(value, Mapping):
        entries = list(value.keys())
    elif isinstance(value, Iterable):
        try:
            entries = list(value)
        except TypeError:
            item_method = getattr(value, "item", None)
            if not callable(item_method):
                return []
            scalar = item_method()
            if scalar is value:
                return []
            return _normalize_valid_letters(scalar)
    else:
        entries = [value]

    normalized: list[str] = []
    for entry in entries:
        text = str(entry).strip().upper()
        if len(text) == 1 and text in string.ascii_uppercase and text not in normalized:
            normalized.append(text)
    return normalized


def compute_gpqa_reward(response: str, label: Any, metadata: Mapping[str, Any] | None = None) -> float:
    """Rule-based scorer for GPQA-style multiple-choice science QA."""

    if response is None:
        return 0.0

    metadata = metadata or {}
    choices = _choices_from_metadata(metadata)
    valid_letters = _normalize_valid_letters(metadata.get("valid_letters"))
    if not valid_letters and choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    elif not valid_letters:
        valid_letters = DEFAULT_VALID_LETTERS

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and correct_letter is None:
            correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        label_index = int(label)
        if 0 <= label_index < len(valid_letters):
            correct_letter = valid_letters[label_index]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for index, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[index]
                break

    template_metadata = metadata.get("template_metadata")
    output_regex = template_metadata.get("output_regex") if isinstance(template_metadata, Mapping) else None
    extracted_letter = _extract_letter_from_response(
        response,
        valid_letters,
        str(output_regex) if output_regex else None,
    )
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0
    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0
    return 0.0


def _rm_type(data_source: str, extra_info: Mapping[str, Any] | None) -> str:
    metadata = _normalize_metadata(extra_info)
    raw = metadata.get("rm_type") or metadata.get("reward_type") or data_source
    text = str(raw or "").lower()
    if "ifbench" in text or text in {"if", "instruction_following", "instruction-following"}:
        return "ifbench"
    if "gpqa" in text or "science" in text or "knowledge" in text:
        return "gpqa"
    return text


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Return a verl-compatible reward dict with ``score`` as the primary scalar."""

    metadata = _normalize_metadata(extra_info)
    rm_type = _rm_type(data_source, metadata)
    if rm_type == "ifbench":
        reward = compute_ifbench_reward(str(solution_str or ""), metadata)
        return {"score": float(reward), "m2rl_ifbench": float(reward)}
    if rm_type == "gpqa":
        reward = compute_gpqa_reward(str(solution_str or ""), ground_truth, metadata)
        return {"score": float(reward), "m2rl_gpqa": float(reward)}
    raise NotImplementedError(f"Unsupported M2RL reward type: {rm_type!r}")
