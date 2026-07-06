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
    input_example = evaluation_lib.InputExample(
        key=int(metadata.get("record_id") or metadata.get("key") or 0),
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


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    if not response:
        return None
    text = _strip_chain_of_thought(response)
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]
    valid_set = {letter.upper() for letter in valid_letters}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_set:
                return letter

    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid_set:
            return letter
    return None


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


def compute_gpqa_reward(response: str, label: Any, metadata: Mapping[str, Any] | None = None) -> float:
    """Rule-based scorer for GPQA-style multiple-choice science QA."""

    if response is None:
        return 0.0

    metadata = metadata or {}
    choices = _choices_from_metadata(metadata)
    raw_letters = metadata.get("valid_letters")
    if raw_letters:
        valid_letters = [str(letter).upper() for letter in raw_letters]
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    else:
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

    extracted_letter = _extract_letter_from_response(response, valid_letters)
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
