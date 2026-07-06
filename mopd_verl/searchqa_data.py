"""SearchQA/Search-R1-style data conversion for MOPD training."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SEARCHQA_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_SEARCHQA_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
DEFAULT_SEARCHQA_TEACHER = "search"
DEFAULT_SEARCHQA_DATA_SOURCE = "searchqa"
DEFAULT_SEARCHQA_PREFIX = "searchR1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            records.append(record)
    return records


def load_searchqa_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return _load_jsonl(source)
    if source.suffix == ".parquet":
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"Unsupported SearchQA input suffix {source.suffix!r}; expected .jsonl or .parquet.")


def _loads_if_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _loads_if_json(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_answer_list(value: Any) -> list[str]:
    value = _loads_if_json(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        answers = [str(item).strip() for item in value if str(item).strip()]
        return answers
    return [str(value).strip()] if str(value).strip() else []


def normalize_searchqa_ground_truth(record: Mapping[str, Any]) -> dict[str, list[str]]:
    reward_model = _as_mapping(record.get("reward_model"))
    ground_truth = reward_model.get("ground_truth")
    ground_truth = _loads_if_json(ground_truth)
    if isinstance(ground_truth, Mapping):
        target = _as_answer_list(ground_truth.get("target", ground_truth.get("answers")))
        if target:
            return {"target": target}
    answers = _as_answer_list(ground_truth)
    if not answers:
        for key in ("golden_answers", "answers", "answer", "target"):
            answers = _as_answer_list(record.get(key))
            if answers:
                break
    if not answers:
        raise ValueError("SearchQA row must contain reward_model.ground_truth, golden_answers, answers, answer, or target.")
    return {"target": answers}


def _question_from_record(record: Mapping[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("SearchQA row must contain a non-empty question/query field.")


def _base_data_source(record: Mapping[str, Any], default_data_source: str | None) -> str:
    value = record.get("data_source") or default_data_source or DEFAULT_SEARCHQA_DATA_SOURCE
    value = str(value).strip() or DEFAULT_SEARCHQA_DATA_SOURCE
    return value


def _tag_data_source(base_data_source: str, prefix: str) -> str:
    if base_data_source.startswith(f"{prefix}_"):
        return base_data_source
    return f"{prefix}_{base_data_source}"


def _sample_id(data_source: str, raw_id: Any, row_position: int, teacher: str) -> str:
    clean_source = str(data_source).replace("/", "_")
    clean_id = str(raw_id if raw_id is not None else row_position).replace("/", "_")
    return f"{teacher}:{clean_source}:{clean_id}"


def searchqa_records_to_verl_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    default_data_source: str | None = None,
    teacher: str = DEFAULT_SEARCHQA_TEACHER,
    system_content: str = DEFAULT_SEARCHQA_SYSTEM_CONTENT,
    user_content_prefix: str = DEFAULT_SEARCHQA_USER_CONTENT_PREFIX,
    data_source_prefix: str = DEFAULT_SEARCHQA_PREFIX,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(records):
        question = _question_from_record(record)
        ground_truth = normalize_searchqa_ground_truth(record)
        base_source = _base_data_source(record, default_data_source)
        tagged_source = _tag_data_source(base_source, data_source_prefix)
        raw_id = record.get("id", record.get("qid", record.get("question_id", row_position)))
        reward_model = _as_mapping(record.get("reward_model"))
        reward_model = {
            "style": str(reward_model.get("style", "rule")),
            "ground_truth": ground_truth,
        }
        tools_kwargs = {
            "search": {
                "create_kwargs": {
                    "ground_truth": ground_truth,
                    "question": question,
                    "data_source": tagged_source,
                }
            }
        }
        rows.append(
            {
                "id": f"{tagged_source}:{raw_id}",
                "data_source": tagged_source,
                "prompt": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content_prefix.rstrip("\n") + question},
                ],
                "ability": str(record.get("ability", "searchqa")),
                "reward_model": reward_model,
                "extra_info": {
                    "index": row_position,
                    "split": split,
                    "sample_id": _sample_id(tagged_source, raw_id, row_position, teacher),
                    "opd_teacher": teacher,
                    "domain": teacher,
                    "source_domain": teacher,
                    "validation_dataset": tagged_source,
                    "need_tools_kwargs": True,
                    "question": question,
                    "tools_kwargs": tools_kwargs,
                },
                "metadata": record.get("metadata"),
            }
        )
    return rows


def searchqa_to_verl_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    data_source: str | None = None,
    teacher: str = DEFAULT_SEARCHQA_TEACHER,
    system_content: str = DEFAULT_SEARCHQA_SYSTEM_CONTENT,
    user_content_prefix: str = DEFAULT_SEARCHQA_USER_CONTENT_PREFIX,
    data_source_prefix: str = DEFAULT_SEARCHQA_PREFIX,
) -> int:
    rows = searchqa_records_to_verl_rows(
        load_searchqa_records(input_path),
        split=split,
        default_data_source=data_source,
        teacher=teacher,
        system_content=system_content,
        user_content_prefix=user_content_prefix,
        data_source_prefix=data_source_prefix,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)
