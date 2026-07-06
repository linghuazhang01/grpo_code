"""Prepare General-Reasoner/WebInstruct data for verl-style OPD training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

GENERAL_REASONER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
DEFAULT_DATASET_NAME = "TIGER-Lab/WebInstruct-verified"
DEFAULT_OUTPUT_DIR = Path("data/GeneralReasoner/WebInstructVerified")


def _read_records(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, dict):
            for key in ("data", "records", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value]
        raise ValueError(f"Unsupported JSON structure in {input_path}")
    if suffix == ".parquet":
        return pd.read_parquet(input_path).to_dict(orient="records")
    raise ValueError(f"Unsupported input file type: {input_path}")


def _nested_get(record: Mapping[str, Any], key: str) -> Any:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, Mapping):
        return extra_info.get(key)
    return None


def _question_from_record(record: Mapping[str, Any]) -> str:
    for key in ("question", "problem", "query"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested_question = _nested_get(record, "question")
    if isinstance(nested_question, str) and nested_question.strip():
        return nested_question.strip()

    prompt = record.get("prompt")
    if isinstance(prompt, list):
        for message in prompt:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    raise ValueError("General-Reasoner record is missing question text")


def _answer_from_record(record: Mapping[str, Any]) -> str:
    reward_model = record.get("reward_model")
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None:
            return str(ground_truth)

    for key in ("answer", "solution", "target", "ground_truth"):
        value = record.get(key)
        if value is not None:
            return str(value)

    nested_answer = _nested_get(record, "answer")
    if nested_answer is not None:
        return str(nested_answer)
    raise ValueError("General-Reasoner record is missing answer text")


def _prompt_from_record(record: Mapping[str, Any], question: str) -> list[dict[str, str]]:
    prompt = record.get("prompt")
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str):
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    content = f"{question.rstrip()} {GENERAL_REASONER_INSTRUCTION}"
    return [{"role": "user", "content": content}]


def _sample_id(record: Mapping[str, Any], index: int) -> str:
    for key in ("id", "uid", "uuid", "sample_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    nested_id = _nested_get(record, "sample_id")
    if nested_id is not None:
        return str(nested_id)
    return str(index)


def general_reasoner_record_to_verl(
    record: Mapping[str, Any],
    *,
    index: int,
    split: str,
) -> dict[str, Any]:
    """Normalize one General-Reasoner/WebInstruct record to verl parquet schema."""

    question = _question_from_record(record)
    answer = _answer_from_record(record)
    sample_id = _sample_id(record, index)
    extra_info = dict(record.get("extra_info") or {})
    level = record.get("difficulty") or record.get("level") or extra_info.get("level")

    extra_info.update(
        {
            "index": index,
            "split": split,
            "sample_id": f"reasoning:general-reasoner:{sample_id}",
            "opd_teacher": "reasoning",
            "domain": "reasoning",
            "source_domain": "reasoning",
            "validation_dataset": "general-reasoner",
            "question": question,
            "answer": answer,
        }
    )
    if level is not None:
        extra_info["level"] = str(level)

    row: dict[str, Any] = {
        "id": f"general-reasoner:{sample_id}",
        "data_source": "general-reasoner",
        "prompt": _prompt_from_record(record, question),
        "ability": "reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }
    if "metadata" in record:
        row["metadata"] = record["metadata"]
    return row


def general_reasoner_records_to_dataframe(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    selected_records = list(records)
    if max_samples is not None and max_samples >= 0:
        selected_records = selected_records[:max_samples]
    rows = [
        general_reasoner_record_to_verl(record, index=index, split=split)
        for index, record in enumerate(selected_records)
    ]
    return pd.DataFrame(rows)


def general_reasoner_to_verl_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    max_samples: int | None = None,
) -> int:
    records = _read_records(Path(input_path))
    dataframe = general_reasoner_records_to_dataframe(
        records,
        split=split,
        max_samples=max_samples,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output, index=False)
    return len(dataframe)


def prepare_general_reasoner_hf_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    test_max_samples: int | None = 100,
) -> dict[str, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "prepare-general-reasoner-hf requires the `datasets` package."
        ) from exc

    dataset = load_dataset(dataset_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name in ("train", "test", "validation"):
        if split_name not in dataset:
            continue
        max_samples = test_max_samples if split_name in {"test", "validation"} else None
        dataframe = general_reasoner_records_to_dataframe(
            dataset[split_name],
            split=split_name,
            max_samples=max_samples,
        )
        split_output = output / f"{split_name}.parquet"
        dataframe.to_parquet(split_output, index=False)
        counts[split_name] = len(dataframe)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input JSON/JSONL/parquet file.")
    parser.add_argument("--output", type=Path, help="Output parquet file.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-max-samples", type=int, default=100)
    args = parser.parse_args()

    if args.from_hf:
        counts = prepare_general_reasoner_hf_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            test_max_samples=args.test_max_samples,
        )
        for split_name, count in counts.items():
            print(f"Wrote {count} {split_name} examples")
        return

    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --from-hf is set")
    count = general_reasoner_to_verl_parquet(
        args.input,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Wrote {count} {args.split} examples to {args.output}")


if __name__ == "__main__":
    main()
