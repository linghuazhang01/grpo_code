"""Create tiny parquet files for MOPD smoke tests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _sample(prompt: str, answer: str, teacher: str, index: int) -> dict[str, Any]:
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "user",
                "content": prompt + ' Think briefly and output the final answer after "####".',
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "smoke",
            "index": index,
            "opd_teacher": teacher,
            "domain": teacher,
            "source_domain": teacher,
            "sample_id": f"{teacher}:smoke:{index}",
            "answer": f"#### {answer}",
            "question": prompt,
        },
    }


def build_smoke_frame() -> pd.DataFrame:
    rows = [
        _sample("What is 1 + 1?", "2", "math", 0),
        _sample("What is 2 + 2?", "4", "code", 1),
    ]
    return pd.DataFrame(rows)


def write_smoke_data(output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = build_smoke_frame()
    train_path = output_path / "train.parquet"
    val_path = output_path / "val.parquet"
    frame.to_parquet(train_path, index=False)
    frame.to_parquet(val_path, index=False)
    return {"train": train_path, "val": val_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory where train.parquet and val.parquet are written.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = write_smoke_data(args.output_dir)
    payload = {name: str(path) for name, path in paths.items()}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
