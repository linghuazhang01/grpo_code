"""Code validation rewards for paper-eval datasets used in MOPD validation."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
from typing import Any


def _extract_code(completion: str) -> str:
    if "```python" in completion:
        return completion.split("```python", 1)[1].split("```", 1)[0]
    if "```" in completion:
        block = completion.split("```", 2)[1]
        if "\n" in block:
            first_line, rest = block.split("\n", 1)
            if first_line.strip().isalpha():
                return rest
        return block
    return completion


def _reliability_guard() -> None:
    try:
        from verl.utils.reward_score.prime_code.testing_util import reliability_guard

        reliability_guard()
    except Exception:
        pass


def _run_assert_case(source: str, result: Any) -> None:
    _reliability_guard()
    namespace: dict[str, Any] = {}
    try:
        signal.alarm(10)
        exec(source, namespace)  # noqa: S102 - benchmark reward execution path.
        signal.alarm(0)
        result.append(True)
    except Exception as exc:
        signal.alarm(0)
        result.append({"error": repr(exc)})


def _assert_score(completion: str, payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    code = _extract_code(completion)
    prompt = str(payload.get("prompt", ""))
    assert_case = str(payload.get("assert_case", "")).strip()
    if not assert_case:
        return 0.0, [{"error": "missing assert_case"}]

    candidates = [code]
    if prompt and not code.lstrip().startswith(prompt.lstrip()[:20]):
        candidates.append(prompt + "\n" + code)

    for source in candidates:
        manager = multiprocessing.Manager()
        result = manager.list()
        process = multiprocessing.Process(target=_run_assert_case, args=(source + "\n" + assert_case, result))
        process.start()
        process.join(timeout=12)
        if process.is_alive():
            process.kill()
            metadata = [{"error": "timeout"}]
        else:
            metadata = list(result) or [{"error": "empty result"}]
        if metadata and metadata[0] is True:
            return 1.0, [{"passed": True}]
    return 0.0, metadata if isinstance(metadata, list) else [{"error": str(metadata)}]


def _input_output_score(completion: str, payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    from verl.utils.reward_score import prime_code

    return prime_code.compute_score(completion, payload, continuous=False)


def compute_score(data_source: str, completion: str, ground_truth: str | dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Compute validation score for EvalPlus/LiveCodeBench-style code datasets."""

    if isinstance(ground_truth, str):
        payload = json.loads(ground_truth)
    else:
        payload = ground_truth

    os.environ.setdefault("PYTHONINTMAXSTRDIGITS", "0")
    if data_source in {"HumanEvalPlus", "MBPPPlus"}:
        return _assert_score(completion, payload)
    if data_source == "LiveCodeBench":
        return _input_output_score(completion, payload)
    return 0.0, [{"error": f"unsupported data_source: {data_source}"}]
