"""Compatibility wrapper for the evaluation runtime hook."""

from __future__ import annotations

try:
    from eval.paper_eval import run_paper_eval_from_config
except ModuleNotFoundError as exc:
    if exc.name not in {"eval", "eval.paper_eval"}:
        raise

    def run_paper_eval_from_config(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "paper_eval is not available in the standalone GRPO project. "
            "Disable paper_eval or add the original eval.paper_eval module."
        ) from exc

__all__ = ["run_paper_eval_from_config"]
