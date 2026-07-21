"""Compatibility wrapper for the evaluation runtime hook."""

from __future__ import annotations

try:
    from eval.paper_eval import run_paper_eval_from_config
except ModuleNotFoundError as exc:
    if exc.name not in {"eval", "eval.paper_eval"}:
        raise

    def run_paper_eval_from_config(config: object, **_: object) -> dict[str, float]:
        config_get = getattr(config, "get", None)
        paper_eval = config_get("paper_eval", {}) if callable(config_get) else {}
        paper_eval_get = getattr(paper_eval, "get", None)
        enabled = bool(paper_eval_get("enabled", False)) if callable(paper_eval_get) else False
        if not enabled:
            return {}
        raise RuntimeError(
            "paper_eval is not available in the standalone GRPO project. "
            "Disable paper_eval or add the original eval.paper_eval module."
        )

__all__ = ["run_paper_eval_from_config"]
