"""Compatibility exports for relocated thinking-eval utilities."""

from eval.common import *  # noqa: F401,F403
from eval.domains.math import extract_boxed_answer, extract_final_answer, normalize_answer, simple_score_math_answer
from eval.domains.scoring import score_completion, score_with_project_reward

score_math_answer = simple_score_math_answer
