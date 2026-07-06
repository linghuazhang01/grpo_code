"""Backward-compatible imports for ToolRL GRPO data adapters."""

from grpo.data.toolrl import toolrl_frame_to_verl, toolrl_to_verl_parquet

__all__ = ["toolrl_frame_to_verl", "toolrl_to_verl_parquet"]
