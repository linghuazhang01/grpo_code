"""Checkpoint retention helpers shared with the vendored verl trainer."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")
CHECKPOINT_COMPLETE_MARKER = ".complete"


def checkpoint_retention_limit(
    actor_limit: int | None,
    critic_limit: int | None,
    *,
    use_critic: bool,
) -> int | None:
    """Return the number of complete, resumable checkpoints to retain."""

    limits = [limit for limit in (actor_limit,) if limit is not None and limit > 0]
    if use_critic and critic_limit is not None and critic_limit > 0:
        limits.append(critic_limit)
    return min(limits) if limits else None


def prune_global_checkpoints(checkpoint_root: str | Path, max_to_keep: int | None) -> list[Path]:
    """Delete the oldest complete ``global_step_*`` directories.

    Pruning the global directory, instead of only its actor/critic child, keeps
    dataloader state and model shards under the same retention policy. Scanning
    the filesystem also makes retention work after a process restart.
    """

    if max_to_keep is None:
        return []
    if max_to_keep <= 0:
        raise ValueError("max_to_keep must be positive or null")

    root = Path(checkpoint_root)
    if not root.is_dir():
        return []

    checkpoints: list[tuple[int, Path]] = []
    incomplete_checkpoints: list[Path] = []
    for child in root.iterdir():
        match = GLOBAL_STEP_PATTERN.fullmatch(child.name)
        if not child.is_dir() or match is None:
            continue
        is_marked_complete = (child / CHECKPOINT_COMPLETE_MARKER).is_file()
        is_legacy_complete = (child / "actor").is_dir() and (child / "data.pt").is_file()
        if is_marked_complete or is_legacy_complete:
            checkpoints.append((int(match.group(1)), child))
        else:
            incomplete_checkpoints.append(child)
    checkpoints.sort(key=lambda item: item[0])

    removed: list[Path] = []
    expired_paths = incomplete_checkpoints + [path for _, path in checkpoints[:-max_to_keep]]
    for checkpoint_path in expired_paths:
        shutil.rmtree(checkpoint_path)
        removed.append(checkpoint_path)
    return removed
