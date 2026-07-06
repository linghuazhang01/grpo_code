"""Checkpoint discovery and tensor loading for weight-diff analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    storage: str
    shape: tuple[int, ...]


class WeightSource:
    def __init__(self, path: Path, device: str) -> None:
        self.path = path
        self.device = device
        self.locations = discover_locations(path)
        if not self.locations:
            raise ValueError(f"No supported checkpoint tensors found under {path}")

    def tensor_names(self) -> set[str]:
        return set(self.locations)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.locations[name].shape

    def load_tensor(self, name: str) -> Any:
        location = self.locations[name]
        if location.storage == "safetensors":
            safe_open = require_safetensors()
            with safe_open(str(location.path), framework="pt", device=self.device) as handle:
                return handle.get_tensor(name)
        state = torch_load(location.path, self.device)
        try:
            return state[name]
        except KeyError as exc:
            raise KeyError(f"{name} not found in {location.path}") from exc


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for weight-diff analysis. Run this inside the existing MOPD training env."
        ) from exc
    return torch


def require_safetensors() -> Any:
    try:
        from safetensors.torch import safe_open
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "safetensors is required for Hugging Face .safetensors checkpoints."
        ) from exc
    return safe_open


def torch_load(path: Path, device: str) -> Mapping[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        return payload
    raise ValueError(f"Unsupported torch checkpoint payload in {path}")


def discover_locations(path: Path) -> dict[str, TensorLocation]:
    path = path.expanduser()
    if path.is_file():
        return discover_file_locations(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.exists():
            return discover_index_locations(index_path)
    for file_name in ("model.safetensors", "pytorch_model.bin"):
        file_path = path / file_name
        if file_path.exists():
            return discover_file_locations(file_path)
    safetensor_files = sorted(path.glob("*.safetensors"))
    if safetensor_files:
        return merge_location_maps(discover_file_locations(item) for item in safetensor_files)
    torch_files = sorted([*path.glob("*.bin"), *path.glob("*.pt"), *path.glob("*.pth")])
    return merge_location_maps(discover_file_locations(item) for item in torch_files)


def discover_file_locations(path: Path) -> dict[str, TensorLocation]:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".json"):
        return discover_index_locations(path)
    if suffixes.endswith(".safetensors"):
        return discover_safetensor_file(path)
    state = torch_load(path, "cpu")
    return {
        name: TensorLocation(path=path, storage="torch", shape=tuple(tensor.shape))
        for name, tensor in state.items()
        if hasattr(tensor, "shape")
    }


def discover_index_locations(index_path: Path) -> dict[str, TensorLocation]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"{index_path} does not contain a Hugging Face weight_map")
    by_file: dict[Path, list[str]] = {}
    for name, shard in weight_map.items():
        by_file.setdefault(index_path.parent / str(shard), []).append(str(name))
    locations: dict[str, TensorLocation] = {}
    for shard_path, names in sorted(by_file.items()):
        shard_locations = discover_file_locations(shard_path)
        for name in names:
            locations[name] = shard_locations[name]
    return locations


def discover_safetensor_file(path: Path) -> dict[str, TensorLocation]:
    safe_open = require_safetensors()
    locations: dict[str, TensorLocation] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            locations[str(name)] = TensorLocation(
                path=path,
                storage="safetensors",
                shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
            )
    return locations


def merge_location_maps(maps: Iterable[dict[str, TensorLocation]]) -> dict[str, TensorLocation]:
    merged: dict[str, TensorLocation] = {}
    for item in maps:
        overlap = set(merged) & set(item)
        if overlap:
            raise ValueError(f"Duplicate tensor names in checkpoint shards: {sorted(overlap)[:5]}")
        merged.update(item)
    return merged
