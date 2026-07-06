"""Domain-aware weighted training sampler helpers for MOPD verl runs."""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

DOMAIN_LABEL_KEYS = ("domain", "opd_teacher", "source_domain", "ability")


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def normalize_domain_sampling_weights(raw_weights: Any) -> dict[str, float]:
    if raw_weights is None:
        return {}
    if not hasattr(raw_weights, "items"):
        raise ValueError("data.domain_sampling_weights must be a mapping from domain to positive weight.")

    weights: dict[str, float] = {}
    for domain, value in raw_weights.items():
        numeric = float(value)
        if numeric <= 0:
            raise ValueError(f"Domain sampling weight for {domain!r} must be positive.")
        weights[str(domain)] = numeric
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {domain: weight / total for domain, weight in weights.items()}


def normalize_domain_train_files(raw_files: Any) -> dict[str, list[str]]:
    if raw_files is None:
        return {}
    if not hasattr(raw_files, "items"):
        raise ValueError("data.domain_train_files must be a mapping from domain to file path list.")

    output: dict[str, list[str]] = {}
    for domain, value in raw_files.items():
        files = [value] if isinstance(value, str) else list(value)
        if not files or not all(isinstance(item, str) and item for item in files):
            raise ValueError(f"data.domain_train_files.{domain} must contain at least one file path.")
        output[str(domain)] = [str(item) for item in files]
    return output


def flattened_domain_train_files(raw_files: Any) -> list[str]:
    return [file_path for files in normalize_domain_train_files(raw_files).values() for file_path in files]


def _normalize_path_key(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _domain_file_lookup(raw_files: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for domain, files in normalize_domain_train_files(raw_files).items():
        for file_path in files:
            lookup[str(file_path)] = domain
            lookup[_normalize_path_key(file_path)] = domain
    return lookup


def domain_for_data_file(config: Any, file_path: str) -> str | None:
    raw_files = _cfg_get(config, "domain_train_files", None)
    lookup = _domain_file_lookup(raw_files)
    if not lookup:
        return None
    return lookup.get(str(file_path)) or lookup.get(_normalize_path_key(file_path))


def annotate_hf_dataset_domain(dataframe: Any, domain: str) -> Any:
    row_count = len(dataframe)
    for column in ("domain", "opd_teacher", "source_domain"):
        values = [domain] * row_count
        if column in getattr(dataframe, "column_names", []):
            dataframe = dataframe.remove_columns([column])
        dataframe = dataframe.add_column(column, values)
    return dataframe


def domain_label_from_row(row: Mapping[str, Any]) -> str:
    for key in DOMAIN_LABEL_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)

    extra_info = row.get("extra_info")
    if isinstance(extra_info, Mapping):
        for key in DOMAIN_LABEL_KEYS:
            value = extra_info.get(key)
            if value is not None:
                return str(value)
    return "unknown"


def domain_sample_weights(rows: Iterable[Mapping[str, Any]], raw_weights: Any) -> list[float]:
    target_weights = normalize_domain_sampling_weights(raw_weights)
    if not target_weights:
        return []

    labels = [domain_label_from_row(row) for row in rows]
    return _sample_weights_from_labels(labels, target_weights)


def _sample_weights_from_labels(labels: list[str], target_weights: Mapping[str, float]) -> list[float]:
    counts = Counter(label for label in labels if label in target_weights)
    if not counts:
        raise ValueError(
            "data.domain_sampling_weights is set, but no training samples matched "
            f"configured domains: {sorted(target_weights)}"
        )

    return [
        target_weights[label] / counts[label] if label in counts else 0.0
        for label in labels
    ]


def allocate_domain_batch_counts(
    batch_size: int,
    raw_weights: Any,
    domains: Sequence[str] | None = None,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    target_weights = normalize_domain_sampling_weights(raw_weights)
    if not target_weights:
        return {}

    ordered_domains = list(domains or target_weights.keys())
    unknown_domains = [domain for domain in ordered_domains if domain not in target_weights]
    if unknown_domains:
        raise ValueError(f"Domains missing from data.domain_sampling_weights: {unknown_domains}")
    if batch_size < len(ordered_domains):
        raise ValueError(
            f"Batch size {batch_size} is smaller than the number of configured domains {len(ordered_domains)}."
        )

    exact_counts = {domain: target_weights[domain] * batch_size for domain in ordered_domains}
    counts = {domain: int(math.floor(value)) for domain, value in exact_counts.items()}
    remainder = batch_size - sum(counts.values())
    ranked_domains = sorted(
        ordered_domains,
        key=lambda domain: (exact_counts[domain] - counts[domain], target_weights[domain], domain),
        reverse=True,
    )
    for domain in ranked_domains[:remainder]:
        counts[domain] += 1
    return counts


class DomainBatchSampler:
    """Yield full batches with exact domain counts derived from target weights."""

    def __init__(
        self,
        labels: Sequence[str],
        target_weights: Mapping[str, float],
        batch_size: int,
        *,
        replacement: bool = True,
        seed: int | None = None,
    ) -> None:
        self.labels = [str(label) for label in labels]
        self.target_weights = normalize_domain_sampling_weights(target_weights)
        self.batch_size = int(batch_size)
        self.replacement = bool(replacement)
        self.seed = seed
        self.batch_counts = allocate_domain_batch_counts(
            self.batch_size,
            self.target_weights,
            domains=list(self.target_weights.keys()),
        )
        self.indices_by_domain: dict[str, list[int]] = {
            domain: [idx for idx, label in enumerate(self.labels) if label == domain]
            for domain in self.target_weights
        }

        missing_domains = [
            domain for domain, quota in self.batch_counts.items() if quota > 0 and not self.indices_by_domain[domain]
        ]
        if missing_domains:
            raise ValueError(f"No training samples found for configured domains: {missing_domains}")

        if self.replacement:
            self.length = len(self.labels) // self.batch_size
        else:
            lengths = [
                len(self.indices_by_domain[domain]) // quota
                for domain, quota in self.batch_counts.items()
                if quota > 0
            ]
            self.length = min(lengths) if lengths else 0

        if self.length <= 0:
            raise ValueError("DomainBatchSampler would produce zero batches.")

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[int]]:
        import torch

        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))

        if self.replacement:
            for _ in range(self.length):
                batch = []
                for domain, quota in self.batch_counts.items():
                    if quota <= 0:
                        continue
                    pool = self.indices_by_domain[domain]
                    sampled = torch.randint(len(pool), (quota,), generator=generator).tolist()
                    batch.extend(pool[idx] for idx in sampled)
                yield self._shuffle_batch(batch, generator)
            return

        domain_orders: dict[str, list[int]] = {}
        for domain, pool in self.indices_by_domain.items():
            order = torch.randperm(len(pool), generator=generator).tolist()
            domain_orders[domain] = [pool[idx] for idx in order]
        for batch_idx in range(self.length):
            batch = []
            for domain, quota in self.batch_counts.items():
                if quota <= 0:
                    continue
                start = batch_idx * quota
                end = start + quota
                batch.extend(domain_orders[domain][start:end])
            yield self._shuffle_batch(batch, generator)

    @staticmethod
    def _shuffle_batch(batch: list[int], generator: Any) -> list[int]:
        import torch

        order = torch.randperm(len(batch), generator=generator).tolist()
        return [batch[idx] for idx in order]


def _dataset_rows(dataset: Any) -> Iterable[Mapping[str, Any]]:
    dataframe = getattr(dataset, "dataframe", dataset)
    return (dataframe[idx] for idx in range(len(dataframe)))


def create_domain_weighted_sampler(data_config: Any, dataset: Any) -> Any | None:
    raw_weights = _cfg_get(data_config, "domain_sampling_weights", None)
    target_weights = normalize_domain_sampling_weights(raw_weights)
    if not target_weights:
        return None

    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = [domain_label_from_row(row) for row in _dataset_rows(dataset)]
    weights = _sample_weights_from_labels(labels, target_weights)
    if not weights or sum(weights) <= 0:
        raise ValueError("Domain weighted sampler received no positive sample weights.")

    generator = torch.Generator()
    seed = _cfg_get(data_config, "seed", None)
    if seed is not None:
        generator.manual_seed(int(seed))

    label_counts = Counter(labels)
    logger.info(
        "Using MOPD domain weighted sampler with target_weights=%s and dataset_domain_counts=%s",
        target_weights,
        dict(sorted(label_counts.items())),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def create_domain_batch_sampler(data_config: Any, dataset: Any, batch_size: int) -> DomainBatchSampler | None:
    raw_files = _cfg_get(data_config, "domain_train_files", None)
    domain_files = normalize_domain_train_files(raw_files)
    if not domain_files:
        return None

    target_weights = normalize_domain_sampling_weights(_cfg_get(data_config, "domain_sampling_weights", None))
    if not target_weights:
        target_weights = {domain: 1.0 / len(domain_files) for domain in domain_files}

    missing_weight_domains = [domain for domain in domain_files if domain not in target_weights]
    if missing_weight_domains:
        raise ValueError(f"Domains missing from data.domain_sampling_weights: {missing_weight_domains}")

    labels = [domain_label_from_row(row) for row in _dataset_rows(dataset)]
    replacement = bool(_cfg_get(data_config, "domain_sampling_replacement", True))
    seed = _cfg_get(data_config, "seed", None)
    sampler = DomainBatchSampler(
        labels=labels,
        target_weights={domain: target_weights[domain] for domain in domain_files},
        batch_size=batch_size,
        replacement=replacement,
        seed=None if seed is None else int(seed),
    )
    logger.info(
        "Using MOPD exact domain batch sampler with batch_counts=%s, replacement=%s",
        sampler.batch_counts,
        sampler.replacement,
    )
    return sampler
