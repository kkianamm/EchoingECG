from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from src.datasets.ecg import MIMIC_ECGDataset
from src.datasets.echo import MIMIC_ECHOECGDataset
from src.datasets.helpers import MultiDatasetWrapper, collate_text
from src.datasets.sampler import MultiDatasetSampler

CONFIG_DATASETS = {
    "mimic_ecg": MIMIC_ECGDataset,
    "mimic_echo": MIMIC_ECHOECGDataset,
}


def build_train_valid_loaders(
    all_cfg: dict[str, Any],
    batch_size: int,
    *,
    train_transforms: bool = True,
    valid_transforms: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,  # defaults to True if num_workers > 0
    prefetch_factor: Optional[int] = None,  # defaults to 2 if num_workers > 0
    drop_last: bool = False,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    **kwargs: dict[str, Any],
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Defaults that mirror PyTorch behavior but are explicit:
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 2

    # Create dataset factories (mirrors your original class)
    training_factory = MultiDatasetFactory(
        all_cfg=all_cfg,
        split="train",
        transforms=train_transforms,
        batch_size=batch_size,
    )
    valid_factory = MultiDatasetFactory(
        all_cfg=all_cfg,
        split="valid",
        transforms=valid_transforms,
        batch_size=batch_size,
    )

    # Each factory returns (dataset, batch_sampler)
    train_dataset, train_batch_sampler = training_factory.get_datasets()
    valid_dataset, valid_batch_sampler = valid_factory.get_datasets()

    # Build loaders (no shuffle/batch_size when batch_sampler is provided)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_text,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_sampler=valid_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # usually keep all validation examples
        collate_fn=collate_text,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, valid_loader


class MultiDatasetFactory:
    def __init__(
        self,
        all_cfg: dict[str, Any],
        split: str,
        transforms: bool,
        batch_size: int,
        **kwargs: dict[str, Any],
    ) -> None:
        dataset_cfg = all_cfg.get("dataset_cfg")
        sampler_cfg = all_cfg.get("sampler_cfg")
        assert isinstance(dataset_cfg, dict), "must pass a dictionary for datasets"
        self.batch_size = batch_size
        self.datasets = []

        for i, dataset_key in enumerate(dataset_cfg.keys()):
            print(f"{i}. Instantiating {dataset_key}...")
            CLASS_FN = CONFIG_DATASETS[dataset_key]
            self.datasets.append(
                CLASS_FN(
                    dataset_cfg=dataset_cfg[dataset_key],
                    split=split,
                )
            )
        # Ensure sampler_cfg is a dict
        if sampler_cfg is None:
            sampler_cfg = {}
        self.build_sampler(sampler_cfg)

    def build_sampler(self, sampler_cfg: dict[str, Any]) -> None:
        assert isinstance(sampler_cfg, dict), "must pass valid sampler_cfg"
        self.sampler = MultiDatasetSampler(
            datasets=self.datasets, batch_size=self.batch_size, **sampler_cfg
        )

    def get_datasets(self) -> tuple[Any, Any]:
        return MultiDatasetWrapper(self.datasets), self.sampler
