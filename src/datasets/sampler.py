import random
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Optional

from torch.utils.data import Dataset, Sampler


class MultiDatasetSampler(Sampler[list[tuple[int, int]]]):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not datasets:
            raise ValueError("`datasets` must be a non-empty sequence.")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last  # reserved for future use / API compatibility
        self._rng = random.Random(seed)

        # Cache lengths to avoid calling len(dataset) repeatedly.
        self._lengths: list[int] = [len(ds) for ds in datasets]  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Return the number of full batches produced across all datasets.

        Since batches are *not* mixed across datasets and partial batches are
        not yielded, the total is the sum of floor(len_i / batch_size).
        """
        bs = self.batch_size
        return sum(n // bs for n in self._lengths)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        # Prepare shuffled deques of indices per dataset for the epoch.
        per_ds_indices: list[deque[int]] = []
        available: list[int] = []  # dataset ids with at least one full batch remaining

        for ds_id, n in enumerate(self._lengths):
            idxs = list(range(n))
            self._rng.shuffle(idxs)
            dq = deque(idxs)
            per_ds_indices.append(dq)
            if len(dq) >= self.batch_size:
                available.append(ds_id)

        # Randomize the initial dataset order for fairness.
        self._rng.shuffle(available)

        # Yield batches until no dataset can supply a full batch.
        bs = self.batch_size
        while available:
            ds_id = self._rng.choice(available)
            dq = per_ds_indices[ds_id]

            # Pop a full batch worth of indices from the left (front).
            batch: list[int] = [dq.popleft() for _ in range(bs)]
            yield [(ds_id, idx) for idx in batch]

            # If fewer than bs remain, this dataset is no longer available.
            if len(dq) < bs:
                # Remove ds_id from `available` (swap-remove for O(1)).
                i = available.index(ds_id)
                available[i] = available[-1]
                available.pop()
