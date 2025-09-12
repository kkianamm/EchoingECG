import io
import tarfile
from collections.abc import Sequence
from math import gcd
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

MAP = {
    "lvef30": ("lvef_value", 30, "lessthan"),
    "lvef40": ("lvef_value", 40, "lessthan"),
    "lvef50": ("lvef_value", 50, "lessthan"),
}

PADKEYS = ("text", "attention_mask")


class MultiDatasetWrapper(Dataset):
    """Wraps multiple datasets and indexes them via (dataset_id, sample_idx)."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets)  # type: ignore

    def __getitem__(
        self, key: tuple[int, int]
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, list[str], torch.Tensor, list[str]],
        tuple[torch.Tensor, list[str], torch.Tensor, list[str], torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        ds_id, sample_idx = key
        return self.datasets[ds_id][sample_idx]


def key_padding_collate(
    batch: Sequence[Mapping[str, torch.Tensor]],
    *,
    pad_keys: tuple[str, ...] = PADKEYS,
    padding_value: Union[int, float] = 0,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    keys = set(batch[0].keys())  # assume consistent keys for this dataset
    pad_keys_set = set(pad_keys)
    assert pad_keys_set.issubset(keys), "pad_keys must be a subset of batch keys"
    for k in keys:
        values = [sample[k] for sample in batch]

        if k in pad_keys_set:
            out[k] = pad_sequence(values, batch_first=True, padding_value=padding_value)
        else:
            # Stack normally (must have equal shapes)
            if not all(v.shape == values[0].shape for v in values):
                raise ValueError(f"Key {k!r} has mismatched shapes and is not in pad_keys")
            out[k] = torch.stack(values, dim=0)
    return out


def collate_text(batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return key_padding_collate(batch, pad_keys=PADKEYS, padding_value=0)


def scale_ecg(_center: torch.Tensor, _scale: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
    ecg = ecg.to(torch.float32)
    ecg = (ecg - _center[:, None]) / _scale[:, None]
    return ecg


def downsample_ecg(
    x: np.ndarray,
    src_fs: int = 512,
    target_fs: int = 100,
    band: Optional[tuple[float, float]] = (0.5, 40.0),  # Hz, (low, high); set None to skip
    bp_order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Downsample ECG from src_fs to target_fs with optional notch + bandpass.
    - Works for 1D or multi-lead arrays; 'axis' is the time axis.
    - Uses zero-phase IIR for notch/bandpass and resample_poly for rational resampling.
    """
    x = np.asarray(x)

    # Optional bandpass (zero-phase). Clamp highcut to <= 0.45 * target_fs to avoid aliasing.
    if band is not None:
        lowcut, highcut = band
        # ensure stable / alias-safe highcut
        max_high = 0.45 * target_fs
        highcut = min(highcut, max_high)

        nyq = src_fs / 2.0
        if lowcut <= 0:
            wn = highcut / nyq
            sos = signal.butter(bp_order, wn, btype="low", output="sos")
        else:
            wn = (lowcut / nyq, highcut / nyq)
            sos = signal.butter(bp_order, wn, btype="band", output="sos")
        x = signal.sosfiltfilt(sos, x, axis=axis)

    # Resample with rational ratio using polyphase filtering (anti-aliasing FIR included)
    g = gcd(src_fs, target_fs)
    up = target_fs // g
    down = src_fs // g
    # Kaiser window with beta=5 is a good default; pad to reduce edge effects.
    y = signal.resample_poly(x, up, down, axis=axis, window=("kaiser", 5.0), padtype="median")

    return y


def load_ecg_tar(tar_path: str | Path) -> np.ndarray:
    tar_path = Path(tar_path)
    stem = tar_path.stem

    with tarfile.open(tar_path, mode="r:*") as tf:  # r:* handles plain tar or .tar.gz
        assert f"{stem}.npy" in tf.getnames(), f"Expected {stem}.npy to be in tar file"
        npy_file = tf.extractfile(f"{stem}.npy")
        if npy_file is None:
            raise FileNotFoundError(f"{stem}.npy not found in tar file {tar_path}")
        npy_bytes = npy_file.read()

    x = np.load(io.BytesIO(npy_bytes)).astype(np.float32)  # (C,S) float32

    return x  # ,meta


def get_ecg(idx: int, metadata: pd.DataFrame, column_name: str) -> torch.Tensor:
    path = metadata.loc[idx, column_name]
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    return torch.from_numpy(load_ecg_tar(str(path)))  # new_path


def get_text(idx: int, metadata: pd.DataFrame, column_name: str) -> str:
    value = metadata.loc[idx, column_name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)  # concatenated_reports


def convert_text(text: str, method: str) -> str:
    return f"Given the data for {method}. The corresponding {method} report is : {text}\n\n."


def tokenize_text(temp: str, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    output = tokenizer(temp)
    return torch.tensor(output["input_ids"]), torch.tensor(output["attention_mask"])
