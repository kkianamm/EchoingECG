from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.ecg import scale_ecg
from src.datasets.helpers import load_ecg_tar


class MIMIC_ECHOECGDataset(Dataset):
    """
    ECHOs were preprocessed (.avi) then features extracted via echoclip.
    Please see preprocess/echoclip.py for details.
    """

    def __init__(
        self,
        dataset_cfg: dict[str, Any],
        split: str,
        label: str = "lvef_value",
        lvef_threshold: int = 50,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        hdf5_echo_path = dataset_cfg.get("hdf5_echo_path")
        if hdf5_echo_path is None:
            raise ValueError("hdf5_echo_path must be provided in dataset_cfg")
        self.root_dir = Path(hdf5_echo_path)
        self.epsilon = dataset_cfg.get("epsilon")
        if self.epsilon is None:
            self.epsilon = 1e-8  # default small value for numerical stability
        else:
            self.epsilon = float(self.epsilon)
        self.label = label
        self.lvef_threshold = lvef_threshold
        self.hdf5_echo_path = self.root_dir / f"{split}_output.hdf5"
        self.echo_ecg_df = pd.read_csv(self.root_dir / "echo-ecg.csv")
        self.split = split
        sc = joblib.load(dataset_cfg.get("scaler"))
        self._center = torch.from_numpy(sc.mean_.astype(np.float32))  # shape (L,)
        self._scale = torch.from_numpy(sc.scale_.astype(np.float32)).clamp_min(1e-8)  # (L,)
        assert not ((self._center is None) or (self._scale is None)), "recommended scaler"

    def compute_mean_logvar(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and log variance of the given data tensor.
        """
        mean = data.mean(dim=0)
        mean_norm = mean.norm(p=2)

        if mean_norm > 0:
            mean = mean / mean_norm
        else:
            mean = mean  # If norm is zero, leave mean as is to avoid division by zero

        var = data.var(dim=0, unbiased=False)

        logvar = torch.log(var + self.epsilon)  # type: ignore

        return mean, logvar

    def get_echo(self, echo_id: str) -> torch.Tensor:
        with h5py.File(self.hdf5_echo_path, "r") as hf:
            x = hf[f"{echo_id}_leads"]
            x = np.array(x)  # Ensure x is a numpy array
            return torch.from_numpy(x)
        raise ValueError(f"Echo data not found for echo_id {echo_id}")

    def __len__(self) -> int:
        return len(self.echo_ecg_df)

    def __getitem__(self, idx: int) -> dict:
        # ecg pull
        ecg_path = self.echo_ecg_df.loc[idx, "ecg_path"]
        if isinstance(ecg_path, bytes):
            ecg_path = ecg_path.decode("utf-8")
        ecg = torch.from_numpy(load_ecg_tar(str(ecg_path)))  # new_path
        ecg = scale_ecg(self._center, self._scale, ecg)

        # echo-clip embeddings pull
        echo_id = self.echo_ecg_df.loc[idx, "study_id"]
        if isinstance(echo_id, bytes):
            echo_id = echo_id.decode("utf-8")
        echo_id = str(echo_id)
        echo = self.get_echo(echo_id)
        mean, var = self.compute_mean_logvar(echo)
        lvef_value = self.echo_ecg_df.loc[idx, self.label]
        try:
            # Convert numpy/pandas scalar to Python float if needed
            lvef_value_numeric = float(lvef_value.item())  # type: ignore
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert lvef_value '{lvef_value}' to float for comparison.")
        label = torch.tensor(1 if lvef_value_numeric > self.lvef_threshold else 0).long()
        return {"ecg": ecg, "echo_mean": mean, "echo_logvar": var, "label": label}
