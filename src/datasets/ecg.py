from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.datasets.helpers import convert_text, get_ecg, get_text, scale_ecg, tokenize_text


class MIMIC_ECGDataset(Dataset):
    def __init__(
        self,
        dataset_cfg: dict[str, Any],
        split: str,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.rootpath, metadata = self.extract_meta(dataset_cfg)
        split = self.get_split(split)
        metadata = metadata.loc[~(metadata["new_path"] == "contains_nans")].copy(deep=True)
        self.metadata = (
            metadata.loc[metadata["split"] == split].reset_index(drop=True).copy(deep=True)
        )
        self.length = len(self.metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/cluster/projects/mcintoshgroup/Trained_Models/bioBERT",
            padding_side="right",
        )

        self.split = split
        sc = joblib.load(dataset_cfg.get("scaler"))
        self._center = torch.from_numpy(sc.mean_.astype(np.float32))  # shape (L,)
        self._scale = torch.from_numpy(sc.scale_.astype(np.float32)).clamp_min(1e-8)  # (L,)
        assert not ((self._center is None) or (self._scale is None)), "recommended scaler"

    def get_split(self, split: str) -> str:
        if split in ["valid", "validate", "val"]:
            return "validate"
        return split

    @staticmethod
    def extract_meta(dataset_cfg: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
        assert dataset_cfg is not None, "ecg_dataset config missing"

        root_path_value = dataset_cfg.get("root_path")
        if root_path_value is None:
            raise ValueError("Missing 'root_path' in dataset_cfg")
        rootpath = Path(root_path_value)
        csv_path_value = dataset_cfg.get("csv_path")
        if csv_path_value is None:
            raise ValueError("Missing 'csv_path' in dataset_cfg")
        metadata = pd.read_csv(rootpath / csv_path_value)
        return rootpath, metadata

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw_text = get_text(idx, self.metadata, "concatenated_reports").lower()
        ecg: torch.Tensor = get_ecg(idx, self.metadata, "new_path")
        ecg = scale_ecg(self._center, self._scale, ecg)

        text_for_ecg = convert_text(raw_text, method="ECG")
        text_tokens, attention_mask = tokenize_text(text_for_ecg, self.tokenizer)

        return {"ecg": ecg, "text": text_tokens, "attention_mask": attention_mask}


class MUSICECGDataset(Dataset):
    def __init__(
        self,
        dataset_cfg: dict[str, Any],
        split: str,
        **kwargs: dict[str, Any],
    ) -> None:
        # Load the CSV file with patient info
        csv_path_value = dataset_cfg.get("csv_path")
        if csv_path_value is None:
            raise ValueError("Missing 'csv_path' in dataset_cfg")
        df = pd.read_csv(csv_path_value)
        df = df[df["split"] == split].copy(deep=True)
        self.studyid = df["studyid"].to_list()
        self.label = dict(zip(df["studyid"], df[dataset_cfg.get("y_column")]))
        self.hdf5_path = dataset_cfg.get("hdf5_path")
        self.y_column = dataset_cfg.get("y_column")
        self.length = dataset_cfg.get("length")

        # scaler
        sc = joblib.load(dataset_cfg.get("scaler"))
        self._center = torch.from_numpy(sc.mean_.astype(np.float32))  # shape (L,)
        self._scale = torch.from_numpy(sc.scale_.astype(np.float32)).clamp_min(1e-8)  # (L,)
        assert not ((self._center is None) or (self._scale is None)), "recommended scaler"

    def __len__(self) -> int:
        return len(self.studyid)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        patient_id = self.studyid[idx]  # ['studyid']

        with h5py.File(self.hdf5_path, "r") as hf:
            x = hf[f"{patient_id}_leads"]
            x = np.array(x)  # Ensure x is a numpy array

        assert x is not None, f"ECG data not found for patient_id {patient_id}"
        x = x[:, : self.length]  # Shape: (12, 1000)
        ecg = scale_ecg(self._center, self._scale, torch.from_numpy(x))

        # Get the corresponding label from the DataFrame
        y = self.label[patient_id]  # [self.y_column]

        # Convert the leads and label to torch tensors
        finalecg = torch.tensor(ecg, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return {"ecg": finalecg, "label": y}
