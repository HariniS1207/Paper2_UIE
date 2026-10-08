"""Dataset helpers for image-disjoint consequence-control experiments."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedManifestDataset(Dataset):
    def __init__(self, rows, dataset_root):
        self.rows = list(rows)
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset_root / row["input_path"]) as image:
            x = np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        with Image.open(self.dataset_root / row["target_path"]) as image:
            target = np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        if x.shape != target.shape:
            raise ValueError(f"Paired dimensions differ for {row['image_id']}")
        x_tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target.copy()).permute(2, 0, 1)
        return x_tensor, target_tensor


class OracleStateDataset(Dataset):
    def __init__(self, rows, dataset_root):
        self.rows = list(rows)
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset_root / row["input_path"]) as image:
            x = np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        x_tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)
        return (
            x_tensor,
            torch.tensor(float(row["alpha"]), dtype=torch.float32),
            torch.tensor(float(row["oracle_delta"]), dtype=torch.float32),
        )
