from pathlib import Path

import torch

from src.data.euvp_dataset import EUVPPairedDataset


DATASET_ROOT = Path("data/EUVP/EUVP-Dataset/EUVP")

COLLECTIONS = [
    "underwater_dark",
    "underwater_imagenet",
    "underwater_scenes",
]


def test_euvp_training_dataset_count():
    dataset = EUVPPairedDataset(
        root=str(DATASET_ROOT),
        collections=COLLECTIONS,
        split="train",
    )

    assert len(dataset) == 11435


def test_euvp_validation_dataset_count():
    dataset = EUVPPairedDataset(
        root=str(DATASET_ROOT),
        collections=COLLECTIONS,
        split="validation",
    )

    assert len(dataset) == 1970


def test_euvp_first_training_pair():
    dataset = EUVPPairedDataset(
        root=str(DATASET_ROOT),
        collections=COLLECTIONS,
        split="train",
    )

    sample = dataset[0]

    assert isinstance(sample["input"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)

    assert sample["input"].shape == (3, 256, 256)
    assert sample["target"].shape == (3, 256, 256)

    assert sample["input"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32

    assert sample["input"].min() >= 0.0
    assert sample["input"].max() <= 1.0

    assert sample["target"].min() >= 0.0
    assert sample["target"].max() <= 1.0