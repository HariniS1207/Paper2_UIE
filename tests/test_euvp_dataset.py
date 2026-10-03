from pathlib import Path

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

    assert sample["input"].mode == "RGB"
    assert sample["target"].mode == "RGB"
    assert sample["input"].size == (256, 256)
    assert sample["target"].size == (256, 256)