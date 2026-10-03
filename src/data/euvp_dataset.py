from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class EUVPPairedDataset(Dataset):
    """
    Paired EUVP dataset loader.

    Each sample consists of:
        input_image  -> trainA
        target_image -> trainB

    The original EUVP directory structure is preserved.
    """

    def __init__(
        self,
        root: str,
        collections: List[str],
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root)
        self.collections = collections
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[Path, Path]] = []

        for collection in collections:
            collection_root = self.root / "Paired" / collection

            if split == "train":
                input_dir = collection_root / "trainA"
                target_dir = collection_root / "trainB"
            elif split == "validation":
                input_dir = collection_root / "validation"
                target_dir = None
            else:
                raise ValueError(
                    f"Unsupported split: {split}. "
                    "Use 'train' or 'validation'."
                )

            if not input_dir.exists():
                raise FileNotFoundError(
                    f"Input directory not found: {input_dir}"
                )

            if split == "train":
                if not target_dir.exists():
                    raise FileNotFoundError(
                        f"Target directory not found: {target_dir}"
                    )

                input_files = self._get_images(input_dir)
                target_files = self._get_images(target_dir)

                target_map = {
                    path.stem: path
                    for path in target_files
                }

                for input_path in input_files:
                    target_path = target_map.get(input_path.stem)

                    if target_path is None:
                        raise RuntimeError(
                            f"No matching target found for "
                            f"{input_path.name} in {target_dir}"
                        )

                    self.samples.append(
                        (input_path, target_path)
                    )

            else:
                input_files = self._get_images(input_dir)

                for input_path in input_files:
                    self.samples.append(
                        (input_path, input_path)
                    )

    @staticmethod
    def _get_images(directory: Path) -> List[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        input_path, target_path = self.samples[index]

        input_image = self._load_image(input_path)
        target_image = self._load_image(target_path)

        if self.transform is not None:
            input_image, target_image = self.transform(
                input_image,
                target_image,
            )

        return {
            "input": input_image,
            "target": target_image,
            "input_path": str(input_path),
            "target_path": str(target_path),
        }