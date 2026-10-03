from pathlib import Path

import yaml


def load_training_config(path: str | Path) -> dict:
    """
    Load the Paper 2 training configuration.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")

    return config