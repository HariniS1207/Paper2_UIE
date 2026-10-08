from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader, Subset

from src.data.euvp_dataset import EUVPPairedDataset
from src.models.closed_loop import OneStepClosedLoop
from src.losses.total import TotalLoss
from src.training.config import load_training_config
from src.training.train_step import train_step


def main():
    config = load_training_config("configs/training.yaml")

    device = torch.device(
        config["runtime"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = EUVPPairedDataset(
        root="data/EUVP/EUVP-Dataset/EUVP",
        collections=["underwater_dark"],
        split="train",
    )

    # Controlled first experiment.
    subset_size = min(100, len(dataset))
    dataset = Subset(dataset, range(subset_size))

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["runtime"]["num_workers"],
    )

    model = OneStepClosedLoop().to(device)

    loss_fn = TotalLoss(
        reconstruction_weight=config["loss"]["reconstruction_weight"],
        consequence_weight=config["loss"]["consequence_weight"],
        control_weight=config["loss"]["control_weight"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_dir = Path("checkpoints")
    metrics_dir = Path("results/metrics")

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    model.train()

    for epoch in range(1, 2):
        running_loss = 0.0

        for batch in loader:
            input_image = batch["input"].float().to(device)
            target = batch["target"].float().to(device)

            alpha_values = torch.tensor(
            config["training"]["alpha"]["values"],
            device=device,
            dtype=input_image.dtype,
            )

            alpha_indices = torch.randint(
            low=0,
            high=len(alpha_values),
            size=(input_image.shape[0],),
            device=device,
            )

            alpha = alpha_values[alpha_indices]

            loss = train_step(
                model,
                optimizer,
                loss_fn,
                input_image,
                target,
                alpha,
            )

            running_loss += loss.item()

        average_loss = running_loss / len(loader)

        history.append(
            {
                "epoch": epoch,
                "loss": average_loss,
            }
        )

        print(
            f"Epoch {epoch}/1 | "
            f"Loss: {average_loss:.6f}"
        )

        checkpoint_path = (
            checkpoint_dir / f"paper2_epoch_{epoch:03d}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss,
            },
            checkpoint_path,
        )

    history_path = metrics_dir / "training_history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "loss"],
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Training history saved: {history_path}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )


if __name__ == "__main__":
    main()