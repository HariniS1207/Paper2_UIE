import torch
from torch.utils.data import DataLoader

from src.data.euvp_dataset import EUVPPairedDataset
from src.models.closed_loop import OneStepClosedLoop
from src.losses.total import TotalLoss
from src.training.train_step import train_step


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = EUVPPairedDataset(
        root="data/EUVP/EUVP-Dataset/EUVP",
        collections=["underwater_dark"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    input_image = batch["input"].float().to(device)
    target = batch["target"].float().to(device)

    model = OneStepClosedLoop().to(device)

    loss_fn = TotalLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    alpha = torch.tensor(
        [0.5],
        device=device,
        dtype=input_image.dtype,
    )

    loss = train_step(
        model,
        optimizer,
        loss_fn,
        input_image,
        target,
        alpha,
    )

    print(f"Device: {device}")
    print(f"Input shape: {tuple(input_image.shape)}")
    print(f"Loss: {loss.item():.6f}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()