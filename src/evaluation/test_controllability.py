import torch

from src.models.controllable_enhancer import ControllableEnhancer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ControllableEnhancer().to(device)
    model.eval()

    image = torch.rand(
        1, 3, 256, 256,
        device=device,
    )

    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    with torch.no_grad():
        for alpha_value in alpha_values:
            alpha = torch.tensor(
                [alpha_value],
                device=device,
                dtype=image.dtype,
            )

            output = model(image, alpha)

            change = torch.mean(
                torch.abs(output - image)
            ).item()

            print(
                f"alpha={alpha_value:.2f} | "
                f"mean_change={change:.6f}"
            )


if __name__ == "__main__":
    main()