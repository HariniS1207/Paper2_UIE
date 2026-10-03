import torch

from src.training.loss_utils import create_controlled_outputs


def train_step(
    model,
    optimizer,
    loss_fn,
    input_image,
    target,
    alpha,
):
    """
    Execute one optimization step for the one-step closed loop.
    """

    model.train()
    optimizer.zero_grad()

    outputs = model(input_image, alpha)

    prediction = outputs["enhanced"]

    low_output, high_output = create_controlled_outputs(
        model.enhancer,
        input_image,
    )

    low_redegraded = model.redegrader(low_output)
    high_redegraded = model.redegrader(high_output)

    consequence_a = model.consequence_encoder.encode_map(
        torch.abs(input_image - low_redegraded)
    )

    consequence_b = model.consequence_encoder.encode_map(
        torch.abs(input_image - high_redegraded)
    )

    loss = loss_fn(
        prediction,
        target,
        consequence_a,
        consequence_b,
        input_image,
        low_output,
        high_output,
    )

    loss.backward()
    optimizer.step()

    return loss.detach()