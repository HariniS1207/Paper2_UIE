import torch


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

    # Use the initial enhancement as the reconstruction prediction.
    prediction = outputs["enhanced"]

    # Reuse the consequence representation for the initial
    # consistency objective. A later training stage can introduce
    # separate consequence paths.
    consequence_a = outputs["consequence_vector"]
    consequence_b = outputs["consequence_vector"].detach()

    # Generate controlled outputs for the control objective.
    low_alpha = torch.zeros_like(alpha)
    high_alpha = torch.ones_like(alpha)

    low_output = model.enhancer(input_image, low_alpha)
    high_output = model.enhancer(input_image, high_alpha)

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