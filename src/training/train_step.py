import torch

from src.losses.alpha import IdentityLoss, EndpointLoss

def train_step(
    model,
    optimizer,
    input_image,
    target,
    alpha,
    identity_weight=1.0,
    endpoint_weight=1.0,
):
    """
    Execute one optimization step for the alpha-controlled enhancer.

    Current objective:
        1. E(X, 0) ~= X
        2. E(X, 1) ~= target

    The consequence-aware feedback objective will be added
    after the alpha endpoint behavior is validated.
    """

    model.train()
    enhancer = getattr(model, "enhancer", model)
    optimizer.zero_grad()

    batch_size = input_image.shape[0]

    # ---------------------------------------------------------
    # Endpoint 0: identity
    # ---------------------------------------------------------
    zero_alpha = torch.zeros(
        batch_size,
        device=input_image.device,
        dtype=input_image.dtype,
    )

    identity_output = enhancer(
        input_image,
        zero_alpha,
    )

    # ---------------------------------------------------------
    # Endpoint 1: maximum enhancement
    # ---------------------------------------------------------
    one_alpha = torch.ones(
        batch_size,
        device=input_image.device,
        dtype=input_image.dtype,
    )

    endpoint_output = enhancer(
        input_image,
        one_alpha,
    )

    # ---------------------------------------------------------
    # Losses
    # ---------------------------------------------------------
    identity_loss_fn = IdentityLoss()
    endpoint_loss_fn = EndpointLoss()

    identity_loss = identity_loss_fn(
        identity_output,
        input_image,
    )

    endpoint_loss = endpoint_loss_fn(
        endpoint_output,
        target,
    )

    total_loss = (
        identity_weight * identity_loss
        + endpoint_weight * endpoint_loss
    )

    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.detach(),
        "identity_loss": identity_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
    }
