import torch


def validate_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Validate and normalize the enhancement control parameter.

    Expected range:
        0 <= alpha <= 1

    Args:
        alpha: Tensor containing enhancement control values.

    Returns:
        The validated alpha tensor.
    """
    if not torch.is_tensor(alpha):
        raise TypeError("alpha must be a torch.Tensor")

    if not torch.is_floating_point(alpha):
        alpha = alpha.float()

    if torch.any(alpha < 0) or torch.any(alpha > 1):
        raise ValueError("alpha must be in the range [0, 1]")

    return alpha


def expand_alpha(alpha: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert alpha into a batch-compatible tensor.

    Accepted:
        scalar tensor:        []
        single-value tensor:  [1]
        batch tensor:         [B]

    Returns:
        Tensor with shape [B, 1].
    """
    alpha = validate_alpha(alpha)

    if alpha.ndim == 0:
        alpha = alpha.reshape(1)

    if alpha.ndim != 1:
        raise ValueError("alpha must be a scalar tensor or 1D tensor")

    if alpha.numel() == 1:
        alpha = alpha.expand(batch_size)
    elif alpha.numel() != batch_size:
        raise ValueError(
            f"alpha contains {alpha.numel()} values, "
            f"but batch size is {batch_size}"
        )

    return alpha.reshape(batch_size, 1)