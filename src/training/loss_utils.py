import torch


def create_controlled_outputs(
    enhancer,
    input_image: torch.Tensor,
):
    """
    Generate conservative and aggressive outputs for
    control-consistency training.
    """

    batch_size = input_image.shape[0]
    device = input_image.device
    dtype = input_image.dtype

    low_alpha = torch.zeros(
        batch_size,
        device=device,
        dtype=dtype,
    )

    high_alpha = torch.ones(
        batch_size,
        device=device,
        dtype=dtype,
    )

    low_output = enhancer(
        input_image,
        low_alpha,
    )

    high_output = enhancer(
        input_image,
        high_alpha,
    )

    return low_output, high_output