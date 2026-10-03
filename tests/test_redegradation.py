import torch

from src.models.redegradation import ReDegradation


def test_output_shape():
    model = ReDegradation()

    enhanced = torch.rand(2, 3, 256, 256)
    degraded = model(enhanced)

    assert degraded.shape == enhanced.shape


def test_output_range():
    model = ReDegradation()

    enhanced = torch.rand(2, 3, 64, 64)
    degraded = model(enhanced)

    assert torch.all(degraded >= 0.0)
    assert torch.all(degraded <= 1.0)


def test_batch_processing():
    model = ReDegradation()

    enhanced = torch.rand(4, 3, 32, 32)
    degraded = model(enhanced)

    assert degraded.shape[0] == 4