from src.training.config import load_training_config


def test_training_config():
    config = load_training_config("configs/training.yaml")

    assert config["training"]["epochs"] == 20
    assert config["training"]["batch_size"] == 4
    assert config["training"]["learning_rate"] == 0.0001

    assert config["loss"]["reconstruction_weight"] == 1.0
    assert config["loss"]["consequence_weight"] == 0.1
    assert config["loss"]["control_weight"] == 0.1

    assert config["runtime"]["device"] == "cuda"