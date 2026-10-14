from torch import nn


def steering_loss() -> nn.Module:
    return nn.MSELoss()
