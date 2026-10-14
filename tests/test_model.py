import pytest
import torch

from robustdrive.models.nvidia_cnn import NvidiaCNN
from robustdrive.models.uncertainty import mc_predict


def test_shape_backward_and_parameter_count():
    model = NvidiaCNN(0.25)
    x = torch.randn(2, 3, 66, 200)
    y = model(x)
    assert y.shape == (2,)
    y.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.parameter_count == sum(p.numel() for p in model.parameters())
    assert model.parameter_count < 1_000_000


def test_dropout_uncertainty_restores_states():
    torch.manual_seed(1)
    model = NvidiaCNN(0.4).eval()
    x = torch.randn(3, 3, 66, 200)
    assert torch.equal(model(x), model(x))
    result = mc_predict(model, x, 20)
    assert torch.all(result["std"] > 0)
    torch.testing.assert_close(result["std"].square(), result["variance"])
    assert all(not m.training for m in model.modules())
    model.train()
    mc_predict(model, x, 2)
    assert all(m.training for m in model.modules())


def test_zero_dropout_is_not_uncertainty():
    with pytest.raises(ValueError, match="nonzero"):
        mc_predict(NvidiaCNN(), torch.zeros(1, 3, 66, 200))
