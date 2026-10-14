import torch
from torch import nn


@torch.inference_mode()
def mc_predict(model: nn.Module, images: torch.Tensor, passes: int = 20) -> dict[str, torch.Tensor]:
    """Enable dropout only, preserving all prior module states even on failure.

    Variance uses the sample estimate (Bessel correction). This is a model
    dispersion heuristic, not a calibrated probability of collision.
    """
    if passes < 2:
        raise ValueError("At least two MC passes are required.")
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    if not dropouts:
        raise ValueError("MC dropout requires a model trained with nonzero dropout.")
    states = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for module in dropouts:
            module.train()
        predictions = torch.stack([model(images) for _ in range(passes)])
        variance = predictions.var(dim=0, unbiased=True)
        return {"mean": predictions.mean(dim=0), "std": variance.sqrt(), "variance": variance}
    finally:
        for module, training in states.items():
            module.training = training
