from copy import deepcopy

import pytest
import torch

from robustdrive.data.preprocessing import read_rgb
from robustdrive.evaluation.runner import EvaluationRun, evaluate
from robustdrive.simulator.interface import NeuralSteering
from robustdrive.training.trainer import load_model, read_checkpoint, train


def test_training_evaluation_resume_and_data_integrity(config):
    output = train(config)
    assert (output / "best.pt").exists()
    run = EvaluationRun(output / "best.pt")
    result = evaluate(run)
    assert result["data_kind"] == "synthetic"
    assert result["n"] == len(run.datasets["test"])
    assert result["mse"] >= 0
    with pytest.raises(FileExistsError):
        train(config)
    # The same RGB frame takes exactly the same deterministic route online and offline.
    policy = NeuralSteering(output / "best.pt", device="cpu")
    rgb, _, _ = run.datasets["test"].raw(0)
    online = policy.predict(rgb)
    model, _ = load_model(output / "best.pt")
    with torch.inference_mode():
        offline = model(run.datasets["test"].preprocess(rgb).unsqueeze(0)).item()
    assert online["prediction"] == pytest.approx(offline, abs=1e-7)
    # Epoch-wise RNG makes a resumed second epoch match uninterrupted training.
    config["training"]["epochs"] = 2
    train(config, output / "latest.pt")
    uninterrupted = deepcopy(config)
    uninterrupted["experiment"]["name"] = "uninterrupted"
    other = train(uninterrupted)
    resumed_state = read_checkpoint(output / "latest.pt")["model"]
    full_state = read_checkpoint(other / "latest.pt")["model"]
    for key in resumed_state:
        torch.testing.assert_close(resumed_state[key], full_state[key], rtol=0, atol=0)
    image_path = run.datasets["test"].samples[0][0].images["center"]
    from PIL import Image

    changed = read_rgb(image_path)
    changed[80:90] = 0
    Image.fromarray(changed).save(image_path)
    with pytest.raises(ValueError, match="Dataset differs"):
        EvaluationRun(output / "best.pt")
