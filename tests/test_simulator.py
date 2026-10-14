import asyncio
import base64
import json

import numpy as np
import pandas as pd
import pytest
from aiohttp.test_utils import TestClient, TestServer

from robustdrive.evaluation.autonomy import autonomy_metrics
from robustdrive.simulator.controller import SpeedController
from robustdrive.simulator.interface import DriveSession
from robustdrive.simulator.udacity import create_app, decode_telemetry, encode_event
from robustdrive.utils.logging import write_json


class TestPolicy:
    __test__ = False

    def predict(self, rgb):
        return {"prediction": 0.2, "steering": 0.2, "uncertainty": 0.05}


def test_controller_clamps_and_validates():
    controller = SpeedController()
    assert controller(0) == 0.3
    assert controller(30) == 0
    with pytest.raises(ValueError):
        controller(float("nan"))
    with pytest.raises(ValueError):
        SpeedController(target_speed=-5)


@pytest.mark.parametrize("eio", ["3", "4"])
def test_websocket_roundtrip_and_manual_markers(sample, tmp_path, eio):
    session = DriveSession(
        TestPolicy(),
        tmp_path / "drive",
        SpeedController(),
        {"source": "synthetic-protocol-test", "training_data_kind": "synthetic"},
    )
    image = base64.b64encode((sample.parent / "IMG/center_000000.jpg").read_bytes()).decode()

    async def exchange():
        async with TestClient(TestServer(create_app(session))) as client:
            assert (await client.get("/")).status == 200
            assert (await client.get("/socket.io/?transport=polling")).status == 400
            ws = await client.ws_connect(f"/socket.io/?EIO={eio}&transport=websocket")
            assert (await ws.receive_str()).startswith("0{")
            assert await ws.receive_str() == "40"
            await ws.send_str("2probe")
            assert await ws.receive_str() == "3probe"
            assert (await client.post("/event", json={"event": "lap_start"})).status == 200
            for _ in range(2):
                await ws.send_str(encode_event("telemetry", {"image": image, "speed": "10"}))
                name, control = json.loads((await ws.receive_str())[2:])
                assert name == "steer" and float(control["steering_angle"]) == 0.2
                assert 0 <= float(control["throttle"]) <= 0.3
            await ws.send_str(encode_event("telemetry", {"image": "malformed", "speed": "1"}))
            assert json.loads((await ws.receive_str())[2:])[1]["throttle"] == "0"
            assert (await client.post("/event", json={"event": "takeover"})).status == 200
            assert (await client.post("/event", json={"event": "takeover"})).status == 400
            await ws.send_str(encode_event("telemetry", {"image": image, "speed": "10"}))
            assert json.loads((await ws.receive_str())[2:])[0] == "manual"
            assert (await client.post("/event", json={"event": "resume"})).status == 200
            assert (await client.post("/event", json={"event": "lap_abort"})).status == 200
            assert (await client.post("/event", json={"event": "lap_complete"})).status == 400
            await ws.close()

    try:
        asyncio.run(exchange())
    finally:
        session.close()
    result = autonomy_metrics(session.output)
    assert result["interventions"] == 1
    assert result["lap_completion_percent"] == 0
    assert "SYNTHETIC" in result["status"]


def test_known_autonomy_values(tmp_path):
    pd.DataFrame({"elapsed_seconds": [0, 10, 20, 30], "steering": [0, 0.1, 0.2, 0.1]}).to_csv(
        tmp_path / "telemetry.csv", index=False
    )
    pd.DataFrame(
        {
            "elapsed_seconds": [0, 10, 15, 25, 30],
            "event": ["lap_start", "takeover", "resume", "lap_complete", "lap_start"],
        }
    ).to_csv(tmp_path / "events.csv", index=False)
    write_json(
        tmp_path / "drive_metadata.json",
        {"source": "synthetic-protocol-test", "training_data_kind": "synthetic"},
    )
    result = autonomy_metrics(tmp_path, 6)
    assert result["penalty_autonomy_percent"] == pytest.approx(80)
    assert result["time_autonomy_percent"] == pytest.approx(100 * 25 / 30)
    assert result["lap_completion_percent"] == 50
    assert result["unassisted_lap_completion_percent"] == 0
    assert result["mean_absolute_steering_change"] == pytest.approx(0.1)
    assert result["laps"][0]["lap_seconds"] == 25


def test_invalid_telemetry_fails():
    with pytest.raises(ValueError):
        decode_telemetry({"speed": np.nan, "image": ""})
