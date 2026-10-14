import asyncio
import base64
import contextlib
import json

import numpy as np
import pandas as pd
import pytest
from aiohttp import ClientConnectorError, ClientSession, WSMsgType
from aiohttp.test_utils import TestClient, TestServer, unused_port

from robustdrive.evaluation.autonomy import autonomy_metrics
from robustdrive.simulator.controller import SpeedController
from robustdrive.simulator.interface import DriveSession
from robustdrive.simulator.udacity import (
    create_app,
    decode_telemetry,
    encode_event,
    run_until_stopped,
)
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


def test_higher_target_is_not_limited_to_default_speed():
    controller = SpeedController(target_speed=30)
    assert controller(12) == 0.3
    assert controller(29) == pytest.approx(0.08)
    assert controller(30) == 0


def test_recording_rate_and_frame_limit_preserve_all_telemetry(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("robustdrive.simulator.interface.time.monotonic", lambda: clock[0])
    session = DriveSession(
        TestPolicy(),
        tmp_path / "drive",
        SpeedController(),
        {},
        record_frames=True,
        record_fps=2,
        max_recorded_frames=2,
    )
    try:
        rows = []
        for elapsed in (0, 0.1, 0.5, 1.0, 1.5):
            clock[0] = 100 + elapsed
            rows.append(session.step(np.zeros((80, 120, 3), dtype=np.uint8), 10))
    finally:
        session.close()
    assert [bool(row["image"]) for row in rows] == [True, False, True, False, False]
    assert len(list((session.output / "frames").glob("*.jpg"))) == 2
    assert len(pd.read_csv(session.output / "telemetry.csv")) == 5
    assert json.loads((session.output / "session_end.json").read_text())["frame_limit_reached"]


def test_unannotated_real_drive_cannot_claim_perfect_autonomy(tmp_path):
    pd.DataFrame({"elapsed_seconds": [0, 10], "steering": [0, 0.1]}).to_csv(
        tmp_path / "telemetry.csv", index=False
    )
    pd.DataFrame(columns=["elapsed_seconds", "event"]).to_csv(tmp_path / "events.csv", index=False)
    write_json(
        tmp_path / "drive_metadata.json", {"source": "udacity-live", "training_data_kind": "real"}
    )
    result = autonomy_metrics(tmp_path)
    assert "UNANNOTATED" in result["status"]
    assert result["annotation_count"] == 0
    assert result["time_autonomy_percent"] is None
    assert result["penalty_autonomy_percent"] is None
    assert result["lap_completion_percent"] is None


def test_launcher_stop_file_closes_live_connection(tmp_path):
    session = DriveSession(TestPolicy(), tmp_path / "drive", SpeedController(30), {})
    stop_file = tmp_path / "stop.request"

    async def exchange():
        port = unused_port()
        server = asyncio.create_task(run_until_stopped(session, port, stop_file))
        try:
            async with ClientSession() as client:
                for _ in range(100):
                    try:
                        async with client.get(f"http://127.0.0.1:{port}/") as response:
                            assert (await response.json())["target_speed_mph"] == 30
                        break
                    except ClientConnectorError:
                        await asyncio.sleep(0.01)
                else:
                    pytest.fail("Server did not become ready")
                ws = await client.ws_connect(
                    f"http://127.0.0.1:{port}/socket.io/?EIO=4&transport=websocket"
                )
                await ws.receive_str()
                await ws.receive_str()
                stop_file.touch()
                assert (await ws.receive(timeout=3)).type == WSMsgType.CLOSE
                await asyncio.wait_for(server, timeout=3)
        finally:
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server

    try:
        asyncio.run(exchange())
    finally:
        session.close()


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


def test_lap_started_during_takeover_is_not_unassisted(tmp_path):
    pd.DataFrame({"elapsed_seconds": [0, 10, 20, 30], "steering": [0, 0.1, 0.2, 0.3]}).to_csv(
        tmp_path / "telemetry.csv", index=False
    )
    pd.DataFrame(
        {
            "elapsed_seconds": [2, 5, 8, 25],
            "event": ["takeover", "lap_start", "resume", "lap_complete"],
        }
    ).to_csv(tmp_path / "events.csv", index=False)
    write_json(
        tmp_path / "drive_metadata.json",
        {"source": "synthetic-protocol-test", "training_data_kind": "synthetic"},
    )
    result = autonomy_metrics(tmp_path)
    assert result["lap_completion_percent"] == 100
    assert result["unassisted_lap_completion_percent"] == 0


def test_manual_tail_is_included_in_duration(tmp_path):
    pd.DataFrame({"elapsed_seconds": [0, 10], "steering": [0, 0.1]}).to_csv(
        tmp_path / "telemetry.csv", index=False
    )
    pd.DataFrame({"elapsed_seconds": [10], "event": ["takeover"]}).to_csv(
        tmp_path / "events.csv", index=False
    )
    write_json(
        tmp_path / "drive_metadata.json",
        {"source": "synthetic-protocol-test", "training_data_kind": "synthetic"},
    )
    write_json(tmp_path / "session_end.json", {"elapsed_seconds": 20})
    result = autonomy_metrics(tmp_path)
    assert result["time_autonomy_percent"] == 50
    assert result["lap_completion_percent"] is None
