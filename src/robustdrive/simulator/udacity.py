"""Narrow Udacity Socket.IO-over-WebSocket adapter using maintained aiohttp.

Supports the simulator's text packets, default namespace, client/server ping and
the legacy EIO=3/4 query variants. This is deliberately NOT a general Socket.IO
implementation: polling, binary attachments and arbitrary namespaces are absent.
"""

import asyncio
import base64
import contextlib
import io
import json
import logging
import uuid

import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

from robustdrive.simulator.interface import DriveSession

LOGGER = logging.getLogger(__name__)


def decode_telemetry(data: dict) -> tuple[np.ndarray, float]:
    if not isinstance(data, dict) or "image" not in data or "speed" not in data:
        raise ValueError("Telemetry must contain image and speed.")
    speed = float(data["speed"])
    if not np.isfinite(speed):
        raise ValueError("Invalid simulator speed.")
    encoded = data["image"]
    if not isinstance(encoded, str) or len(encoded) > 6_000_000:
        raise ValueError("Invalid/oversized camera frame.")
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > 4_000_000:
            raise ValueError("Unexpected simulator image dimensions.")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy(), speed


def encode_event(event: str, data: dict) -> str:
    return "42" + json.dumps([event, data], separators=(",", ":"), allow_nan=False)


def create_app(session: DriveSession) -> web.Application:
    app = web.Application(client_max_size=6_000_000)
    connected = False

    async def socket(request: web.Request) -> web.StreamResponse:
        nonlocal connected
        if request.query.get("transport") != "websocket":
            raise web.HTTPBadRequest(text="Udacity adapter supports WebSocket transport only.")
        if connected:
            raise web.HTTPConflict(text="Only one simulator may connect to a drive session.")
        ws = web.WebSocketResponse(max_msg_size=6_000_000)
        await ws.prepare(request)
        connected = True
        sid = uuid.uuid4().hex
        await ws.send_str("0" + json.dumps({"sid": sid, "upgrades": [], "pingInterval": 25000,
                                          "pingTimeout": 60000, "maxPayload": 6_000_000}))
        await ws.send_str("40")

        async def heartbeat():
            while True:
                await asyncio.sleep(25)
                await ws.send_str("2")

        task = asyncio.create_task(heartbeat())
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                packet = message.data
                if packet.startswith("2"):
                    await ws.send_str("3" + packet[1:])
                elif packet.startswith("42"):
                    try:
                        value = json.loads(packet[2:])
                        if not isinstance(value, list) or len(value) != 2:
                            raise ValueError("Invalid event packet")
                        event, data = value
                        if event != "telemetry":
                            continue
                        if not data or session.manual:
                            await ws.send_str(encode_event("manual", {}))
                            continue
                        rgb, speed = decode_telemetry(data)
                        row = await asyncio.to_thread(session.step, rgb, speed)
                        await ws.send_str(encode_event("steer", {"steering_angle": str(row["steering"]),
                                                               "throttle": str(row["throttle"])}))
                    except (ValueError, TypeError, OSError) as error:
                        LOGGER.warning("Invalid telemetry: %s", error)
                        await ws.send_str(encode_event("steer", {"steering_angle": "0", "throttle": "0"}))
                elif packet in {"1", "41"}:
                    break
        finally:
            connected = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await ws.close()
        return ws

    async def mark(request: web.Request) -> web.Response:
        # No cross-origin website can POST JSON to this loopback annotation service.
        if request.headers.get("Origin"):
            raise web.HTTPForbidden(text="Use the local mark_event.py command.")
        try:
            body = await request.json()
            session.event(body["event"], str(body.get("note", "")))
        except (ValueError, KeyError, TypeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"recorded": body["event"]})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"adapter": "udacity-websocket", "connected": connected,
                                  "frames": session.frame, "manual": session.manual})

    app.router.add_get("/socket.io/", socket)
    app.router.add_get("/", health)
    app.router.add_post("/event", mark)
    return app
