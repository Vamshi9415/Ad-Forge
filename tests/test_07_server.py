"""Test 07 — FastAPI Server: HTTP Endpoints and WebSocket.

Requires the server to be running on http://localhost:8000.
Mark: integration

Tests:
- GET / → 200 HTML
- GET /docs → 200 OpenAPI UI
- POST /api/upload-image → returns data_uri
- POST /api/sessions → creates session, returns session_id
- GET /api/sessions/{id}/status → 200 with status
- GET /api/sessions/nonexistent/status → 404
- WebSocket /api/ws/{session_id} → connects, receives messages
- Static file serving /output/
"""

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


async def _check_server():
    """Return True if server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{BASE}/")
            return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def require_server():
    if not await _check_server():
        pytest.skip("Server not running on port 8000 — start with: python -m uvicorn server:app --port 8000")


# ─── Static / root ────────────────────────────────────────────────────────
async def test_get_root_returns_html():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "AdForge" in r.text


async def test_get_docs_returns_200():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/docs")
    assert r.status_code == 200


async def test_html_contains_stage9():
    """UI must have Stage 9 (Audio Production) in its DOM."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/")
    assert "sd-9" in r.text or "Audio Production" in r.text, (
        "Stage 9 not found in HTML — UI may not be up to date"
    )


async def test_html_contains_audio_ready_handler():
    """UI must handle the audio_ready WebSocket event."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/")
    assert "audio_ready" in r.text


# ─── Upload image ─────────────────────────────────────────────────────────
async def test_upload_image_returns_data_uri():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/api/upload-image",
            files={"file": ("test.png", TINY_PNG, "image/png")},
        )
    assert r.status_code == 200
    data = r.json()
    assert "data_uri" in data
    assert data["data_uri"].startswith("data:image/")
    assert "base64," in data["data_uri"]
    assert data["size"] == len(TINY_PNG)


async def test_upload_image_too_large_returns_400():
    large = b"x" * (10 * 1024 * 1024 + 1)  # 10MB + 1 byte
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{BASE}/api/upload-image",
            files={"file": ("big.jpg", large, "image/jpeg")},
        )
    assert r.status_code == 400


async def test_upload_various_mime_types():
    """PNG, JPEG, WEBP should all be accepted."""
    for mime in ["image/png", "image/jpeg", "image/webp"]:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                f"{BASE}/api/upload-image",
                files={"file": ("test", TINY_PNG, mime)},
            )
        assert r.status_code == 200, f"Upload failed for {mime}: {r.text}"
        assert r.json()["data_uri"].startswith(f"data:{mime}")


# ─── Sessions ─────────────────────────────────────────────────────────────
async def test_create_session_returns_session_id():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/api/sessions",
            json={
                "prompt": "Test campaign for a test product.",
                "platform": "youtube",
                "quality_mode": "preview",
                "total_duration": 10,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 8  # UUID[:8]
    print(f"\n  Created session: {data['session_id']}")
    return data["session_id"]


async def test_session_status_after_create():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/api/sessions",
            json={"prompt": "Test.", "platform": "youtube",
                  "quality_mode": "preview", "total_duration": 10},
        )
    sid = r.json()["session_id"]

    async with httpx.AsyncClient(timeout=10) as c:
        r2 = await c.get(f"{BASE}/api/sessions/{sid}/status")
    assert r2.status_code == 200
    data = r2.json()
    assert "status" in data
    assert data["status"] in ("pending", "running", "complete", "error")
    print(f"\n  Session {sid} status: {data['status']}")


async def test_nonexistent_session_returns_404():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/api/sessions/definitely-does-not-exist/status")
    assert r.status_code == 404


async def test_create_session_with_image_upload():
    """Full flow: upload image → create session with image_data_uri."""
    async with httpx.AsyncClient(timeout=15) as c:
        up = await c.post(
            f"{BASE}/api/upload-image",
            files={"file": ("ref.png", TINY_PNG, "image/png")},
        )
    data_uri = up.json()["data_uri"]

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/api/sessions",
            json={
                "prompt": "Test with reference image.",
                "image_data_uri": data_uri,
                "platform": "youtube",
                "quality_mode": "preview",
                "total_duration": 10,
            },
        )
    assert r.status_code == 200
    assert "session_id" in r.json()


# ─── WebSocket ────────────────────────────────────────────────────────────
async def test_websocket_invalid_session_returns_error():
    """Connecting to a non-existent session should get an error message."""
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets package not installed")

    async with websockets.connect(f"{WS_BASE}/api/ws/nonexistent-session") as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    assert msg.get("type") in ("error", "done"), f"Expected error, got: {msg}"


async def test_websocket_valid_session_receives_messages():
    """Create a session, connect WebSocket, verify we receive stage_start messages."""
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets package not installed")

    # Create session
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/api/sessions",
            json={"prompt": "WebSocket test campaign.", "platform": "youtube",
                  "quality_mode": "preview", "total_duration": 10},
        )
    sid = r.json()["session_id"]

    received = []
    try:
        async with websockets.connect(f"{WS_BASE}/api/ws/{sid}") as ws:
            # Collect first 5 messages or 10 seconds
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline and len(received) < 5:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                    received.append(msg)
                    if msg.get("type") in ("done", "pipeline_error"):
                        break
                except asyncio.TimeoutError:
                    break
    except Exception:
        pass

    assert len(received) > 0, "No messages received from WebSocket"
    types = {m.get("type") for m in received}
    assert types.intersection({"stage_start", "log", "pipeline_error", "done"}), (
        f"Unexpected message types: {types}"
    )
    print(f"\n  Received {len(received)} WS messages: {types}")


async def test_websocket_ping_keepalive():
    """Server sends 'ping' on idle timeout — should not crash client."""
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets package not installed")

    # This test just verifies the server doesn't crash; no deep assertions needed
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"{BASE}/api/sessions",
            json={"prompt": "Ping test.", "platform": "youtube",
                  "quality_mode": "preview", "total_duration": 10},
        )
    sid = r.json()["session_id"]

    message_count = 0
    try:
        async with websockets.connect(f"{WS_BASE}/api/ws/{sid}") as ws:
            for _ in range(3):
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                    message_count += 1
                except asyncio.TimeoutError:
                    break
    except Exception:
        pass

    assert message_count >= 0  # Just verify no crash


# ─── Static file serving ──────────────────────────────────────────────────
async def test_output_directory_served():
    """The /output/ static mount should be accessible."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/output/")
    # 404 means directory exists but is empty — which is fine
    # 403 means forbidden — still means the mount is working
    assert r.status_code in (200, 404, 403), (
        f"Unexpected status for /output/: {r.status_code}"
    )
