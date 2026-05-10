"""Test 02 — External API Connections: RunwayML and OpenAI.

Verifies:
- RunwayML API is reachable and key is authenticated (expects 404/422, not 401)
- OpenAI chat completion responds correctly
- OpenAI TTS produces a non-empty MP3 file
- OpenAI Vision API accepts an image payload
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.api


# ─── RunwayML ─────────────────────────────────────────────────────────────
async def test_runway_api_reachable():
    """Probe a non-existent task — should return 404/422, not 401 (wrong key)."""
    import httpx

    key = os.environ.get("RUNWAYML_API_SECRET", "")
    assert key, "RUNWAYML_API_SECRET not set"

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            "https://api.dev.runwayml.com/v1/tasks/nonexistent-id-00000000",
            headers={
                "Authorization": f"Bearer {key}",
                "X-Runway-Version": "2024-11-06",
            },
        )

    assert r.status_code != 401, (
        f"RunwayML returned 401 — API key may be invalid or expired. "
        f"Response: {r.text[:200]}"
    )
    assert r.status_code in (404, 422, 400), (
        f"Unexpected status from RunwayML: {r.status_code} — {r.text[:200]}"
    )
    print(f"\n  RunwayML auth check: {r.status_code} (expected 404/422 for non-existent task)")


async def test_runway_models_endpoint():
    """Check that the models/org endpoint is accessible."""
    import httpx

    key = os.environ.get("RUNWAYML_API_SECRET", "")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            "https://api.dev.runwayml.com/v1/tasks",
            headers={
                "Authorization": f"Bearer {key}",
                "X-Runway-Version": "2024-11-06",
            },
        )
    # 200 or 405 (method not allowed) — both indicate connectivity
    assert r.status_code in (200, 405, 404, 422), (
        f"Unexpected status {r.status_code}: {r.text[:200]}"
    )
    assert r.status_code != 401, "RunwayML key rejected (401)"


# ─── OpenAI Chat ──────────────────────────────────────────────────────────
async def test_openai_chat_gpt4o_mini():
    """Cheapest model — just verify connection and response format."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with exactly: PING_OK"}],
        max_tokens=10,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().upper()
    assert "PING" in text or "OK" in text, f"Unexpected reply: {text}"
    assert resp.usage.total_tokens > 0
    print(f"\n  GPT-4o-mini response: '{text}' | tokens: {resp.usage.total_tokens}")


async def test_openai_json_mode():
    """Test JSON response mode — used by all pipeline agents."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": "Return: {\"status\": \"ok\", \"value\": 42}"},
        ],
        response_format={"type": "json_object"},
        max_tokens=50,
        temperature=0,
    )
    import json
    data = json.loads(resp.choices[0].message.content)
    assert "status" in data or "value" in data, f"Bad JSON: {data}"
    print(f"\n  JSON mode response: {data}")


async def test_openai_vision_accepts_image(tiny_data_uri):
    """Test GPT-4o Vision with a tiny base64 image — used by VLM validator."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 3 words."},
                    {"type": "image_url",
                     "image_url": {"url": tiny_data_uri, "detail": "low"}},
                ],
            }
        ],
        max_tokens=30,
    )
    reply = resp.choices[0].message.content.strip()
    assert len(reply) > 0, "Vision API returned empty response"
    print(f"\n  Vision API reply: {reply!r}")


# ─── OpenAI TTS ───────────────────────────────────────────────────────────
@pytest.mark.slow
async def test_openai_tts_produces_audio(tmp_path):
    """Synthesize a short phrase with tts-1-hd and verify output is non-empty MP3."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = await client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input="AdForge. Generative advertising. Powered by AI.",
        response_format="mp3",
    )

    out = tmp_path / "test_tts.mp3"
    out.write_bytes(response.content)

    assert out.exists()
    size = out.stat().st_size
    assert size > 5_000, f"TTS output too small ({size} bytes) — likely empty"
    print(f"\n  TTS output: {size / 1024:.1f} KB MP3")


def test_runwayml_tts_voice_map():
    """Verify VOICE_MAP contains valid RunwayML TTS presets (not OpenAI voices).

    We switched from OpenAI TTS (nova/shimmer/echo/...) to RunwayML TTS
    (Maya/Noah/Leslie) — this test confirms the map uses the correct preset names.
    """
    from pipeline.agents.audio_gen import VOICE_MAP, VALID_VOICES

    runway_presets = {"Maya", "Noah", "Leslie"}
    for role, voice in VOICE_MAP.items():
        assert voice in runway_presets, (
            f"VOICE_MAP[{role!r}]={voice!r} is not a valid RunwayML TTS preset. "
            f"Valid presets: {runway_presets}"
        )
    # All three presets should be reachable
    mapped_voices = set(VOICE_MAP.values())
    assert mapped_voices == runway_presets, (
        f"Not all RunwayML voices are mapped. Mapped: {mapped_voices}"
    )
    assert VALID_VOICES == runway_presets, f"VALID_VOICES mismatch: {VALID_VOICES}"
    print(f"\n  RunwayML TTS voice map: {VOICE_MAP}")
