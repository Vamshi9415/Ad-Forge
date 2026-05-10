"""Test 05 — VLM Validator: Retry Logic (mock-based, zero API cost).

Tests the retry loop exhaustively using unittest.mock to patch:
  - _vlm_check  → controlled VERIFIED/FAILED sequence
  - generate_image → returns a dummy URL
  - download_bytes → returns tiny JPEG bytes

Verifies:
  - VERIFIED on first try → no regeneration
  - FAILED then VERIFIED on retry → one regeneration
  - FAILED × MAX_RETRIES + 1 → status=FAILED, correct number of regen calls
  - Exception in generate_image → status=FAILED, no infinite loop
  - Exception in _vlm_check → auto-verified
  - run() processes shots sequentially (not in parallel)
"""

import asyncio
import base64
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from pipeline.state import PipelineState, Shot, CameraParams

pytestmark = pytest.mark.unit

# Tiny valid JPEG bytes for mocked downloads
TINY_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0xFF, 0xD9,
])


def _make_shot(shot_id="shot_01", with_image=True) -> Shot:
    s = Shot(shot_id=shot_id, narrative_function="test", duration=5,
             subject="test subject", environment="test env", action="test action")
    s.image_prompt = "A professional advertisement photograph, high quality, cinematic lighting."
    s.motion_prompt = "Slow camera push in."
    if with_image:
        b64 = base64.b64encode(TINY_JPEG).decode()
        s.image_base64   = f"data:image/jpeg;base64,{b64}"
        s.image_url      = "https://example.com/image.jpg"
        s.image_status   = "generated"
    return s


def _make_state() -> PipelineState:
    return PipelineState(
        session_id="vlm_test",
        original_prompt="test",
        total_duration=5,
        ratio="1280:720",
        image_model="gen4_image",
    )


# ─── Test: VERIFIED on first try — no regen ───────────────────────────────
async def test_verified_on_first_try(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()

    vlm_result = {"verdict": "VERIFIED", "confidence": 0.95,
                  "issues": [], "notes": "Clean.", "refined_prompt": ""}

    with patch("pipeline.agents.vlm_validator._vlm_check", new=AsyncMock(return_value=vlm_result)):
        with patch("pipeline.agents.vlm_validator.generate_image") as mock_gen:
            await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    assert shot.image_status == "VERIFIED"
    mock_gen.assert_not_called()      # No regen needed


async def test_verified_emits_correct_event(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()

    vlm_result = {"verdict": "VERIFIED", "confidence": 0.9,
                  "issues": [], "notes": "Good.", "refined_prompt": ""}

    with patch("pipeline.agents.vlm_validator._vlm_check", new=AsyncMock(return_value=vlm_result)):
        await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    validated_events = collector.of_type("image_validated")
    assert len(validated_events) == 1
    assert validated_events[0]["verdict"] == "VERIFIED"
    assert validated_events[0]["confidence"] == 0.9


# ─── Test: FAILED → VERIFIED on retry ────────────────────────────────────
async def test_failed_then_verified_on_retry(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()

    failed_result   = {"verdict": "FAILED",   "confidence": 0.7,
                       "issues": ["Blurry subject"],
                       "notes": "Blurry.", "refined_prompt": "HD sharp clear advertising photo."}
    verified_result = {"verdict": "VERIFIED", "confidence": 0.92,
                       "issues": [], "notes": "Fixed.", "refined_prompt": ""}

    vlm_responses = [failed_result, verified_result]

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(side_effect=vlm_responses)):
        with patch("pipeline.agents.vlm_validator.generate_image",
                   new=AsyncMock(return_value="https://example.com/regen.jpg")) as mock_gen:
            with patch("pipeline.agents.vlm_validator.download_bytes",
                       new=AsyncMock(return_value=TINY_JPEG)):
                await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    assert shot.image_status == "VERIFIED"
    mock_gen.assert_called_once()    # Exactly 1 regen call
    # The regen prompt must contain either the refined text (if ≥30 words)
    # or the CORRECTIONS fallback. Either way, it must not be empty.
    call_kwargs = mock_gen.call_args.kwargs
    used_prompt = call_kwargs.get("prompt_text", "")
    assert len(used_prompt) > 20, f"Regen prompt too short: {used_prompt!r}"
    # At least one of these signals should be present
    assert "HD sharp" in used_prompt or "CORRECTIONS" in used_prompt or "Blurry" in used_prompt


async def test_retry_updates_shot_image(tmp_session_dir, collector):
    """After a successful regen, shot.image_base64 should be updated."""
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()
    original_b64 = shot.image_base64

    failed_result   = {"verdict": "FAILED",   "confidence": 0.6,
                       "issues": ["Wrong color"], "notes": "Off.", "refined_prompt": "Better prompt."}
    verified_result = {"verdict": "VERIFIED", "confidence": 0.88,
                       "issues": [], "notes": "OK.",  "refined_prompt": ""}

    new_jpeg = TINY_JPEG + b"\x00\x01\x02"  # Different bytes

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(side_effect=[failed_result, verified_result])):
        with patch("pipeline.agents.vlm_validator.generate_image",
                   new=AsyncMock(return_value="https://example.com/new.jpg")):
            with patch("pipeline.agents.vlm_validator.download_bytes",
                       new=AsyncMock(return_value=new_jpeg)):
                await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    assert shot.image_base64 != original_b64, "image_base64 should be updated after regen"
    assert shot.image_url == "https://example.com/new.jpg"


# ─── Test: FAILED × MAX_RETRIES+1 → final status FAILED ─────────────────
async def test_max_retries_exhausted(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot, MAX_RETRIES

    shot  = _make_shot()
    state = _make_state()

    failed_result = {"verdict": "FAILED", "confidence": 0.5,
                     "issues": ["Persistent problem"],
                     "notes": "Still bad.", "refined_prompt": "Try harder."}

    vlm_responses = [failed_result] * (MAX_RETRIES + 1)

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(side_effect=vlm_responses)):
        with patch("pipeline.agents.vlm_validator.generate_image",
                   new=AsyncMock(return_value="https://example.com/regen.jpg")) as mock_gen:
            with patch("pipeline.agents.vlm_validator.download_bytes",
                       new=AsyncMock(return_value=TINY_JPEG)):
                await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    assert shot.image_status == "FAILED"
    assert mock_gen.call_count == MAX_RETRIES, (
        f"Expected {MAX_RETRIES} regen calls, got {mock_gen.call_count}"
    )


async def test_credit_cost_increases_per_regen(tmp_session_dir, collector):
    """Each regeneration should add to shot.cost_credits."""
    from pipeline.agents.vlm_validator import _validate_one_shot
    from pipeline.state import CREDIT_COST

    shot  = _make_shot()
    state = _make_state()
    initial_credits = shot.cost_credits  # 0

    failed_result   = {"verdict": "FAILED",   "confidence": 0.6, "issues": ["x"],
                       "notes": ".", "refined_prompt": "better."}
    verified_result = {"verdict": "VERIFIED", "confidence": 0.9, "issues": [],
                       "notes": ".", "refined_prompt": ""}

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(side_effect=[failed_result, verified_result])):
        with patch("pipeline.agents.vlm_validator.generate_image",
                   new=AsyncMock(return_value="https://example.com/r.jpg")):
            with patch("pipeline.agents.vlm_validator.download_bytes",
                       new=AsyncMock(return_value=TINY_JPEG)):
                await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    expected_credits = CREDIT_COST.get("gen4_image", 6)
    assert shot.cost_credits == initial_credits + expected_credits


# ─── Test: Exception in generate_image ───────────────────────────────────
async def test_generate_image_exception_stops_retry(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()

    failed_result = {"verdict": "FAILED", "confidence": 0.5,
                     "issues": ["bad"], "notes": ".", "refined_prompt": "fix."}

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(return_value=failed_result)):
        with patch("pipeline.agents.vlm_validator.generate_image",
                   new=AsyncMock(side_effect=RuntimeError("RunwayML API error"))):
            await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    # Should set FAILED and stop — no infinite loop
    assert shot.image_status == "FAILED"
    error_logs = [m for m in collector.messages
                  if m.get("type") == "log" and m.get("level") == "error"]
    assert len(error_logs) >= 1


# ─── Test: Exception in _vlm_check → auto-verify ─────────────────────────
async def test_vlm_check_exception_auto_verifies(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import _validate_one_shot

    shot  = _make_shot()
    state = _make_state()

    with patch("pipeline.agents.vlm_validator._vlm_check",
               new=AsyncMock(side_effect=Exception("GPT-4o API timeout"))):
        await _validate_one_shot(shot, tmp_session_dir, state, collector.emit)

    assert shot.image_status == "VERIFIED"  # Auto-verify on VLM failure
    warn_logs = [m for m in collector.messages
                 if m.get("type") == "log" and m.get("level") in ("warning", "error")]
    assert len(warn_logs) >= 1


# ─── Test: run() processes shots without images ───────────────────────────
async def test_run_skips_shots_without_images(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import run

    state = _make_state()
    state.storyboard = [
        _make_shot("shot_01", with_image=True),
        _make_shot("shot_02", with_image=False),  # No image — should be skipped
    ]
    # Mark shot_02 as still pending
    state.storyboard[1].image_status = "pending"

    vlm_ok = {"verdict": "VERIFIED", "confidence": 0.9, "issues": [], "notes": "", "refined_prompt": ""}

    with patch("pipeline.agents.vlm_validator._vlm_check", new=AsyncMock(return_value=vlm_ok)):
        await run(state, tmp_session_dir, collector.emit)

    # shot_01 should be VERIFIED, shot_02 untouched
    assert state.storyboard[0].image_status == "VERIFIED"
    assert state.storyboard[1].image_status == "pending"


# ─── Test: run() is sequential — not parallel ────────────────────────────
async def test_run_is_sequential(tmp_session_dir, collector):
    """Shots are processed one at a time (for statement, not gather)."""
    from pipeline.agents.vlm_validator import run
    import inspect

    src = inspect.getsource(run)
    # Must use 'for' loop, not asyncio.gather
    assert "for shot in" in src, "run() must use a for loop over shots"
    assert "asyncio.gather" not in src, "run() must NOT use asyncio.gather (prevents throttle)"


async def test_run_emits_vlm_summary(tmp_session_dir, collector):
    from pipeline.agents.vlm_validator import run

    state = _make_state()
    state.storyboard = [_make_shot("shot_01"), _make_shot("shot_02")]

    vlm_ok = {"verdict": "VERIFIED", "confidence": 0.9, "issues": [], "notes": "", "refined_prompt": ""}

    with patch("pipeline.agents.vlm_validator._vlm_check", new=AsyncMock(return_value=vlm_ok)):
        await run(state, tmp_session_dir, collector.emit)

    summaries = collector.of_type("vlm_summary")
    assert len(summaries) == 1
    assert summaries[0]["verified"] == 2
    assert summaries[0]["total"]    == 2
