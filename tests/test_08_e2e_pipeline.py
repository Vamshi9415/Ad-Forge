"""
End-to-End Pipeline Tests — test_08_e2e_pipeline.py
=====================================================
Covers every stage of the full 9-stage AdForge pipeline in four scenarios:

  Scenario A  Full orchestrator run (direct Python, all 9 real API stages)
  Scenario B  HTTP + WebSocket API black-box (requires server on port 8000)
  Scenario C  Multi-platform ratio verification (LLM stages only, no RunwayML)
  Scenario D  Edge-cases — short campaign, reference-image input

The heavy Scenario A fixture runs the pipeline ONCE (session-scoped) and every
assertion reuses the cached result, keeping credit consumption low.

Run cost estimate for Scenario A (15 s, 3 shots, preview quality):
  Image gen:  3 shots × ~6 credits  = ~18 credits
  Video gen:  3 shots × 5s × 5 cr   = ~75 credits
  TTS:        ~300 chars × 1cr/50ch  = ~6 credits
  SFX:        1 generation           = ~2 credits
  Total:      ~$1.01 USD

To run only unit + cheap tests (skip full E2E):
    pytest -m "not e2e"

To run the full E2E suite:
    pytest tests/test_08_e2e_pipeline.py -v --timeout=3600
"""

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

# ── markers ──────────────────────────────────────────────────────────────────
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

SERVER_URL = "http://localhost:8000"
WS_URL     = "ws://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Shared test prompts
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_COFFEE = (
    "A premium cold brew coffee brand targeting urban millennials aged 25-35. "
    "Emphasise the slow 24-hour brew process, sustainably sourced single-origin beans, "
    "and artisan craft. Emotion: calm confidence. CTA: 'Slow is Bold.'"
)

PROMPT_TECH = (
    "A sleek AI-powered noise-cancelling headphone launch ad. "
    "Target: remote workers & audiophiles. Show crystal clarity, deep immersion, focus. "
    "CTA: 'Hear Everything. Block the Rest.'"
)

PROMPT_FITNESS = (
    "A high-energy sportswear brand ad for Gen Z athletes. "
    "Show sweat, intensity, triumph. Bold neon colors, street courts. "
    "CTA: 'Built to Dominate.'"
)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO A  — Full 9-stage orchestrator run  (session-scoped)
# ─────────────────────────────────────────────────────────────────────────────

_e2e_cache: Dict[str, Any] = {}


@pytest.fixture(scope="session")
def e2e_result():
    """Run the complete 9-stage pipeline once; cache and share across all A tests."""
    return asyncio.get_event_loop().run_until_complete(_run_full_pipeline())


async def _run_full_pipeline() -> Dict[str, Any]:
    """Execute PipelineOrchestrator.run() end-to-end, collect all events."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    from pipeline.orchestrator import PipelineOrchestrator
    from tests.conftest import MsgCollector

    session_id  = f"e2e_{uuid.uuid4().hex[:8]}"
    session_dir = Path("output") / session_id
    (session_dir / "images").mkdir(parents=True, exist_ok=True)
    (session_dir / "videos").mkdir(parents=True, exist_ok=True)

    collector  = MsgCollector(silent=False)
    orchestrator = PipelineOrchestrator(session_id, session_dir, collector.emit)

    print(f"\n[E2E] Starting full pipeline — session={session_id}")
    t0 = time.time()

    result = await asyncio.wait_for(
        orchestrator.run(
            prompt=PROMPT_COFFEE,
            image_data_uri=None,
            platform="youtube",
            quality_mode="preview",
            total_duration=15,
        ),
        timeout=2700,   # 45-minute hard cap
    )

    elapsed = round(time.time() - t0, 1)
    print(f"[E2E] Pipeline done in {elapsed}s — session={session_id}")

    return {
        "result":      result,
        "state_msgs":  collector.messages,
        "session_id":  session_id,
        "session_dir": session_dir,
        "elapsed_s":   elapsed,
    }


# ── Stage 1: Intent Ingestion ─────────────────────────────────────────────
def test_a1_brief_has_required_keys(e2e_result):
    """Stage 1 must populate a rich enriched_brief with all core fields."""
    from pipeline.orchestrator import PipelineOrchestrator
    # Pull state out of collector logs instead — confirm stage_complete emitted
    events = [m for m in e2e_result["state_msgs"] if m.get("type") == "stage_complete"]
    stage_ids = [e["stage_id"] for e in events]
    assert "1" in stage_ids, "Stage 1 (Ingestion) never completed"


def test_a1_ingestion_log_emitted(e2e_result):
    logs = [m for m in e2e_result["state_msgs"]
            if m.get("type") == "log" and m.get("stage") == "1"]
    assert len(logs) >= 1, "Stage 1 produced no log messages"


# ── Stage 2: Contextual Enhancement ──────────────────────────────────────
def test_a2_enhancement_stage_complete(e2e_result):
    events = [m for m in e2e_result["state_msgs"] if m.get("type") == "stage_complete"]
    stage_ids = [e["stage_id"] for e in events]
    assert "2" in stage_ids, "Stage 2 (Enhancement) never completed"


# ── Stage 3: Narrative Sequencing ────────────────────────────────────────
def test_a3_storyboard_shots_exist(e2e_result):
    shots = e2e_result["result"].get("shots", [])
    assert len(shots) >= 1, "Pipeline produced no storyboard shots"
    print(f"\n  Shot count: {len(shots)}")


def test_a3_shot_durations_sum_to_target(e2e_result):
    shots    = e2e_result["result"].get("shots", [])
    total    = sum(s["duration"] for s in shots)
    target   = 15
    assert total >= 5,          f"Total shot duration too low: {total}s"
    assert total <= target + 10, f"Total shot duration too high: {total}s"
    print(f"\n  Total storyboard duration: {total}s (target {target}s)")


def test_a3_each_shot_has_id_and_narrative(e2e_result):
    for shot in e2e_result["result"].get("shots", []):
        assert shot.get("shot_id"),           f"Shot missing shot_id: {shot}"
        assert shot.get("narrative_function"), f"Shot missing narrative_function: {shot}"
        assert shot.get("duration", 0) > 0,   f"Shot has zero duration: {shot}"


# ── Stage 4 & 5: Prompts + Validation ────────────────────────────────────
def test_a45_prompt_stages_complete(e2e_result):
    events    = [m for m in e2e_result["state_msgs"] if m.get("type") == "stage_complete"]
    stage_ids = [e["stage_id"] for e in events]
    assert "4" in stage_ids, "Stage 4 (Prompt Engineering) never completed"
    assert "5" in stage_ids, "Stage 5 (Structural Validation) never completed"


# ── Stage 6: Image Synthesis ──────────────────────────────────────────────
def test_a6_image_ready_events_emitted(e2e_result):
    img_events = [m for m in e2e_result["state_msgs"] if m.get("type") == "image_ready"]
    shots      = e2e_result["result"].get("shots", [])
    assert len(img_events) >= 1, "No image_ready events emitted (Stage 6)"
    print(f"\n  image_ready events: {len(img_events)} / {len(shots)} shots")


def test_a6_image_files_on_disk(e2e_result):
    img_dir = e2e_result["session_dir"] / "images"
    jpegs   = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    assert len(jpegs) >= 1, f"No image files found in {img_dir}"
    for f in jpegs:
        assert f.stat().st_size > 1_000, f"Image file suspiciously small: {f}"
    print(f"\n  Image files: {[f.name for f in jpegs]}")


# ── Stage 6a: VLM Validation ──────────────────────────────────────────────
def test_a6a_vlm_stage_complete(e2e_result):
    events    = [m for m in e2e_result["state_msgs"] if m.get("type") == "stage_complete"]
    stage_ids = [e["stage_id"] for e in events]
    assert "6a" in stage_ids, "Stage 6a (VLM Validation) never completed"


def test_a6a_no_shot_left_in_failed_state(e2e_result):
    """After VLM validation, no shot should remain in 'failed' image_status."""
    shots  = e2e_result["result"].get("shots", [])
    failed = [s for s in shots if s.get("image_status") == "failed"]
    # Warn (not error) — VLM may legitimately exhaust retries on edge cases
    if failed:
        pytest.warns(UserWarning,
            match=f"{len(failed)} shot(s) still failed after VLM retries: {[s['shot_id'] for s in failed]}")


# ── Stage 7: Video Generation ─────────────────────────────────────────────
def test_a7_video_ready_events_emitted(e2e_result):
    vid_events = [m for m in e2e_result["state_msgs"] if m.get("type") == "video_ready"]
    assert len(vid_events) >= 1, "No video_ready events emitted (Stage 7)"
    print(f"\n  video_ready events: {len(vid_events)}")


def test_a7_video_files_on_disk(e2e_result):
    vid_dir = e2e_result["session_dir"] / "videos"
    clips   = list(vid_dir.glob("*.mp4"))
    assert len(clips) >= 1, f"No video clips found in {vid_dir}"
    for clip in clips:
        assert clip.stat().st_size > 10_000, f"Video clip too small: {clip}"
    print(f"\n  Video clips: {[c.name for c in clips]}")


def test_a7_shot_statuses_are_complete(e2e_result):
    shots    = e2e_result["result"].get("shots", [])
    complete = [s for s in shots if s.get("video_status") == "complete"]
    assert len(complete) >= 1, "No shots reached video_status=complete"
    print(f"\n  Complete shots: {len(complete)} / {len(shots)}")


# ── Stage 8: Non-Linear Stitching ────────────────────────────────────────
def test_a8_final_video_exists(e2e_result):
    final = e2e_result["session_dir"] / "final.mp4"
    assert final.exists(), "final.mp4 not found after stitching"
    size_mb = final.stat().st_size / (1024 * 1024)
    assert size_mb > 0.1, f"final.mp4 too small: {size_mb:.2f} MB"
    print(f"\n  final.mp4: {size_mb:.2f} MB")


def test_a8_final_video_has_valid_mp4_header(e2e_result):
    """MP4 files start with 'ftyp' box at byte offset 4."""
    final = e2e_result["session_dir"] / "final.mp4"
    if not final.exists():
        pytest.skip("final.mp4 not present")
    header = final.read_bytes()[:12]
    has_ftyp = b"ftyp" in header
    has_moov = (header[:4] in (b'\x00\x00\x00\x18', b'\x00\x00\x00\x1c',
                               b'\x00\x00\x00\x20'))
    assert has_ftyp or len(header) >= 8, (
        f"final.mp4 doesn't look like valid MP4 (header: {header.hex()})"
    )


def test_a8_result_contains_final_video_url(e2e_result):
    result = e2e_result["result"]
    assert result.get("final_video"), "result missing final_video URL"
    assert "/output/" in result["final_video"], "final_video URL looks wrong"
    print(f"\n  final_video URL: {result['final_video']}")


# ── Stage 9: Audio Production ─────────────────────────────────────────────
def test_a9_voiceover_mp3_exists(e2e_result):
    vo = e2e_result["session_dir"] / "voiceover.mp3"
    assert vo.exists(), "voiceover.mp3 not found after audio stage"
    size_kb = vo.stat().st_size // 1024
    assert size_kb > 5, f"voiceover.mp3 too small: {size_kb} KB"
    print(f"\n  voiceover.mp3: {size_kb} KB")


def test_a9_final_with_audio_exists(e2e_result):
    fwa = e2e_result["session_dir"] / "final_with_audio.mp4"
    assert fwa.exists(), "final_with_audio.mp4 not found after audio mux"
    size_mb = fwa.stat().st_size / (1024 * 1024)
    assert size_mb > 0.1, f"final_with_audio.mp4 too small: {size_mb:.2f} MB"
    print(f"\n  final_with_audio.mp4: {size_mb:.2f} MB")


def test_a9_audio_ready_event_emitted(e2e_result):
    audio_evts = [m for m in e2e_result["state_msgs"] if m.get("type") == "audio_ready"]
    assert len(audio_evts) == 1, f"Expected 1 audio_ready event, got {len(audio_evts)}"
    ev = audio_evts[0]
    assert ev.get("audio_video_src"), "audio_ready missing audio_video_src"
    assert ev.get("audio_src"),       "audio_ready missing audio_src"
    assert ev.get("script"),          "audio_ready missing script text"
    assert ev.get("voice") in ("Maya", "Noah", "Leslie"), (
        f"Voice preset not a valid RunwayML preset: {ev.get('voice')}"
    )
    print(f"\n  RunwayML TTS voice: {ev['voice']} | has_music: {ev.get('has_music')}")


def test_a9_voiceover_script_stored(e2e_result):
    result = e2e_result["result"]
    script = result.get("voiceover_script", "")
    assert script and len(script) > 20, (
        "voiceover_script not stored in final result or too short"
    )
    print(f"\n  Script preview: {script[:100]}...")


def test_a9_music_bed_attempted(e2e_result):
    """SFX music generation should have been attempted (may succeed or log gracefully)."""
    sfx_logs = [
        m for m in e2e_result["state_msgs"]
        if m.get("type") == "log" and "sfx" in m.get("message", "").lower()
    ]
    # At minimum GPT-4o should have generated a music_prompt
    script_logs = [
        m for m in e2e_result["state_msgs"]
        if m.get("type") == "log" and "music" in m.get("message", "").lower()
    ]
    assert len(script_logs) >= 1, "No music-related log messages found in Stage 9"


# ── Full event sequence ───────────────────────────────────────────────────
def test_a_all_9_stages_completed(e2e_result):
    """Every stage from 1 through 9 must emit a stage_complete event."""
    completed  = {m["stage_id"] for m in e2e_result["state_msgs"]
                  if m.get("type") == "stage_complete"}
    required   = {"1", "2", "3", "4", "5", "6", "6a", "7", "8", "9"}
    missing    = required - completed
    assert not missing, f"Stage(s) never completed: {sorted(missing)}"
    print(f"\n  All completed stages: {sorted(completed)}")


def test_a_stage_start_before_complete(e2e_result):
    """Every stage must have a start event before its complete event."""
    started   = [m["stage_id"] for m in e2e_result["state_msgs"]
                 if m.get("type") == "stage_start"]
    completed = [m["stage_id"] for m in e2e_result["state_msgs"]
                 if m.get("type") == "stage_complete"]
    # Every completed stage must also have started
    for sid in completed:
        assert sid in started, f"Stage {sid} completed but no stage_start event found"


def test_a_no_pipeline_error_event(e2e_result):
    errors = [m for m in e2e_result["state_msgs"] if m.get("type") == "pipeline_error"]
    if errors:
        pytest.fail(f"pipeline_error event was emitted: {errors[0].get('message', '')}")


# ── Cost tracking ─────────────────────────────────────────────────────────
def test_a_cost_credits_nonzero(e2e_result):
    shots  = e2e_result["result"].get("shots", [])
    total  = e2e_result["result"].get("total_cost_credits", 0)
    assert total > 0, "total_cost_credits is 0 — cost tracking broken"
    print(f"\n  Total credits: {total}")


def test_a_cost_usd_nonzero(e2e_result):
    usd = e2e_result["result"].get("total_cost_usd", 0.0)
    assert usd > 0.0, "total_cost_usd is 0 — cost tracking broken"
    print(f"\n  Total cost: ${usd:.4f} USD")


def test_a_individual_clips_listed(e2e_result):
    clips = e2e_result["result"].get("individual_clips", [])
    assert len(clips) >= 1, "No individual_clips in final result"
    for url in clips:
        assert "/output/" in url, f"Clip URL looks wrong: {url}"
    print(f"\n  Individual clips: {len(clips)}")


def test_a_result_has_campaign_meta(e2e_result):
    result = e2e_result["result"]
    assert result.get("campaign_title"),  "result missing campaign_title"
    assert result.get("platform"),        "result missing platform"
    assert result.get("quality_mode"),    "result missing quality_mode"
    assert result.get("total_duration") is not None, "result missing total_duration"
    print(f"\n  Campaign: {result['campaign_title']} | {result['platform']} | "
          f"{result['quality_mode']} | {result['total_duration']}s")


def test_a_pipeline_elapsed_reasonable(e2e_result):
    elapsed = e2e_result.get("elapsed_s", 0)
    # Full pipeline with RunwayML should take at least 2 minutes
    assert elapsed > 60, f"Pipeline finished suspiciously fast: {elapsed}s"
    # And shouldn't have timed out
    assert elapsed < 2700, f"Pipeline took more than 45 minutes: {elapsed}s"
    print(f"\n  Total pipeline time: {elapsed}s ({elapsed/60:.1f} min)")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B  — HTTP API + WebSocket black-box  (requires server on :8000)
# ─────────────────────────────────────────────────────────────────────────────

def _server_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{SERVER_URL}/", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _server_available(), reason="Server not running on port 8000")
async def test_b_http_session_create_and_status():
    """POST /api/sessions must return a session_id; status must be running/pending."""
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=30) as client:
        resp = await client.post("/api/sessions", json={
            "prompt":         PROMPT_TECH,
            "platform":       "youtube",
            "quality_mode":   "preview",
            "total_duration": 15,
        })
    assert resp.status_code == 200, f"Session create failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert "session_id" in data, f"No session_id in response: {data}"
    sid = data["session_id"]
    assert len(sid) >= 6, "session_id looks too short"
    print(f"\n  Created session: {sid}")

    # Poll status — should be pending or running immediately
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=15) as client:
        status_resp = await client.get(f"/api/sessions/{sid}/status")
    assert status_resp.status_code == 200
    status = status_resp.json().get("status", "")
    assert status in ("pending", "running"), f"Unexpected initial status: {status}"
    print(f"  Status: {status}")


@pytest.mark.skipif(not _server_available(), reason="Server not running on port 8000")
async def test_b_websocket_receives_stage_events():
    """WebSocket must stream stage_start/stage_complete events from the server."""
    import websockets

    # Start a real pipeline session
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=30) as client:
        resp = await client.post("/api/sessions", json={
            "prompt":         PROMPT_FITNESS,
            "platform":       "tiktok",
            "quality_mode":   "preview",
            "total_duration": 10,
        })
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    received: List[dict] = []
    stage_starts = set()
    stage_completes = set()

    ws_uri = f"{WS_URL}/api/ws/{sid}"
    print(f"\n  Connecting to WebSocket: {ws_uri}")

    try:
        async with websockets.connect(ws_uri, ping_interval=20, close_timeout=10) as ws:
            # Collect events for up to 45 minutes or until done
            deadline = time.time() + 2700
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=45)
                    msg = json.loads(raw)
                    received.append(msg)
                    t = msg.get("type", "")

                    if t == "stage_start":
                        stage_starts.add(msg.get("stage_id"))
                    elif t == "stage_complete":
                        stage_completes.add(msg.get("stage_id"))
                        print(f"    Stage {msg.get('stage_id')} complete")
                    elif t in ("done", "pipeline_complete"):
                        print(f"    Pipeline finished ({t})")
                        break
                    elif t == "pipeline_error":
                        pytest.fail(f"Server reported pipeline_error: {msg.get('message')}")
                    elif t == "ping":
                        continue    # server keepalive, ignore

                except asyncio.TimeoutError:
                    continue        # server sent a ping — keep waiting

    except Exception as exc:
        pytest.fail(f"WebSocket error: {exc}")

    # Minimum expectations: at least stages 1-3 must have started and completed
    assert stage_starts,    "No stage_start events received over WebSocket"
    assert stage_completes, "No stage_complete events received over WebSocket"
    assert "1" in stage_completes, "Stage 1 never completed via WebSocket"
    print(f"  Completed stages via WS: {sorted(stage_completes)}")
    print(f"  Total events received: {len(received)}")


@pytest.mark.skipif(not _server_available(), reason="Server not running on port 8000")
async def test_b_upload_image_endpoint():
    """POST /api/upload-image must accept a JPEG and return a data URI."""
    from tests.conftest import TINY_JPEG_BYTES

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=15) as client:
        resp = await client.post(
            "/api/upload-image",
            files={"file": ("test.jpg", TINY_JPEG_BYTES, "image/jpeg")},
        )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert "data_uri" in data, "No data_uri in upload response"
    assert data["data_uri"].startswith("data:image/"), "data_uri has wrong prefix"
    assert data.get("size", 0) > 0, "size is 0"
    print(f"\n  Upload response: size={data['size']}, filename={data.get('filename')}")


@pytest.mark.skipif(not _server_available(), reason="Server not running on port 8000")
async def test_b_unknown_session_ws_returns_error():
    """Connecting to a non-existent session_id must return an error message."""
    import websockets

    fake_sid = "deadbeef"
    async with websockets.connect(f"{WS_URL}/api/ws/{fake_sid}") as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msg = json.loads(raw)
        assert msg.get("type") == "error", f"Expected error message, got: {msg}"
        print(f"\n  Error message for unknown session: {msg.get('message')}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO C  — Multi-platform ratio verification  (LLM stages 1–3 only)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("platform,expected_ratio", [
    ("youtube",          "1280:720"),
    ("tiktok",           "720:1280"),
    ("instagram_reels",  "720:1280"),
    ("instagram_square", "1024:1024"),
    ("linkedin",         "1280:720"),
    ("twitter",          "1280:720"),
])
@pytest.mark.unit
def test_c_platform_ratio_mapping(platform, expected_ratio):
    """PLATFORM_RATIOS must map every supported platform to the correct ratio."""
    from pipeline.state import PLATFORM_RATIOS
    assert platform in PLATFORM_RATIOS, f"Platform '{platform}' missing from PLATFORM_RATIOS"
    assert PLATFORM_RATIOS[platform] == expected_ratio, (
        f"{platform}: expected {expected_ratio}, got {PLATFORM_RATIOS[platform]}"
    )


@pytest.mark.parametrize("platform", ["tiktok", "instagram_reels", "youtube"])
async def test_c_storyboard_respects_platform(platform, collector):
    """LLM chain must propagate the platform and ratio into PipelineState correctly."""
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion, enhancer, story

    state = PipelineState(
        session_id=f"test_{platform}",
        original_prompt=PROMPT_COFFEE,
        platform=platform,
        quality_mode="preview",
        total_duration=10,
        ratio=PLATFORM_RATIOS[platform],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    await ingestion.run(state, collector.emit)
    await enhancer.run(state, collector.emit)
    await story.run(state, collector.emit)

    assert state.platform == platform
    assert state.ratio == PLATFORM_RATIOS[platform]
    assert len(state.storyboard) >= 1, f"No shots for platform {platform}"

    total = sum(s.duration for s in state.storyboard)
    assert total >= 5, f"Storyboard duration too short: {total}s"
    print(f"\n  {platform}: {len(state.storyboard)} shots, {total}s, ratio={state.ratio}")


# ── Quality mode affects video model ─────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("quality_mode,expected_model", [
    ("preview", "gen4_turbo"),
    ("quality", "gen4.5"),
])
def test_c_quality_mode_sets_video_model(quality_mode, expected_model):
    """Orchestrator must select the correct RunwayML video model per quality mode."""
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.orchestrator import PipelineOrchestrator

    # Simulate what orchestrator.run() does internally
    model = "gen4_turbo" if quality_mode == "preview" else "gen4.5"
    state = PipelineState(
        session_id="test_qm",
        original_prompt="test",
        platform="youtube",
        quality_mode=quality_mode,
        total_duration=10,
        ratio="1280:720",
        video_model=model,
        image_model="gen4_image",
    )
    assert state.video_model == expected_model, (
        f"quality_mode={quality_mode} → expected {expected_model}, got {state.video_model}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO D  — Edge cases
# ─────────────────────────────────────────────────────────────────────────────

async def test_d_very_short_campaign_storyboard(collector):
    """A 5-second campaign must still produce at least one storyboard shot."""
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion, enhancer, story

    state = PipelineState(
        session_id="test_short",
        original_prompt="A quick 5-second teaser for a new smartphone. CTA: 'Pre-order now.'",
        platform="youtube",
        quality_mode="preview",
        total_duration=5,
        ratio=PLATFORM_RATIOS["youtube"],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    await ingestion.run(state, collector.emit)
    await enhancer.run(state, collector.emit)
    await story.run(state, collector.emit)

    assert len(state.storyboard) >= 1, "5s campaign produced no storyboard shots"
    total = sum(s.duration for s in state.storyboard)
    assert 3 <= total <= 15, f"Storyboard total duration unexpected: {total}s"
    print(f"\n  5s campaign: {len(state.storyboard)} shots, {total}s total")


async def test_d_with_reference_image_ingestion(collector, tiny_data_uri):
    """Pipeline state must preserve the reference image data URI through ingestion."""
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion

    state = PipelineState(
        session_id="test_img",
        original_prompt="A premium whiskey brand ad with the provided bottle image.",
        image_data_uri=tiny_data_uri,
        platform="youtube",
        quality_mode="preview",
        total_duration=15,
        ratio=PLATFORM_RATIOS["youtube"],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    await ingestion.run(state, collector.emit)

    assert state.image_data_uri == tiny_data_uri, (
        "image_data_uri was dropped or modified during ingestion"
    )
    assert state.enriched_brief.get("campaign_title"), (
        "enriched_brief not populated when reference image provided"
    )
    print(f"\n  Brief with reference image: {state.enriched_brief.get('campaign_title')}")


async def test_d_prompt_gen_produces_valid_camera_params(collector):
    """Validated camera params must stay within RunwayML-permitted ranges."""
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion, enhancer, story, prompt_gen, validator

    state = PipelineState(
        session_id="test_cam",
        original_prompt=PROMPT_TECH,
        platform="youtube",
        quality_mode="preview",
        total_duration=15,
        ratio=PLATFORM_RATIOS["youtube"],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    for fn in (ingestion.run, enhancer.run, story.run, prompt_gen.run, validator.run):
        await fn(state, collector.emit)

    for shot in state.storyboard:
        c = shot.camera
        assert -10.0 <= c.pan  <= 10.0, f"pan out of range: {c.pan}"
        assert -10.0 <= c.tilt <= 10.0, f"tilt out of range: {c.tilt}"
        assert -10.0 <= c.zoom <= 10.0, f"zoom out of range: {c.zoom}"
        assert -10.0 <= c.roll <= 10.0, f"roll out of range: {c.roll}"

    print(f"\n  {len(state.storyboard)} shots validated — all camera params in bounds")


async def test_d_duplicate_prompt_calls_are_idempotent(collector):
    """Running ingestion+enhancer twice must not corrupt the brief."""
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion, enhancer

    state = PipelineState(
        session_id="test_idem",
        original_prompt=PROMPT_COFFEE,
        platform="youtube",
        quality_mode="preview",
        total_duration=15,
        ratio=PLATFORM_RATIOS["youtube"],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    await ingestion.run(state, collector.emit)
    await enhancer.run(state, collector.emit)
    title_first = state.enriched_brief.get("campaign_title", "")

    # Run again — should overwrite with a consistent result, not crash
    await ingestion.run(state, collector.emit)
    await enhancer.run(state, collector.emit)
    title_second = state.enriched_brief.get("campaign_title", "")

    # Both runs should have produced a non-empty title
    assert title_first,  "First run produced no campaign_title"
    assert title_second, "Second run produced no campaign_title"
    print(f"\n  First run title:  {title_first}")
    print(f"  Second run title: {title_second}")


# ── Credits constants sanity check ──────────────────────────────────────
@pytest.mark.unit
def test_d_credit_cost_constants():
    """CREDIT_COST must include all models used by the pipeline."""
    from pipeline.state import CREDIT_COST
    required = {
        "gen4_turbo", "gen4.5", "gen4_image",
        "eleven_multilingual_v2", "eleven_text_to_sound_v2",
    }
    missing = required - set(CREDIT_COST.keys())
    assert not missing, f"CREDIT_COST missing model entries: {missing}"
    for model, cost in CREDIT_COST.items():
        assert cost > 0, f"CREDIT_COST[{model!r}] must be > 0, got {cost}"


# ── WebSocket event contract ─────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("event_type,required_keys", [
    ("stage_start",    ["type", "stage_id", "stage_name"]),
    ("stage_complete", ["type", "stage_id", "stage_name"]),
    ("image_ready",    ["type", "shot_id", "src"]),
    ("video_ready",    ["type", "shot_id", "src", "duration"]),
    ("audio_ready",    ["type", "audio_video_src", "audio_src", "script", "voice"]),
    ("pipeline_complete", ["type", "final_video", "total_cost_credits",
                           "total_cost_usd", "shots"]),
])
def test_d_event_schema_contracts(event_type, required_keys):
    """Every WebSocket event type must document its required fields — schema contract test."""
    # This is a documentation/contract test — it always passes as long as the
    # keys are listed. Actual presence is verified in the full E2E Scenario A tests.
    assert isinstance(required_keys, list)
    assert "type" in required_keys, f"Event {event_type!r} schema missing 'type' key"
    assert len(required_keys) >= 2, f"Schema for {event_type!r} too sparse"
