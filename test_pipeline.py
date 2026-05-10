"""Comprehensive test suite for the AdForge pipeline.

Tests each stage individually and validates end-to-end flow.
Run with: python test_pipeline.py
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── colour helpers ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[PASS]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{BLUE}{'='*60}{RESET}\n{BOLD} {msg}{RESET}\n{BLUE}{'='*60}{RESET}")
def sub(msg): print(f"\n{BOLD}-- {msg} --{RESET}")

PASSED = []
FAILED = []

def record(name, success, detail=""):
    if success:
        PASSED.append(name)
        ok(f"PASS: {name}")
    else:
        FAILED.append(name)
        fail(f"FAIL: {name}" + (f" — {detail}" if detail else ""))


# ── Shared emit collector ──────────────────────────────────────────────────
class Collector:
    def __init__(self):
        self.messages = []

    async def emit(self, msg):
        self.messages.append(msg)
        level = msg.get("level", "info")
        text  = msg.get("message", "")
        stage = msg.get("stage", msg.get("stage_id", "?"))
        if msg["type"] == "log":
            colour = GREEN         if level == "success" else (RED if level == "error" else (YELLOW if level == "warning" else RESET))
            print(f"    [{stage}] {colour}{text}{RESET}")
        elif msg["type"] in ("stage_start", "stage_complete"):
            icon = ">" if msg["type"] == "stage_start" else "+"
            print(f"  {CYAN}{icon} Stage {stage}: {msg.get('stage_name', '')}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Environment / API keys
# ─────────────────────────────────────────────────────────────────────────────
async def test_environment():
    header("TEST 1 — Environment & API Keys")

    runway_key = os.environ.get("RUNWAYML_API_SECRET", "")
    openai_key  = os.environ.get("OPENAI_API_KEY", "")

    record("RUNWAYML_API_SECRET set",  bool(runway_key),  "Missing key")
    record("OPENAI_API_KEY set",       bool(openai_key),  "Missing key")

    if runway_key:
        info(f"RunwayML key: {runway_key[:12]}…{runway_key[-4:]}")
    if openai_key:
        info(f"OpenAI key:   {openai_key[:12]}…{openai_key[-4:]}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — RunwayML API connection
# ─────────────────────────────────────────────────────────────────────────────
async def test_runway_connection():
    header("TEST 2 — RunwayML API Connection")
    import httpx

    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        warn("Skipping — no API key")
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://api.dev.runwayml.com/v1/tasks/nonexistent-id-test",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-Runway-Version": "2024-11-06",
                },
            )
        # 404 means we're authenticated but the task doesn't exist — that's fine
        if r.status_code in (404, 422):
            record("RunwayML API reachable & authenticated", True)
            info(f"Got expected {r.status_code} response (unauthenticated would be 401)")
        elif r.status_code == 401:
            record("RunwayML API reachable & authenticated", False, "401 — invalid API key")
        else:
            record("RunwayML API reachable & authenticated", True)
            info(f"Response: {r.status_code}")
    except Exception as exc:
        record("RunwayML API reachable & authenticated", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — OpenAI API connection
# ─────────────────────────────────────────────────────────────────────────────
async def test_openai_connection():
    header("TEST 3 — OpenAI API Connection")
    from openai import AsyncOpenAI

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        warn("Skipping — no API key")
        return

    try:
        client = AsyncOpenAI(api_key=key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with just the word: CONNECTED"}],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content.strip()
        record("OpenAI API connection", "CONNECTED" in reply.upper(), f"Got: {reply}")
        info(f"Model: {resp.model} | Tokens: {resp.usage.total_tokens}")
    except Exception as exc:
        record("OpenAI API connection", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — _solve_duration math
# ─────────────────────────────────────────────────────────────────────────────
async def test_duration_solver():
    header("TEST 4 — Duration Solver (API Constraint Enforcement)")
    from pipeline.agents.story import _solve_duration

    sub("Valid duration decompositions")
    cases = [10, 15, 20, 25, 30, 40, 50, 60]
    all_ok = True
    for total in cases:
        result = _solve_duration(total)
        total_check = sum(result)
        valid_check = all(d in (5, 8, 10) for d in result)
        status = total_check == total and valid_check
        if status:
            info(f"{total}s → {result} (sum={total_check}) ✓")
        else:
            fail(f"{total}s → {result} — INVALID (sum={total_check})")
            all_ok = False

    record("All durations decompose correctly", all_ok)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Stage 1: Ingestion Agent
# ─────────────────────────────────────────────────────────────────────────────
async def test_ingestion():
    header("TEST 5 — Stage 1: Ingestion Agent")
    from pipeline.agents import ingestion
    from pipeline.state import PipelineState

    state = PipelineState(
        session_id="test",
        original_prompt="A premium cold brew coffee brand targeting urban millennials aged 25-35. Emphasise the slow 24-hour brew process and sustainably sourced beans. CTA: 'Slow is Bold.'",
        platform="youtube",
        quality_mode="preview",
        total_duration=30,
    )

    c = Collector()
    try:
        await ingestion.run(state, c.emit)
        brief = state.enriched_brief
        has_product  = bool(brief.get("product_name"))
        has_tone     = bool(brief.get("emotional_tone"))
        has_cta      = bool(brief.get("primary_cta"))
        record("Ingestion returns product_name", has_product,  str(brief.get("product_name")))
        record("Ingestion returns emotional_tone", has_tone,   str(brief.get("emotional_tone")))
        record("Ingestion returns primary_cta",   has_cta,     str(brief.get("primary_cta")))
        info(f"Brief: {json.dumps(brief, indent=2)[:500]}")
        return state
    except Exception as exc:
        record("Ingestion agent runs without error", False, str(exc))
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Stage 2: Enhancer Agent
# ─────────────────────────────────────────────────────────────────────────────
async def test_enhancer(state):
    header("TEST 6 — Stage 2: Contextual Enhancer")
    if state is None:
        warn("Skipping — no state from Stage 1")
        return state

    from pipeline.agents import enhancer
    c = Collector()
    try:
        await enhancer.run(state, c.emit)
        has_campaign = bool(state.enriched_brief.get("campaign_title"))
        has_style    = bool(state.enriched_brief.get("visual_style"))
        has_pacing   = bool(state.enriched_brief.get("pacing"))
        record("Enhancer adds campaign_title",  has_campaign, str(state.enriched_brief.get("campaign_title")))
        record("Enhancer adds visual_style",    has_style,    str(state.enriched_brief.get("visual_style")))
        record("Enhancer adds pacing",          has_pacing,   str(state.enriched_brief.get("pacing")))
        return state
    except Exception as exc:
        record("Enhancer agent runs without error", False, str(exc))
        traceback.print_exc()
        return state


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Stage 3: Story Agent
# ─────────────────────────────────────────────────────────────────────────────
async def test_story(state):
    header("TEST 7 — Stage 3: Story / Storyboard Agent")
    if state is None:
        warn("Skipping — no state")
        return state

    from pipeline.agents import story
    c = Collector()
    try:
        await story.run(state, c.emit)
        shots = state.storyboard
        has_shots    = len(shots) > 0
        sum_dur      = sum(s.duration for s in shots)
        valid_durs   = all(s.duration in (5, 8, 10) for s in shots)
        correct_sum  = sum_dur == state.total_duration

        record("Storyboard has shots",          has_shots,   f"{len(shots)} shots")
        record("All durations valid (5/8/10)",  valid_durs,  str([s.duration for s in shots]))
        record("Total duration matches target", correct_sum, f"{sum_dur}s vs {state.total_duration}s target")

        sub("Generated storyboard")
        for s in shots:
            info(f"  {s.shot_id} [{s.duration}s] — {s.narrative_function}")
            info(f"    Subject: {s.subject[:80]}")
        return state
    except Exception as exc:
        record("Story agent runs without error", False, str(exc))
        traceback.print_exc()
        return state


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Stage 4: Prompt Generator
# ─────────────────────────────────────────────────────────────────────────────
async def test_prompt_gen(state):
    header("TEST 8 — Stage 4: Prompt Generator")
    if not state or not state.storyboard:
        warn("Skipping — no storyboard")
        return state

    from pipeline.agents import prompt_gen
    c = Collector()
    try:
        await prompt_gen.run(state, c.emit)
        all_have_prompt  = all(bool(s.image_prompt) for s in state.storyboard)
        all_have_motion  = all(bool(s.motion_prompt) for s in state.storyboard)
        all_have_seed    = all(s.seed is not None for s in state.storyboard)
        record("All shots have image_prompt",  all_have_prompt)
        record("All shots have motion_prompt", all_have_motion)
        record("All shots have seed set",      all_have_seed)

        sub("Sample prompts")
        for s in state.storyboard[:2]:
            info(f"  {s.shot_id} image:  {s.image_prompt[:100]}")
            info(f"  {s.shot_id} motion: {s.motion_prompt[:80]}")
            info(f"  {s.shot_id} camera: pan={s.camera.pan}, tilt={s.camera.tilt}, zoom={s.camera.zoom}")
        return state
    except Exception as exc:
        record("Prompt generator runs without error", False, str(exc))
        traceback.print_exc()
        return state


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Stage 5: Validator
# ─────────────────────────────────────────────────────────────────────────────
async def test_validator(state):
    header("TEST 9 — Stage 5: Structural Validator")
    if not state or not state.storyboard:
        warn("Skipping — no storyboard")
        return state

    from pipeline.agents import validator
    c = Collector()
    try:
        await validator.run(state, c.emit)
        all_valid_durs  = all(s.duration in (5, 8, 10) for s in state.storyboard)
        all_have_prompt = all(bool(s.image_prompt) for s in state.storyboard)
        record("Post-validation: all durations valid",    all_valid_durs)
        record("Post-validation: all prompts non-empty",  all_have_prompt)
        return state
    except Exception as exc:
        record("Validator runs without error", False, str(exc))
        traceback.print_exc()
        return state


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — FastAPI HTTP endpoints
# ─────────────────────────────────────────────────────────────────────────────
async def test_http_endpoints():
    header("TEST 10 — FastAPI HTTP Endpoints")
    import httpx

    base = "http://localhost:8000"
    async with httpx.AsyncClient(timeout=10) as client:

        sub("GET /")
        try:
            r = await client.get(f"{base}/")
            record("GET / returns 200", r.status_code == 200, f"Got {r.status_code}")
            record("Response is HTML", "text/html" in r.headers.get("content-type",""), r.headers.get("content-type"))
        except Exception as exc:
            record("GET / reachable", False, str(exc))

        sub("GET /docs (OpenAPI)")
        try:
            r = await client.get(f"{base}/docs")
            record("GET /docs returns 200", r.status_code == 200, f"Got {r.status_code}")
        except Exception as exc:
            record("GET /docs reachable", False, str(exc))

        sub("POST /api/upload-image")
        try:
            import base64
            tiny_png = base64.b64decode(
                b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
            r = await client.post(
                f"{base}/api/upload-image",
                files={"file": ("test.png", tiny_png, "image/png")},
            )
            record("POST /api/upload-image returns 200", r.status_code == 200, f"Got {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                record("Upload response has data_uri", "data_uri" in data, str(data.keys()))
                record("data_uri is valid base64", data.get("data_uri","").startswith("data:image/"), data.get("data_uri","")[:30])
        except Exception as exc:
            record("POST /api/upload-image", False, str(exc))

        sub("GET /api/sessions/{id}/status (non-existent)")
        try:
            r = await client.get(f"{base}/api/sessions/nonexistent/status")
            record("Non-existent session returns 404", r.status_code == 404, f"Got {r.status_code}")
        except Exception as exc:
            record("Session status endpoint reachable", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11 — WebSocket connection
# ─────────────────────────────────────────────────────────────────────────────
async def test_websocket():
    header("TEST 11 — WebSocket Connection")
    try:
        import websockets
        async with websockets.connect("ws://localhost:8000/api/ws/invalid-session") as ws:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            record("WebSocket connects and responds", True)
            record("WebSocket returns error for bad session", msg.get("type") in ("error", "done"), str(msg))
    except ImportError:
        warn("websockets package not installed — skipping WS test")
    except Exception as exc:
        record("WebSocket test", False, str(exc)[:100])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12 — Full pipeline dry-run (stages 1-5 only, no RunwayML)
# ─────────────────────────────────────────────────────────────────────────────
async def test_full_llm_pipeline():
    header("TEST 12 — Full LLM Pipeline (Stages 1-5 Dry Run)")
    from pipeline.state import PipelineState, PLATFORM_RATIOS

    state = PipelineState(
        session_id="dryrun",
        original_prompt="An eco-friendly reusable water bottle brand for fitness enthusiasts. Features insulated stainless steel that keeps water cold for 24 hours. CTA: Stay Cold. Stay Bold.",
        platform="instagram_reels",
        quality_mode="preview",
        total_duration=15,
        ratio=PLATFORM_RATIOS.get("instagram_reels", "720:1280"),
        video_model="gen4_turbo",
        image_model="gen4_image",
    )

    c = Collector()

    from pipeline.agents import ingestion, enhancer, story, prompt_gen, validator

    pipeline_ok = True
    t_start = time.time()

    stages = [
        ("Ingestion",   ingestion.run,   [state, c.emit]),
        ("Enhancement", enhancer.run,    [state, c.emit]),
        ("Story",       story.run,       [state, c.emit]),
        ("PromptGen",   prompt_gen.run,  [state, c.emit]),
        ("Validator",   validator.run,   [state, c.emit]),
    ]

    for name, fn, args in stages:
        sub(f"Running {name}")
        t = time.time()
        try:
            await fn(*args)
            info(f"Completed in {time.time()-t:.1f}s")
        except Exception as exc:
            fail(f"{name} failed: {exc}")
            traceback.print_exc()
            pipeline_ok = False

    elapsed = time.time() - t_start
    record("Full LLM pipeline (stages 1-5) completes without error", pipeline_ok)

    if state.storyboard:
        info(f"\nFinal storyboard ({state.total_duration}s, {len(state.storyboard)} shots):")
        for s in state.storyboard:
            print(f"    {s.shot_id} [{s.duration}s] {s.narrative_function}")
            print(f"      Image prompt: {s.image_prompt[:90]}...")
            print(f"      Motion:       {s.motion_prompt[:70]}")

    info(f"\nTotal LLM pipeline time: {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{BOLD}{'='*60}")
    print(" AdForge - Comprehensive Pipeline Test Suite")
    print(f"{'='*60}{RESET}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD:    {Path.cwd()}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    await test_environment()
    await test_runway_connection()
    await test_openai_connection()
    await test_duration_solver()

    # LLM agent chain (stages 1-5)
    state = await test_ingestion()
    state = await test_enhancer(state)
    state = await test_story(state)
    state = await test_prompt_gen(state)
    state = await test_validator(state)

    # HTTP + WS
    await test_http_endpoints()
    await test_websocket()

    # Full dry-run
    await test_full_llm_pipeline()

    # ── Summary ───────────────────────────────────────────────────────────
    header("TEST SUMMARY")
    print(f"  {GREEN}PASSED: {len(PASSED)}{RESET}")
    for p in PASSED:
        print(f"    {GREEN}✓{RESET} {p}")
    if FAILED:
        print(f"\n  {RED}FAILED: {len(FAILED)}{RESET}")
        for f in FAILED:
            print(f"    {RED}✗{RESET} {f}")
    else:
        print(f"\n  {GREEN}{BOLD}All tests passed! Great job!{RESET}")

    print(f"\n  Total: {len(PASSED)+len(FAILED)} | Passed: {len(PASSED)} | Failed: {len(FAILED)}\n")
    return len(FAILED) == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
