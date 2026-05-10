"""Shared fixtures and session-scoped LLM pipeline state.

The heavy LLM chain (stages 1-5) runs ONCE per test session and results are
reused across tests — avoids paying for duplicate API calls.
"""

import asyncio
import base64
import os
import time
import uuid
from pathlib import Path
from typing import List

import pytest
from dotenv import load_dotenv

# Always load .env before anything else
load_dotenv(override=True)

# ─── project root on sys.path ─────────────────────────────────────────────
import sys
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── helper: tiny valid 1×1 JPEG (white pixel) ────────────────────────────
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA"
    "/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9"
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0"
    "NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWW"
    "l5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4"
    "+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APvSyjooA//Z"
)

TINY_JPEG_BYTES: bytes = base64.b64decode(TINY_JPEG_B64.replace("\n", "").replace(" ", ""))
TINY_DATA_URI = f"data:image/jpeg;base64,{base64.b64encode(TINY_JPEG_BYTES).decode()}"


# ─── Async emit collector ─────────────────────────────────────────────────
class MsgCollector:
    """Collects all WebSocket-style messages emitted during a pipeline run."""

    def __init__(self, silent: bool = False):
        self.messages: List[dict] = []
        self.silent = silent

    async def emit(self, msg: dict):
        self.messages.append(msg)
        if not self.silent and msg.get("type") == "log":
            lvl = msg.get("level", "info")
            col = {"success": "\033[92m", "error": "\033[91m", "warning": "\033[93m"}.get(lvl, "")
            print(f"    [{msg.get('stage','?')}] {col}{msg.get('message','')}\033[0m")

    def logs(self, stage=None) -> List[str]:
        return [m["message"] for m in self.messages
                if m.get("type") == "log" and (stage is None or m.get("stage") == stage)]

    def of_type(self, t: str) -> List[dict]:
        return [m for m in self.messages if m.get("type") == t]


# ─── Session-level LLM pipeline fixture ──────────────────────────────────
_pipeline_state_cache = {}


@pytest.fixture(scope="session")
def llm_state():
    """Run stages 1-5 once for the whole test session. Returns a dict with
    state and per-stage timing information."""
    return asyncio.get_event_loop().run_until_complete(_build_llm_state())


async def _build_llm_state():
    from pipeline.state import PipelineState, PLATFORM_RATIOS
    from pipeline.agents import ingestion, enhancer, story, prompt_gen, validator

    state = PipelineState(
        session_id="test_session",
        original_prompt=(
            "A premium cold brew coffee brand targeting urban millennials aged 25-35. "
            "Emphasise the slow 24-hour brew process, sustainably sourced single-origin beans, "
            "and artisan craft. Emotion: calm confidence. CTA: 'Slow is Bold.'"
        ),
        platform="youtube",
        quality_mode="preview",
        total_duration=15,
        ratio=PLATFORM_RATIOS["youtube"],
        video_model="gen4_turbo",
        image_model="gen4_image",
    )
    collector = MsgCollector(silent=True)
    timings = {}

    for name, fn in [
        ("ingestion",  ingestion.run),
        ("enhancer",   enhancer.run),
        ("story",      story.run),
        ("prompt_gen", prompt_gen.run),
        ("validator",  validator.run),
    ]:
        t0 = time.time()
        await fn(state, collector.emit)
        timings[name] = round(time.time() - t0, 2)

    return {"state": state, "collector": collector, "timings": timings}


@pytest.fixture()
def collector():
    return MsgCollector()


@pytest.fixture()
def tiny_jpeg_bytes():
    return TINY_JPEG_BYTES


@pytest.fixture()
def tiny_data_uri():
    return TINY_DATA_URI


@pytest.fixture()
def tmp_session_dir(tmp_path):
    d = tmp_path / "session_test"
    (d / "images").mkdir(parents=True)
    (d / "videos").mkdir(parents=True)
    return d
