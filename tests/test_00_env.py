"""Test 00 — Environment, Dependencies, and Recursive Module Import Trace.

Verifies:
- All required packages are importable
- .env file exists and contains both API keys
- Every pipeline module imports cleanly (recursive trace)
- No circular imports or missing dependencies
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ─── Required packages ───────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    ("fastapi",          "FastAPI web framework"),
    ("uvicorn",          "ASGI server"),
    ("dotenv",           "python-dotenv for .env loading"),
    ("openai",           "OpenAI Python SDK"),
    ("httpx",            "Async HTTP client"),
    ("aiofiles",         "Async file I/O"),
    ("multipart",        "python-multipart for file uploads"),
    ("websockets",       "WebSocket client for tests"),
    ("pydantic",         "Data validation"),
]


@pytest.mark.parametrize("pkg,description", REQUIRED_PACKAGES)
def test_package_importable(pkg, description):
    """Every required package must be importable."""
    try:
        importlib.import_module(pkg)
    except ImportError as exc:
        pytest.fail(f"Cannot import '{pkg}' ({description}): {exc}")


def test_python_version():
    """Python 3.10+ required for union type hints."""
    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 10, (
        f"Python 3.10+ required, got {major}.{minor}"
    )


# ─── .env file ───────────────────────────────────────────────────────────
def test_env_file_exists():
    root = Path(__file__).parent.parent
    assert (root / ".env").exists(), ".env file not found in project root"


def test_runwayml_key_set():
    key = os.environ.get("RUNWAYML_API_SECRET", "")
    assert key and key != "dummy" and len(key) > 20, (
        "RUNWAYML_API_SECRET not set or is a dummy value"
    )


def test_openai_key_set():
    key = os.environ.get("OPENAI_API_KEY", "")
    assert key and key != "dummy" and len(key) > 20, (
        "OPENAI_API_KEY not set or is a dummy value"
    )


def test_key_prefix_format():
    """RunwayML keys start with 'key_', OpenAI keys start with 'sk-'."""
    rw  = os.environ.get("RUNWAYML_API_SECRET", "")
    oai = os.environ.get("OPENAI_API_KEY", "")
    assert rw.startswith("key_"),  f"RunwayML key should start with 'key_', got: {rw[:10]}"
    assert oai.startswith("sk-"),  f"OpenAI key should start with 'sk-', got: {oai[:10]}"


# ─── Recursive module import trace ────────────────────────────────────────
PIPELINE_MODULES = [
    "pipeline",
    "pipeline.state",
    "pipeline.runway_client",
    "pipeline.orchestrator",
    "pipeline.agents",
    "pipeline.agents.ingestion",
    "pipeline.agents.enhancer",
    "pipeline.agents.story",
    "pipeline.agents.prompt_gen",
    "pipeline.agents.validator",
    "pipeline.agents.image_synth",
    "pipeline.agents.vlm_validator",
    "pipeline.agents.video_gen",
    "pipeline.agents.audio_gen",
]


@pytest.mark.parametrize("module_path", PIPELINE_MODULES)
def test_module_imports_clean(module_path):
    """Every pipeline module must import without errors."""
    try:
        mod = importlib.import_module(module_path)
        assert mod is not None
    except ImportError as exc:
        pytest.fail(f"ImportError in {module_path}: {exc}")
    except SyntaxError as exc:
        pytest.fail(f"SyntaxError in {module_path}: {exc}")
    except Exception as exc:
        pytest.fail(f"Unexpected error importing {module_path}: {type(exc).__name__}: {exc}")


def test_server_module_imports():
    """server.py must import cleanly."""
    try:
        import server  # noqa: F401
    except Exception as exc:
        pytest.fail(f"server.py import error: {exc}")


def test_no_circular_imports():
    """Import all modules in different orders — circular imports would fail."""
    import importlib
    modules = [
        "pipeline.state",
        "pipeline.runway_client",
        "pipeline.agents.audio_gen",
        "pipeline.agents.vlm_validator",
        "pipeline.orchestrator",
    ]
    for mod in modules:
        if mod in sys.modules:
            del sys.modules[mod]
    for mod in reversed(modules):
        importlib.import_module(mod)


# ─── Key attributes exist in each module ─────────────────────────────────
def test_state_exports():
    from pipeline.state import PipelineState, Shot, CameraParams, PLATFORM_RATIOS, CREDIT_COST
    assert PipelineState
    assert Shot
    assert CameraParams
    assert isinstance(PLATFORM_RATIOS, dict)
    assert isinstance(CREDIT_COST, dict)


def test_orchestrator_exports():
    from pipeline.orchestrator import PipelineOrchestrator, _find_ffmpeg
    assert callable(PipelineOrchestrator)
    assert callable(_find_ffmpeg)


def test_runway_client_exports():
    from pipeline.runway_client import generate_image, generate_video, download_bytes, poll_task
    assert all(callable(f) for f in [generate_image, generate_video, download_bytes, poll_task])


def test_agents_have_run_function():
    """Every agent module must expose a top-level `run` coroutine."""
    import inspect
    agents = [
        "pipeline.agents.ingestion",
        "pipeline.agents.enhancer",
        "pipeline.agents.story",
        "pipeline.agents.prompt_gen",
        "pipeline.agents.validator",
        "pipeline.agents.image_synth",
        "pipeline.agents.vlm_validator",
        "pipeline.agents.video_gen",
        "pipeline.agents.audio_gen",
    ]
    for mod_path in agents:
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, "run"), f"{mod_path} missing `run` function"
        assert inspect.iscoroutinefunction(mod.run), f"{mod_path}.run must be async"


def test_vlm_validator_has_retry_config():
    from pipeline.agents import vlm_validator
    assert hasattr(vlm_validator, "MAX_RETRIES"), "MAX_RETRIES not found"
    assert vlm_validator.MAX_RETRIES >= 1, "MAX_RETRIES should be >= 1"
    assert hasattr(vlm_validator, "_validate_one_shot"), "_validate_one_shot not found"


def test_audio_gen_exports():
    from pipeline.agents import audio_gen
    assert hasattr(audio_gen, "VOICE_MAP")
    assert hasattr(audio_gen, "_find_ffmpeg")
    assert "female" in audio_gen.VOICE_MAP
    assert "male" in audio_gen.VOICE_MAP


def test_static_html_exists():
    root = Path(__file__).parent.parent
    html = root / "static" / "index.html"
    assert html.exists(), "static/index.html not found"
    content = html.read_text(encoding="utf-8")
    assert "AdForge" in content, "index.html missing 'AdForge' title"
    assert "stage_9" in content or "sd-9" in content, (
        "index.html missing Stage 9 DOM element"
    )
    assert "audio_ready" in content, "index.html missing audio_ready handler"
