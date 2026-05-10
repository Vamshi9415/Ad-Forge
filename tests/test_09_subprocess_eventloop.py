"""Test 09 — Event-loop compatibility for subprocess calls.

Regression test for the silent NotImplementedError that crashed Stage 8
when uvicorn's reloader used Windows SelectorEventLoop.

Root cause: asyncio.create_subprocess_exec is NOT supported on Windows
SelectorEventLoop. It raises NotImplementedError with an empty message,
which surfaced in our orchestrator as "Pipeline error" with no detail.

Fix: use subprocess.run wrapped in asyncio.to_thread (or asyncio.run_in_executor)
which works on every event loop on every platform.

These tests verify that orchestrator._stitch and audio_gen.mux_audio_video
still work even when the event loop does NOT natively support subprocesses.
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ─── Module-scoped fixtures ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def ffmpeg_exe():
    from pipeline.orchestrator import _find_ffmpeg
    return shutil.which("ffmpeg") or _find_ffmpeg()


async def _make_test_clip(ffmpeg: str, path: Path, duration: int = 1, color: str = "red"):
    """Create a tiny synthetic mp4 clip using sync subprocess (always works)."""
    subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:size=64x36:rate=5:duration={duration}",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True, capture_output=True, timeout=30,
    )


# ─── Test 1: Confirm SelectorEventLoop fails at create_subprocess_exec ────
def test_selector_event_loop_does_not_support_subprocess():
    """Documentation/regression: SelectorEventLoop on Windows raises
    NotImplementedError when create_subprocess_exec is called.
    This is the exact failure mode the orchestrator originally hit."""
    if sys.platform != "win32":
        pytest.skip("Windows-only behavior")

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        async def attempt():
            await asyncio.create_subprocess_exec(
                "cmd", "/c", "echo", "hi",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        with pytest.raises(NotImplementedError):
            loop.run_until_complete(attempt())
    finally:
        loop.close()
        # Restore default policy for subsequent tests
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ─── Test 2: Orchestrator stitch survives SelectorEventLoop ───────────────
def test_orchestrator_stitch_works_on_selector_event_loop(tmp_path, ffmpeg_exe):
    """Reproduce the exact failure mode: run _stitch inside the event loop
    that originally caused the silent 'Pipeline error' crash."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not available")
    if sys.platform != "win32":
        pytest.skip("Windows-only regression test")

    from pipeline.state import PipelineState, Shot
    from pipeline.orchestrator import PipelineOrchestrator

    # Set up two real test clips on disk
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    asyncio.run(_make_test_clip(ffmpeg_exe, vid_dir / "shot_01.mp4", 1, "red"))
    asyncio.run(_make_test_clip(ffmpeg_exe, vid_dir / "shot_02.mp4", 1, "blue"))

    # Force the event loop policy that originally broke the pipeline
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        async def run():
            messages = []

            async def send(m):
                messages.append(m)

            state = PipelineState(
                session_id="selector_test",
                original_prompt="test",
                total_duration=2,
            )
            for sid in ("shot_01", "shot_02"):
                s = Shot(shot_id=sid, narrative_function="t", duration=1,
                         subject="t", environment="t", action="t")
                s.video_status = "complete"
                state.storyboard.append(s)

            orch = PipelineOrchestrator("selector_test", tmp_path, send)
            return await orch._stitch(state), messages

        result, messages = loop.run_until_complete(run())

        # CRITICAL ASSERTION: stitch must succeed even on SelectorEventLoop
        assert result is not None, (
            "Stitch returned None on SelectorEventLoop! Logs:\n" +
            "\n".join(m.get("message", "") for m in messages if m.get("type") == "log")
        )
        assert Path(result).exists(), f"Output file missing: {result}"
        assert Path(result).stat().st_size > 1000, "Output file suspiciously small"
        print(f"\n  Stitch on SelectorEventLoop: {Path(result).stat().st_size // 1024} KB")
    finally:
        loop.close()
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ─── Test 3: Stitch with 5 clips (matches real pipeline scenario) ─────────
def test_orchestrator_stitch_5_clips_real_scenario(tmp_path, ffmpeg_exe):
    """Reproduces the exact 5-clip scenario from real campaigns (Old Mumbai
    Ice Cream / Sincerely AI). This is what stage 8 sees in production."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not available")

    from pipeline.state import PipelineState, Shot
    from pipeline.orchestrator import PipelineOrchestrator

    # Match real session structure: session_dir contains videos/ subdir
    session_dir = tmp_path / "output" / "deadbeef"
    (session_dir / "videos").mkdir(parents=True)

    state = PipelineState(
        session_id="deadbeef", original_prompt="test", total_duration=30
    )
    durations = [10, 5, 5, 5, 5]
    colors = ["red", "blue", "green", "yellow", "magenta"]
    for i, (d, c) in enumerate(zip(durations, colors), 1):
        sid = f"shot_{i:02d}"
        asyncio.run(_make_test_clip(ffmpeg_exe, session_dir / "videos" / f"{sid}.mp4", 1, c))
        s = Shot(shot_id=sid, narrative_function="", duration=d,
                 subject="", environment="", action="")
        s.video_status = "complete"
        state.storyboard.append(s)

    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    async def run():
        msgs = []

        async def send(m):
            msgs.append(m)

        orch = PipelineOrchestrator("deadbeef", session_dir, send)
        return await orch._stitch(state), msgs

    result, messages = asyncio.run(run())

    assert result is not None, (
        "5-clip stitch failed. Logs:\n" +
        "\n".join(m.get("message", "") for m in messages if m.get("type") == "log")
    )
    final = Path(result)
    assert final.exists()
    size_mb = final.stat().st_size / (1024 * 1024)
    assert size_mb > 0.001, f"Output too small: {size_mb}"

    # Critical: every line in concat.txt must be absolute
    concat_lines = (session_dir / "concat.txt").read_text().strip().splitlines()
    assert len(concat_lines) == 5, f"Expected 5 concat entries, got {len(concat_lines)}"
    for line in concat_lines:
        path_str = line.replace("file '", "").rstrip("'")
        import os
        assert os.path.isabs(path_str), (
            f"Non-absolute path in concat.txt: {path_str}"
        )
    print(f"\n  5-clip stitch: {size_mb*1024:.1f} KB, all paths absolute")


# ─── Test 4: Audio mux survives SelectorEventLoop ─────────────────────────
async def test_audio_gen_mux_uses_thread_subprocess(tmp_path, ffmpeg_exe):
    """audio_gen.mux_audio_video must work on any event loop."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not available")

    # Build a synthetic video and audio file
    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.mp3"
    await _make_test_clip(ffmpeg_exe, video, duration=2, color="cyan")
    subprocess.run(
        [ffmpeg_exe, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:a", "libmp3lame", "-b:a", "128k", str(audio)],
        check=True, capture_output=True, timeout=15,
    )

    from pipeline.agents.audio_gen import mux_audio_video

    output = tmp_path / "out.mp4"
    msgs = []

    async def emit(m): msgs.append(m)

    result = await mux_audio_video(
        video_path=video,
        audio_path=audio,
        output_path=output,
        total_duration=2,
        emit=emit,
    )

    assert result is not None, (
        "Mux returned None. Logs:\n" +
        "\n".join(m.get("message", "") for m in msgs if m.get("type") == "log")
    )
    assert output.exists() and output.stat().st_size > 1000


# ─── Test 5: Verify orchestrator no longer uses asyncio.create_subprocess_exec ─
def test_orchestrator_does_not_use_async_subprocess():
    """Source-level guard: orchestrator must NOT call asyncio.create_subprocess_exec
    (which fails silently on Windows SelectorEventLoop)."""
    import inspect
    from pipeline import orchestrator
    src = inspect.getsource(orchestrator)
    assert "create_subprocess_exec" not in src, (
        "orchestrator.py still uses asyncio.create_subprocess_exec — "
        "this fails silently on Windows SelectorEventLoop! "
        "Use asyncio.to_thread(subprocess.run, ...) instead."
    )
    assert "asyncio.to_thread" in src or "subprocess.run" in src, (
        "orchestrator.py should use subprocess.run + asyncio.to_thread"
    )


def test_audio_gen_does_not_use_async_subprocess():
    """Source-level guard for audio_gen too."""
    import inspect
    from pipeline.agents import audio_gen
    src = inspect.getsource(audio_gen)
    assert "create_subprocess_exec" not in src, (
        "audio_gen.py still uses asyncio.create_subprocess_exec — "
        "this fails silently on Windows SelectorEventLoop!"
    )
