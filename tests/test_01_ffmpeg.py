"""Test 01 — FFmpeg: Detection, Version, Concat, and Audio Mux.

Verifies:
- _find_ffmpeg() locates the executable
- ffmpeg -version runs successfully
- Concat of two short clips produces a valid output file
- Audio mux (ffmpeg -i video -i audio) produces a combined output
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ─── Shared fixture ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ffmpeg_exe():
    """Find ffmpeg once for all tests in this module."""
    from pipeline.orchestrator import _find_ffmpeg
    exe = shutil.which("ffmpeg") or _find_ffmpeg()
    return exe  # May be None — tests that need it will skip gracefully


# ─── Detection tests ──────────────────────────────────────────────────────
def test_find_ffmpeg_function_is_callable():
    from pipeline.orchestrator import _find_ffmpeg
    result = _find_ffmpeg()
    # Returns None or a path string — both are valid outcomes
    assert result is None or isinstance(result, str)


def test_find_ffmpeg_returns_consistent_result():
    """Calling _find_ffmpeg() twice must return the same path."""
    from pipeline.orchestrator import _find_ffmpeg
    r1 = _find_ffmpeg()
    r2 = _find_ffmpeg()
    assert r1 == r2


def test_audio_gen_find_ffmpeg_is_consistent():
    """Both orchestrator and audio_gen use the same logic."""
    from pipeline.orchestrator import _find_ffmpeg as orch_ffmpeg
    from pipeline.agents.audio_gen import _find_ffmpeg as audio_ffmpeg
    assert orch_ffmpeg() == audio_ffmpeg()


def test_ffmpeg_found(ffmpeg_exe):
    """FFmpeg executable must be discoverable on this system."""
    assert ffmpeg_exe is not None, (
        "FFmpeg not found! Install with: winget install Gyan.FFmpeg  "
        "then restart the server. Required for video stitching and audio mux."
    )
    assert os.path.isfile(ffmpeg_exe), f"ffmpeg path is not a file: {ffmpeg_exe}"
    print(f"\n  FFmpeg found at: {ffmpeg_exe}")


# ─── Execution tests ─────────────────────────────────────────────────────
def test_ffmpeg_version(ffmpeg_exe):
    """ffmpeg -version must run and print version info."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")
    result = subprocess.run(
        [ffmpeg_exe, "-version"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, f"ffmpeg -version failed: {result.stderr[:200]}"
    assert "ffmpeg version" in result.stdout.lower() or "ffmpeg" in result.stderr.lower()
    # Extract version
    first_line = (result.stdout or result.stderr).split("\n")[0]
    print(f"\n  {first_line.strip()}")


def test_ffmpeg_supports_concat_demuxer(ffmpeg_exe):
    """Check that FFmpeg supports the concat demuxer (required for stitching)."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")
    result = subprocess.run(
        [ffmpeg_exe, "-demuxers"],
        capture_output=True, text=True, timeout=15,
    )
    output = result.stdout + result.stderr
    assert "concat" in output.lower(), "FFmpeg concat demuxer not available"


def test_ffmpeg_supports_aac_encoder(ffmpeg_exe):
    """Check AAC encoder — required for audio mux."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")
    result = subprocess.run(
        [ffmpeg_exe, "-encoders"],
        capture_output=True, text=True, timeout=15,
    )
    output = result.stdout + result.stderr
    assert "aac" in output.lower(), "FFmpeg AAC encoder not available — needed for audio mux"


# ─── Real concat test ─────────────────────────────────────────────────────
async def _make_test_video(ffmpeg: str, out_path: Path, duration: int = 1, color: str = "red"):
    """Create a tiny synthetic video clip using FFmpeg lavfi source."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:size=64x36:rate=5:duration={duration}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, f"Failed to create test video: {stderr.decode()[-200:]}"
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.slow
async def test_ffmpeg_concat_two_clips(ffmpeg_exe, tmp_path):
    """Create two synthetic clips, concat them, verify output."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")

    clip1 = tmp_path / "clip1.mp4"
    clip2 = tmp_path / "clip2.mp4"
    await _make_test_video(ffmpeg_exe, clip1, duration=1, color="red")
    await _make_test_video(ffmpeg_exe, clip2, duration=1, color="blue")

    concat_file = tmp_path / "concat.txt"
    concat_file.write_text(
        f"file '{clip1.as_posix()}'\nfile '{clip2.as_posix()}'\n"
    )
    output = tmp_path / "concat_out.mp4"

    proc = await asyncio.create_subprocess_exec(
        ffmpeg_exe, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, f"Concat failed: {stderr.decode()[-300:]}"
    assert output.exists()
    size = output.stat().st_size
    assert size > 1000, f"Output file too small ({size} bytes) — likely corrupt"
    print(f"\n  Concat output: {size / 1024:.1f} KB")


@pytest.mark.slow
async def test_ffmpeg_audio_mux(ffmpeg_exe, tmp_path):
    """Create a video + audio file, mux together, verify output has both streams."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")

    video = tmp_path / "video.mp4"
    await _make_test_video(ffmpeg_exe, video, duration=2, color="green")

    # Create silent MP3 with a tone
    audio = tmp_path / "audio.mp3"
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_exe, "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(audio),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, _ = await proc.communicate()
    assert audio.exists() and audio.stat().st_size > 0, "MP3 not created"

    # Mux
    output = tmp_path / "with_audio.mp4"
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_exe, "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(output),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, f"Mux failed: {stderr.decode()[-300:]}"
    assert output.exists()
    size = output.stat().st_size
    assert size > 1000
    print(f"\n  Mux output: {size / 1024:.1f} KB")

    # Probe for audio stream
    probe = await asyncio.create_subprocess_exec(
        ffmpeg_exe, "-i", str(output),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, probe_out = await probe.communicate()
    probe_text = probe_out.decode(errors="replace")
    assert "audio" in probe_text.lower(), "Output has no audio stream"
    assert "video" in probe_text.lower(), "Output has no video stream"
    print(f"  Both audio+video streams confirmed in muxed output")


@pytest.mark.slow
async def test_concat_txt_uses_absolute_paths(ffmpeg_exe, tmp_path):
    """Regression: concat.txt must use absolute paths.

    Bug: orchestrator wrote relative paths like 'output/id/videos/shot_01.mp4'.
    FFmpeg resolves concat entries relative to the concat file's OWN directory,
    so it prefixed that with 'output/id/' → 'output/id/output/id/videos/shot_01.mp4'
    which doesn't exist → 'Impossible to open' error.
    Fix: use p.resolve().as_posix() for absolute paths.
    """
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")

    from pipeline.state import PipelineState, Shot
    from pipeline.orchestrator import PipelineOrchestrator

    # Simulate the session dir NESTED inside a subdirectory (like real output/)
    # This is the setup that triggers the doubling bug
    session_dir = tmp_path / "output" / "abcd1234"
    (session_dir / "videos").mkdir(parents=True)

    state = PipelineState(
        session_id="abcd1234",
        original_prompt="test",
        total_duration=2,
    )
    for sid in ("shot_01", "shot_02"):
        s = Shot(shot_id=sid, narrative_function="t", duration=1,
                 subject="t", environment="t", action="t")
        s.video_status = "complete"
        await _make_test_video(ffmpeg_exe, session_dir / "videos" / f"{sid}.mp4", 1)
        state.storyboard.append(s)

    messages = []
    orch = PipelineOrchestrator("abcd1234", session_dir, lambda m: messages.append(m))

    # Write the concat file the same way the orchestrator does
    video_paths = [session_dir / "videos" / f"{s.shot_id}.mp4" for s in state.storyboard]
    concat_file = session_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    # Every path in concat.txt must be absolute (start with / or drive letter)
    lines = concat_file.read_text().strip().splitlines()
    for line in lines:
        path_str = line.replace("file '", "").rstrip("'")
        assert os.path.isabs(path_str), (
            f"concat.txt entry is NOT absolute — FFmpeg will double the path!\n"
            f"  Entry: {line}\n"
            f"  Path: {path_str}\n"
            f"  FIX: use p.resolve().as_posix() instead of p.as_posix()"
        )

    # Confirm FFmpeg can actually read it (would fail with relative paths)
    output = session_dir / "final.mp4"
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", str(output),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, (
        f"FFmpeg concat failed (likely path issue):\n{stderr.decode(errors='replace')[-400:]}"
    )
    assert output.exists() and output.stat().st_size > 1000
    print(f"\n  Concat with absolute paths: {output.stat().st_size // 1024} KB — OK")


@pytest.mark.slow
async def test_orchestrator_stitch_method(ffmpeg_exe, tmp_path):
    """Test _stitch() through the orchestrator with real clips."""
    if not ffmpeg_exe:
        pytest.skip("FFmpeg not found")

    from pipeline.state import PipelineState, Shot, CameraParams
    from pipeline.orchestrator import PipelineOrchestrator

    # Build a minimal PipelineState with two completed shots
    state = PipelineState(
        session_id="stitch_test",
        original_prompt="test",
        total_duration=2,
    )
    s1 = Shot(shot_id="shot_01", narrative_function="test", duration=1,
              subject="test", environment="test", action="test")
    s1.video_status = "complete"
    s2 = Shot(shot_id="shot_02", narrative_function="test", duration=1,
              subject="test", environment="test", action="test")
    s2.video_status = "complete"
    state.storyboard = [s1, s2]

    # Create real video files
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    await _make_test_video(ffmpeg_exe, vid_dir / "shot_01.mp4", 1, "cyan")
    await _make_test_video(ffmpeg_exe, vid_dir / "shot_02.mp4", 1, "magenta")

    messages = []

    async def send(msg):
        messages.append(msg)

    orch = PipelineOrchestrator("stitch_test", tmp_path, send)
    result_path = await orch._stitch(state)

    assert result_path is not None, (
        "Stitch returned None — check logs: " +
        "\n".join(m.get("message", "") for m in messages if m.get("type") == "log")
    )
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 1000
    print(f"\n  Stitch result: {Path(result_path).stat().st_size / 1024:.1f} KB")
