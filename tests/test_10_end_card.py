"""Unit tests for promo end card helpers."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ffmpeg_exe():
    from pipeline.orchestrator import _find_ffmpeg
    return shutil.which("ffmpeg") or _find_ffmpeg()


def test_parse_ratio_defaults():
    from pipeline.end_card import parse_ratio

    assert parse_ratio("1280:720") == (1280, 720)
    assert parse_ratio("720:1280") == (720, 1280)
    assert parse_ratio("") == (1280, 720)


def test_brief_lines_fallback():
    from pipeline.end_card import brief_lines
    from pipeline.state import PipelineState

    state = PipelineState(session_id="x", original_prompt="hi")
    state.enriched_brief = {"campaign_title": "Acme", "primary_cta": "Buy now"}
    assert brief_lines(state) == ("Acme", "Buy now")

    state.enriched_brief = {}
    t, c = brief_lines(state)
    assert t == "Your Brand"
    assert c == "Shop now"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_try_build_and_append_end_card_smoke(tmp_path, ffmpeg_exe):
    """Synthetic clip + keyframe → promo MP4 appended (integration-sized smoke)."""
    from pipeline.end_card import (
        BODY_PROMO_CROSSFADE_SEC,
        END_CARD_CLIP_SEC,
        try_build_and_append_end_card,
    )
    from pipeline.state import PipelineState, Shot

    session_dir = tmp_path / "sess"
    vdir = session_dir / "videos"
    vdir.mkdir(parents=True)

    clip = vdir / "shot_01.mp4"
    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:size=320x180:rate=25:duration=2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", "2",
            str(clip),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    shot = Shot(
        shot_id="shot_01",
        narrative_function="CTA",
        duration=2,
        subject="x",
        environment="x",
        action="x",
    )
    shot.video_status = "complete"

    state = PipelineState(
        session_id="sess",
        original_prompt="test",
        total_duration=2,
        ratio="320:180",
    )
    state.storyboard = [shot]
    state.enriched_brief = {"campaign_title": "SmokeCo", "primary_cta": "Try it"}

    body = session_dir / "final_body.mp4"
    shutil.copy2(clip, body)
    final = session_dir / "final.mp4"

    appended, warn, pad_sec = try_build_and_append_end_card(
        ffmpeg_exe,
        session_dir,
        state,
        body,
        final,
        [clip],
    )

    assert appended, f"append failed: {warn}"
    assert pad_sec >= 1
    assert final.is_file()
    assert final.stat().st_size > clip.stat().st_size * 0.5

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(final),
            ],
            capture_output=True,
            timeout=15,
            text=True,
        )
        assert probe.returncode == 0
        dur = float(probe.stdout.strip())
        # ~body + promo − crossfade (overlap subtracted vs hard-cut).
        assert dur >= 2.0 + END_CARD_CLIP_SEC - BODY_PROMO_CROSSFADE_SEC - 0.35
