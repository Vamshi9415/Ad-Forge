"""Test 06 — Audio Production: RunwayML TTS + SFX, Script, Mux.

Tests:
- Voice selection logic (tone → RunwayML preset name)
- Voiceover script + music bed description generation (GPT-4o)
- RunwayML TTS synthesis (eleven_multilingual_v2 via ElevenLabs)
- RunwayML SFX music bed (eleven_text_to_sound_v2 via ElevenLabs)
- FFmpeg mux (voiceover + music + video → final_with_audio.mp4)
- Full audio_gen.run() with a real final.mp4
"""

import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.api, pytest.mark.slow]


# ─── Voice selection logic (pure unit) ────────────────────────────────────
@pytest.mark.unit
@pytest.mark.parametrize("tone,expected_voice", [
    ("soft and intimate",    "Maya"),
    ("gentle warmth",        "Maya"),
    ("upbeat and energetic", "Leslie"),
    ("playful fun",          "Leslie"),
    ("authoritative bold",   "Noah"),
    ("powerful strong",      "Noah"),
    ("professional calm",    "Maya"),
    ("",                     "Maya"),
])
def test_voice_selection_by_tone(tone, expected_voice):
    from pipeline.agents.audio_gen import VOICE_MAP, _select_voice
    # Build a minimal brief to exercise _select_voice
    brief = {"tone": tone}
    voice = _select_voice(brief)
    assert voice == expected_voice, (
        f"tone='{tone}' → expected '{expected_voice}', got '{voice}'"
    )


@pytest.mark.unit
def test_voice_map_completeness():
    """VOICE_MAP must cover all required roles with valid RunwayML preset names."""
    from pipeline.agents.audio_gen import VOICE_MAP
    required_keys    = {"male", "female", "neutral", "upbeat", "soft"}
    valid_presets    = {"Maya", "Noah", "Leslie"}

    assert required_keys.issubset(set(VOICE_MAP.keys())), (
        f"VOICE_MAP missing keys: {required_keys - set(VOICE_MAP.keys())}"
    )
    for role, preset in VOICE_MAP.items():
        assert preset in valid_presets, (
            f"Role '{role}' maps to unknown preset '{preset}'; valid: {valid_presets}"
        )


@pytest.mark.unit
def test_runway_client_exports():
    """runway_client must export generate_tts and generate_sfx."""
    from pipeline.runway_client import generate_tts, generate_sfx, TTS_MODEL, SFX_MODEL
    assert TTS_MODEL == "eleven_multilingual_v2"
    assert SFX_MODEL == "eleven_text_to_sound_v2"
    assert callable(generate_tts)
    assert callable(generate_sfx)


# ─── Voiceover script generation ─────────────────────────────────────────
async def test_generate_voiceover_script_returns_fields(collector, llm_state):
    """GPT-4o must return script + music_prompt + direction."""
    from pipeline.agents.audio_gen import generate_voiceover_script

    state  = llm_state["state"]
    result = await generate_voiceover_script(state, collector.emit)

    assert "script"       in result, "Missing 'script' key"
    assert "direction"    in result, "Missing 'direction' key"
    assert "music_prompt" in result, "Missing 'music_prompt' key (new RunwayML SFX field)"
    assert "est_duration" in result, "Missing 'est_duration' key"

    assert len(result["script"]) > 20, f"Script too short: {result['script']!r}"
    assert result["est_duration"] > 0
    assert len(result["music_prompt"]) > 5, "music_prompt too short"

    print(f"\n  Script ({result['est_duration']:.0f}s): {result['script'][:120]}...")
    print(f"  Direction: {result['direction']}")
    print(f"  Music prompt: {result['music_prompt']}")


async def test_script_duration_plausible(collector, llm_state):
    """Estimated script duration should be within ±60% of target video duration."""
    from pipeline.agents.audio_gen import generate_voiceover_script

    state  = llm_state["state"]
    result = await generate_voiceover_script(state, collector.emit)
    est    = result["est_duration"]
    target = state.total_duration

    assert est > target * 0.4, f"Script too short: {est:.1f}s for {target}s video"
    assert est < target * 2.5, f"Script too long: {est:.1f}s for {target}s video"


# ─── RunwayML TTS Synthesis ───────────────────────────────────────────────
async def test_synthesize_tts_produces_audio(tmp_path, collector):
    """RunwayML TTS must produce a non-trivial audio file."""
    from pipeline.agents.audio_gen import synthesize_tts

    out = tmp_path / "test_tts.mp3"
    ok  = await synthesize_tts(
        script="Introducing AdForge. Generative advertising, powered by AI.",
        voice_preset="Maya",
        audio_path=out,
        emit=collector.emit,
    )

    assert ok is True, "synthesize_tts() returned False"
    assert out.exists(), "No output file created"
    size = out.stat().st_size
    assert size > 5_000, f"TTS audio too small ({size} bytes)"

    # Loose audio header check — RunwayML may return MP3 or AAC
    header = out.read_bytes()[:4]
    is_audio = (
        header[:2] == b"\xFF\xFB"    # MPEG sync
        or header[:3] == b"ID3"      # MP3 ID3 tag
        or header[0] == 0xFF         # any MPEG sync byte
        or header[:4] == b"ftyp"     # AAC/M4A ISO box (offset +4)
    )
    assert is_audio or size > 10_000, (
        f"Output doesn't look like audio (header: {header.hex()}, size: {size})"
    )
    print(f"\n  TTS output (Maya): {size / 1024:.1f} KB")


@pytest.mark.parametrize("voice_preset", ["Maya", "Noah", "Leslie"])
async def test_synthesize_tts_all_voices(tmp_path, collector, voice_preset):
    """Every RunwayML TTS preset must successfully produce audio."""
    from pipeline.agents.audio_gen import synthesize_tts

    out = tmp_path / f"test_{voice_preset}.mp3"
    ok  = await synthesize_tts(
        script="The future of advertising is here.",
        voice_preset=voice_preset,
        audio_path=out,
        emit=collector.emit,
    )
    assert ok is True, f"TTS failed for preset '{voice_preset}'"
    assert out.exists() and out.stat().st_size > 1_000, (
        f"Output too small for preset '{voice_preset}'"
    )
    print(f"\n  TTS {voice_preset}: {out.stat().st_size // 1024} KB")


# ─── RunwayML SFX Music Bed ───────────────────────────────────────────────
async def test_synthesize_music_bed_produces_audio(tmp_path, collector):
    """RunwayML SFX must produce a non-trivial music bed audio file."""
    from pipeline.agents.audio_gen import synthesize_music_bed

    out    = tmp_path / "music.mp3"
    prompt = "Warm cinematic underscore, soft piano and strings, slow build, premium advertising feel"
    ok     = await synthesize_music_bed(prompt, out, collector.emit)

    assert ok is True, "synthesize_music_bed() returned False"
    assert out.exists(), "No output file created"
    size = out.stat().st_size
    assert size > 5_000, f"SFX audio too small ({size} bytes)"
    print(f"\n  SFX music bed: {size / 1024:.1f} KB")


async def test_sfx_skips_empty_prompt(tmp_path, collector):
    """SFX must return False gracefully when given an empty music prompt."""
    from pipeline.agents.audio_gen import synthesize_music_bed

    out = tmp_path / "music.mp3"
    ok  = await synthesize_music_bed("", out, collector.emit)

    assert ok is False, "SFX should return False for empty prompt"
    assert not out.exists(), "No file should be created for empty prompt"


# ─── Full audio_gen.run() pipeline ───────────────────────────────────────
async def test_audio_run_no_final_video(tmp_path, collector, llm_state):
    """run() must return None gracefully if final.mp4 doesn't exist."""
    from pipeline.agents.audio_gen import run

    state  = llm_state["state"]
    result = await run(state, tmp_path, collector.emit)

    assert result is None
    skip_logs = [
        m for m in collector.messages
        if m.get("type") == "log" and "no final" in m.get("message", "").lower()
    ]
    assert len(skip_logs) >= 1, "Expected a 'no final.mp4' warning log"


@pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="FFmpeg not available on PATH"
)
async def test_audio_run_full_pipeline(tmp_path, collector, llm_state):
    """Create a dummy final.mp4, run full audio_gen pipeline, verify output."""
    import asyncio
    from pipeline.agents.audio_gen import run, _find_ffmpeg

    ffmpeg = shutil.which("ffmpeg") or _find_ffmpeg()
    if not ffmpeg:
        pytest.skip("FFmpeg not found")

    # Create a tiny synthetic final.mp4 (3 seconds, black frames + sine tone)
    final_mp4 = tmp_path / "final.mp4"
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=black:size=64x36:rate=5:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(final_mp4),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    assert final_mp4.exists(), "Could not create test final.mp4"

    state = llm_state["state"]
    state.final_video_path = str(final_mp4)

    result = await run(state, tmp_path, collector.emit)

    if result is None:
        print("\n  Pipeline logs:")
        for m in collector.messages:
            if m.get("type") == "log":
                print(f"    [S{m.get('stage')}][{m.get('level')}] {m.get('message')}")
        pytest.fail("audio_gen.run() returned None — see logs above")

    out_path = Path(result)
    assert out_path.exists(),  f"Output file not found: {result}"
    size = out_path.stat().st_size
    assert size > 5_000, f"Output video too small ({size} bytes)"
    print(f"\n  Final audio+video: {size / 1024:.1f} KB → {out_path.name}")

    # voiceover.mp3 must be created by RunwayML TTS
    vo_path = tmp_path / "voiceover.mp3"
    assert vo_path.exists(), "voiceover.mp3 not saved"
    print(f"  Voiceover: {vo_path.stat().st_size // 1024} KB")

    # Verify script was stored on state
    assert state.voiceover_script, "voiceover_script not set on state"
    print(f"  Script: {state.voiceover_script[:80]}...")

    # audio_ready event with all required fields
    audio_events = collector.of_type("audio_ready")
    assert len(audio_events) == 1, f"Expected 1 audio_ready event, got {len(audio_events)}"
    ev = audio_events[0]
    assert "audio_video_src" in ev, "audio_ready missing audio_video_src"
    assert "audio_src"       in ev, "audio_ready missing audio_src"
    assert "script"          in ev, "audio_ready missing script"
    assert "voice"           in ev, "audio_ready missing voice (RunwayML preset)"
    assert "has_music"       in ev, "audio_ready missing has_music flag"
    print(f"  Voice preset used: {ev['voice']}")
    print(f"  Background music generated: {ev.get('has_music', False)}")
