"""Stage 9: Audio Production — RunwayML TTS voiceover + SFX music bed → mux into final video.

Pipeline:
  1. GPT-4o writes a timed voiceover script + a music bed description
     AND selects the best voice for the story/audience
  2. RunwayML TTS (eleven_multilingual_v2) renders the narration → voiceover.mp3
     Supports 29 languages: en, ja, zh, de, hi, fr, ko, pt, it, es, id, nl, tr,
     fil, pl, sv, bg, ro, ar, cs, el, fi, hr, ms, sk, da, ta, uk, ru
  3. RunwayML SFX (eleven_text_to_sound_v2) generates a background music bed → music.mp3
  4. FFmpeg mixes both (music at 30% volume, narration at 100%, music looped to full duration)
     → mixed_audio.aac
  5. FFmpeg muxes mixed audio + final.mp4 → final_with_audio.mp4
  6. (Optional) RunwayML Dubbing (eleven_voice_dubbing) dubs narration into additional
     languages → voiceover_{lang}.mp3 for each target language

Multilingual models:
  eleven_multilingual_v2     — TTS: text in any of 29 languages → speech
  eleven_multilingual_sts_v2 — STS: voice conversion preserving original language
  eleven_voice_dubbing       — Dubbing: translates + re-voices to target language

RunwayML TTS voices:
  Maya   — warm, nurturing female; luxury, beauty, wellness, emotional storytelling
  Noah   — deep, authoritative male; tech, finance, B2B, sports, problem-solving
  Leslie — clear, neutral, versatile; youth brands, SaaS, Gen-Z, high-energy, gender-neutral
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI

from pipeline.state import PipelineState
from pipeline.runway_client import (
    generate_tts,
    generate_sfx,
    generate_sts,
    generate_dubbing,
    download_bytes,
    SUPPORTED_LANGUAGES,
)


def _get_openai():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── RunwayML TTS voice presets ────────────────────────────────────────────
VOICE_MAP = {
    "male":    "Noah",   # deep, authoritative
    "female":  "Maya",   # warm, clear, multilingual (default)
    "neutral": "Leslie", # neutral, versatile
    "soft":    "Maya",   # warm and intimate
    "upbeat":  "Leslie", # energetic, clear
}

# ── Prompts ───────────────────────────────────────────────────────────────
SCRIPT_SYSTEM = """You are an award-winning advertising copywriter and voice casting director.
Write a tight, emotionally resonant voiceover script AND cast the best voice for this specific story.

━━ THREE AVAILABLE VOICES ━━
• Maya   — warm, nurturing female voice. Best for: beauty, wellness, luxury, skincare, food,
           emotional/aspirational storytelling, female-skewing audiences, slow-burn narratives.
• Noah   — deep, authoritative male voice. Best for: tech, finance, automotive, sports, B2B,
           male-skewing audiences, problem-solution narratives, bold/confident tone.
• Leslie — clear, neutral, versatile voice. Best for: youth brands, SaaS, Gen-Z products,
           gender-neutral campaigns, fast-paced energetic ads, lifestyle/fitness, TikTok style.

━━ CASTING RULES ━━
1. Match voice to PRIMARY audience gender/age skew when clearly defined.
2. Match voice to product category (luxury → Maya; hard-tech → Noah; startup/youth → Leslie).
3. Match voice to narrative arc (emotional → Maya; authoritative → Noah; punchy/energetic → Leslie).
4. When in doubt about audience → pick based on emotional_tone and visual_style.
5. Never default-cast Maya just because no other signal exists — consider all signals.

━━ SCRIPT RULES ━━
- Match the exact total duration (1 word ≈ 0.40 seconds at a measured pace; " ... " = 0.7s pause)
- Use vivid, sensory language — not corporate jargon
- Structure: hook (15%) → problem/tension (25%) → product solution (35%) → CTA (25%)
- Leave natural pauses between beats using " ... " in the script
- Feel premium, cinematic, and human — not robotic

━━ MUSIC BED RULES ━━
The music_bed_prompt drives an AI music/sound generator. Keep it under 400 characters.
Include: 3 instruments + BPM + genre/era + one dynamic arc sentence + 2 mood words + production feel.
Example: "Fingerpicked acoustic guitar, upright bass, warm Rhodes piano. 72 bpm, languid swing.
70s Bombay jazz nostalgia. Sparse opening, gentle swell at midpoint, quiet resolution on CTA.
Bittersweet, sun-warmed. Analog warmth with vinyl crackle."

━━ AMBIENT SFX RULES ━━
Generate 2-3 ambient atmosphere prompts — one per key scene environment in the story.
Each is a SHORT description of real-world sound (NOT music), max 350 characters each.
These layer UNDER the music and voiceover as atmospheric texture.
Examples:
  • "Busy Mumbai street market: vendor calls, rickshaws passing, distant crowd murmur, light breeze"
  • "Cozy café interior: soft clinking cups, coffee machine hiss, low conversation, rain on windows"
  • "Quiet kitchen: gentle sizzling pan, water tap, ceramic bowl placed on wooden table"

Return ONLY valid JSON:
{
  "voiceover_script": "<full script text with ... for pauses>",
  "voice": "Maya" or "Noah" or "Leslie",
  "voice_rationale": "<one sentence: why this voice fits THIS campaign>",
  "voice_direction": "<1-sentence tone note, e.g. 'warm and intimate, hushed confidence'>",
  "music_bed_prompt": "<under 400 chars: instruments + BPM + genre + arc + mood + production feel>",
  "ambient_sfx_prompts": ["<scene 1 atmosphere, max 350 chars>", "<scene 2 atmosphere, max 350 chars>"],
  "estimated_duration_seconds": <float>
}"""


# ── Script generation ─────────────────────────────────────────────────────
VALID_VOICES = {"Maya", "Noah", "Leslie"}


async def generate_voiceover_script(state: PipelineState, emit: Callable) -> dict:
    """Ask GPT-4o to write a timed voiceover script, cast the voice, and describe the music bed."""
    brief     = state.enriched_brief
    total_dur = state.total_duration

    # Detailed shot breakdown for music arc mapping
    shots_desc = "\n".join(
        f"  Shot {i+1} ({s.duration}s): {s.narrative_function} — {s.action}"
        for i, s in enumerate(state.storyboard)
    )

    # Key scenes from the creative brief (enricher generates these)
    key_scenes = brief.get("key_scenes", [])
    key_scenes_str = "\n".join(f"  • {sc}" for sc in key_scenes[:5]) if key_scenes else "N/A"

    await emit({"type": "log", "stage": "9",
                "message": "Generating voiceover script, casting voice, and music brief with GPT-4o...",
                "level": "info"})

    resp = await _get_openai().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SCRIPT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Campaign: {brief.get('campaign_title', 'Ad Campaign')}\n"
                    f"Product/Brand: {brief.get('product_name', state.original_prompt[:80])}\n"
                    f"Tagline: {brief.get('tagline', '')}\n"
                    f"Emotional tone: {brief.get('tone', brief.get('emotional_tone', 'professional'))}\n"
                    f"Visual style: {brief.get('visual_style', '')}\n"
                    f"Pacing: {brief.get('pacing', '')}\n"
                    f"Narrative arc: {brief.get('narrative_arc', '')}\n"
                    f"Creative direction: {brief.get('creative_direction', '')}\n"
                    f"Target audience / demographics: "
                    f"{brief.get('target_audience', brief.get('demographic_insights', 'general'))}\n"
                    f"Platform: {state.platform}\n"
                    f"CTA: {brief.get('primary_cta', brief.get('cta', 'Learn more'))}\n"
                    f"Unique selling point: {brief.get('unique_selling_point', '')}\n"
                    f"Suggested music mood (from creative director): {brief.get('music_mood', 'not specified')}\n"
                    f"Color grade / visual mood: {brief.get('color_grade', '')}\n"
                    f"Transition style: {brief.get('transition_style', '')}\n\n"
                    f"Total duration: {total_dur}s\n"
                    f"Shot-by-shot story breakdown (map music dynamics to these beats):\n{shots_desc}\n\n"
                    f"Key scenes:\n{key_scenes_str}\n\n"
                    f"Script language: {brief.get('script_language', 'en').upper()} "
                    f"(write the voiceover script in this language; eleven_multilingual_v2 supports it natively)\n"
                    f"Cast the best voice. Write a voiceover script that fills ~{total_dur}s naturally.\n"
                    f"For music_bed_prompt: write a rich 50-70 word brief that maps the music's dynamic "
                    f"arc to the story beats above (e.g. sparse opening → swell at midpoint → resolution)."
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
    )

    result          = json.loads(resp.choices[0].message.content)
    script          = result.get("voiceover_script", "")
    direction       = result.get("voice_direction", "warm, confident")
    music_prompt    = result.get("music_bed_prompt", "")
    ambient_prompts = result.get("ambient_sfx_prompts", [])
    est_dur         = float(result.get("estimated_duration_seconds", total_dur))
    gpt_voice       = result.get("voice", "")
    voice_rationale = result.get("voice_rationale", "")

    # Validate GPT-4o's pick — fall back to heuristic if hallucinated
    picked_voice = gpt_voice if gpt_voice in VALID_VOICES else None

    # Ensure ambient_prompts is a list of non-empty strings
    if not isinstance(ambient_prompts, list):
        ambient_prompts = []
    ambient_prompts = [str(p).strip() for p in ambient_prompts if str(p).strip()][:3]

    print(f"[audio_gen] GPT-4o cast voice={gpt_voice!r} rationale={voice_rationale!r}")
    print(f"[audio_gen] script: {len(script)} chars, ~{est_dur:.0f}s | direction: {direction}")
    print(f"[audio_gen] music prompt: {music_prompt}")
    print(f"[audio_gen] ambient SFX prompts: {len(ambient_prompts)}")

    await emit({"type": "log", "stage": "9",
                "message": (
                    f"Script ready (~{est_dur:.0f}s) | voice: {picked_voice or 'heuristic'} "
                    f"— {voice_rationale or direction}"
                ),
                "level": "info"})
    if music_prompt:
        await emit({"type": "log", "stage": "9",
                    "message": f"Music brief: {music_prompt}",
                    "level": "info"})
    if ambient_prompts:
        await emit({"type": "log", "stage": "9",
                    "message": f"Ambient SFX: {len(ambient_prompts)} atmosphere tracks planned",
                    "level": "info"})

    return {
        "script":           script,
        "direction":        direction,
        "music_prompt":     music_prompt,
        "ambient_prompts":  ambient_prompts,
        "est_duration":     est_dur,
        "gpt_voice":        picked_voice,   # None if GPT-4o returned invalid/empty
    }


# ── RunwayML TTS ──────────────────────────────────────────────────────────
async def synthesize_tts(
    script: str,
    voice_preset: str,
    audio_path: Path,
    emit: Callable,
) -> bool:
    """Render voiceover narration via RunwayML TTS (ElevenLabs eleven_multilingual_v2).
    Returns True on success.
    """
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        await emit({"type": "log", "stage": "9",
                    "message": "RUNWAYML_API_SECRET not set — cannot use RunwayML TTS",
                    "level": "error"})
        return False

    # Approximate credit cost: 1 credit per 50 chars
    est_credits = max(1, len(script) // 50)
    await emit({"type": "log", "stage": "9",
                "message": f"RunwayML TTS: voice={voice_preset} | ~{len(script)} chars (~{est_credits} credits)",
                "level": "info"})
    print(f"[audio_gen] RunwayML TTS: voice={voice_preset}, chars={len(script)}")

    try:
        async def tts_log(msg, lvl="info"):
            await emit({"type": "log", "stage": "9", "message": f"[TTS] {msg}", "level": lvl})

        audio_url = await generate_tts(
            api_key=api_key,
            text=script,
            voice_preset=voice_preset,
            log=tts_log,
        )
        audio_bytes = await download_bytes(audio_url)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio_bytes)

        size_kb = audio_path.stat().st_size // 1024
        print(f"[audio_gen] TTS complete: {size_kb} KB -> {audio_path}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Voiceover rendered: {size_kb} KB (RunwayML TTS / ElevenLabs)",
                    "level": "success"})
        return True

    except Exception as exc:
        print(f"[audio_gen] TTS error: {exc}")
        await emit({"type": "log", "stage": "9",
                    "message": f"RunwayML TTS error: {str(exc)[:160]}",
                    "level": "error"})
        return False


# ── RunwayML SFX (Music Bed) ──────────────────────────────────────────────
async def synthesize_music_bed(
    music_prompt: str,
    sfx_path: Path,
    emit: Callable,
) -> bool:
    """Generate a background music bed via RunwayML SFX (ElevenLabs eleven_text_to_sound_v2).
    Returns True on success.
    """
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key:
        return False

    if not music_prompt:
        await emit({"type": "log", "stage": "9",
                    "message": "No music prompt — skipping background music generation.",
                    "level": "warning"})
        return False

    await emit({"type": "log", "stage": "9",
                "message": f"RunwayML SFX: generating music bed — \"{music_prompt[:70]}\"",
                "level": "info"})
    print(f"[audio_gen] RunwayML SFX: {music_prompt}")

    try:
        async def sfx_log(msg, lvl="info"):
            await emit({"type": "log", "stage": "9", "message": f"[SFX] {msg}", "level": lvl})

        sfx_url = await generate_sfx(
            api_key=api_key,
            prompt_text=music_prompt,
            log=sfx_log,
        )
        sfx_bytes = await download_bytes(sfx_url)
        sfx_path.parent.mkdir(parents=True, exist_ok=True)
        sfx_path.write_bytes(sfx_bytes)

        size_kb = sfx_path.stat().st_size // 1024
        print(f"[audio_gen] SFX complete: {size_kb} KB -> {sfx_path}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Music bed generated: {size_kb} KB (RunwayML SFX / ElevenLabs)",
                    "level": "success"})
        return True

    except Exception as exc:
        print(f"[audio_gen] SFX error: {exc}")
        await emit({"type": "log", "stage": "9",
                    "message": f"RunwayML SFX error (music skipped): {str(exc)[:160]}",
                    "level": "warning"})
        return False


# ── RunwayML SFX — multiple ambient tracks ───────────────────────────────
async def synthesize_ambient_sfx(
    ambient_prompts: list,
    session_dir: Path,
    emit: Callable,
) -> list:
    """Generate multiple ambient atmosphere SFX tracks in parallel.
    Returns list of Path objects for successfully generated files.
    """
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key or not ambient_prompts:
        return []

    await emit({"type": "log", "stage": "9",
                "message": f"Generating {len(ambient_prompts)} ambient atmosphere SFX tracks...",
                "level": "info"})

    async def gen_one(idx: int, prompt: str) -> Optional[Path]:
        sfx_path = session_dir / f"ambient_{idx+1}.mp3"
        try:
            async def sfx_log(msg, lvl="info"):
                await emit({"type": "log", "stage": "9",
                            "message": f"[SFX-atm{idx+1}] {msg}", "level": lvl})

            sfx_url = await generate_sfx(
                api_key=api_key,
                prompt_text=prompt,   # runway_client truncates to 450 chars
                log=sfx_log,
            )
            sfx_bytes = await download_bytes(sfx_url)
            sfx_path.parent.mkdir(parents=True, exist_ok=True)
            sfx_path.write_bytes(sfx_bytes)
            size_kb = sfx_path.stat().st_size // 1024
            await emit({"type": "log", "stage": "9",
                        "message": f"Ambient SFX {idx+1}: {size_kb} KB — {prompt[:60]}...",
                        "level": "success"})
            return sfx_path
        except Exception as exc:
            await emit({"type": "log", "stage": "9",
                        "message": f"Ambient SFX {idx+1} failed (skipped): {str(exc)[:120]}",
                        "level": "warning"})
            return None

    results = await asyncio.gather(*[gen_one(i, p) for i, p in enumerate(ambient_prompts)])
    paths = [p for p in results if p is not None and p.exists()]
    await emit({"type": "log", "stage": "9",
                "message": f"Ambient SFX: {len(paths)}/{len(ambient_prompts)} tracks ready",
                "level": "success" if paths else "warning"})
    return paths


# ── Multilingual dubbing ──────────────────────────────────────────────────
async def synthesize_multilingual_dubs(
    voiceover_url: str,
    target_languages: list,
    session_dir: Path,
    emit: Callable,
) -> dict:
    """Dub the voiceover into multiple languages in parallel.

    Uses eleven_voice_dubbing to translate and re-voice the narration while
    preserving the speaker's characteristics.

    Returns {lang_code: Path} for successfully dubbed files.
    """
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    if not api_key or not target_languages:
        return {}

    valid_langs = [l for l in target_languages if l in SUPPORTED_LANGUAGES]
    if not valid_langs:
        await emit({"type": "log", "stage": "9",
                    "message": f"No valid target languages for dubbing (got: {target_languages})",
                    "level": "warning"})
        return {}

    await emit({"type": "log", "stage": "9",
                "message": f"Dubbing voiceover into {len(valid_langs)} languages: {', '.join(valid_langs)}",
                "level": "info"})

    async def dub_one(lang: str) -> tuple:
        dub_path = session_dir / f"voiceover_{lang}.mp3"
        try:
            async def dub_log(msg, lvl="info"):
                await emit({"type": "log", "stage": "9",
                            "message": f"[Dub-{lang}] {msg}", "level": lvl})

            dub_url = await generate_dubbing(
                api_key=api_key,
                audio_url=voiceover_url,
                target_language=lang,
                log=dub_log,
            )
            dub_bytes = await download_bytes(dub_url)
            dub_path.parent.mkdir(parents=True, exist_ok=True)
            dub_path.write_bytes(dub_bytes)
            size_kb = dub_path.stat().st_size // 1024
            await emit({"type": "log", "stage": "9",
                        "message": f"Dubbed [{lang}]: {size_kb} KB",
                        "level": "success"})
            return (lang, dub_path)
        except Exception as exc:
            await emit({"type": "log", "stage": "9",
                        "message": f"Dubbing [{lang}] failed: {str(exc)[:120]}",
                        "level": "warning"})
            return (lang, None)

    results = await asyncio.gather(*[dub_one(lang) for lang in valid_langs])
    return {lang: path for lang, path in results if path is not None}


# ── FFmpeg helpers ────────────────────────────────────────────────────────
def _find_ffmpeg() -> Optional[str]:
    import shutil, glob
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.EXE"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg*\bin\ffmpeg.exe"),
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expandvars(r"%USERPROFILE%\scoop\shims\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        if os.path.isfile(pattern):
            return pattern
    return None


async def _encode_vo_to_aac(
    ffmpeg: str, vo_path: Path, out_path: Path, pad_seconds: int = 0,
) -> bool:
    """Re-encode voiceover MP3 → AAC so it can be muxed cleanly into MP4.

    Optional silence tail (e.g. promo end card) via apad."""
    def _run():
        args = [ffmpeg, "-y", "-i", str(vo_path)]
        if pad_seconds > 0:
            args += ["-af", f"apad=pad_dur={pad_seconds}"]
        args += ["-c:a", "aac", "-b:a", "192k", str(out_path)]
        return subprocess.run(args, capture_output=True, timeout=90)
    try:
        r = await asyncio.to_thread(_run)
        return r.returncode == 0
    except Exception as e:
        print(f"[audio_gen] aac encode error: {e}")
        return False


async def _mix_audio_tracks(
    ffmpeg: str,
    vo_path: Path,
    music_path: Optional[Path],
    mixed_path: Path,
    total_duration: int,
    emit: Callable,
    ambient_paths: Optional[list] = None,
    playback_padding_seconds: int = 0,
) -> bool:
    """Mix voiceover + optional music bed + optional ambient SFX tracks into one AAC file.

    Levels:
      Voiceover:    100% (always loudest — the ad script)
      Music bed:    28%  (audible background, ducked under VO)
      Each ambient: 12%  (atmospheric texture, barely perceptible)

    Music and ambient SFX are looped so they fill any video length.
    Returns True on success.
    """
    ambient_paths = [p for p in (ambient_paths or []) if p and Path(p).exists()]
    has_music     = bool(music_path and music_path.exists())
    has_ambient   = bool(ambient_paths)

    if not has_music and not has_ambient:
        # No supplemental audio — just re-encode voiceover MP3 → AAC
        await emit({"type": "log", "stage": "9",
                    "message": "No music/SFX tracks — encoding voiceover to AAC.",
                    "level": "info"})
        ok = await _encode_vo_to_aac(
            ffmpeg, vo_path, mixed_path, pad_seconds=max(0, playback_padding_seconds),
        )
        if not ok:
            # Last resort: copy file (may not play in all players but better than nothing)
            import shutil as _sh
            _sh.copy2(vo_path, mixed_path)
        return True

    n_ambient  = len(ambient_paths)
    n_inputs   = 1 + (1 if has_music else 0) + n_ambient   # vo + music? + ambients
    fade_start = max(0, total_duration - 3)

    layer_desc = []
    if has_music:
        layer_desc.append("music 28%")
    if has_ambient:
        layer_desc.append(f"{n_ambient} ambient SFX 12% each")
    await emit({"type": "log", "stage": "9",
                "message": f"Mixing audio: voiceover 100% + {', '.join(layer_desc)} → {total_duration}s AAC",
                "level": "info"})

    # Build -i arguments and filter_complex
    cmd = [ffmpeg, "-y", "-i", str(vo_path)]   # input 0 = voiceover

    filter_parts = []
    mix_inputs   = ["[0:a]"]   # voiceover always present
    input_idx    = 1

    if has_music:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]
        filter_parts.append(
            f"[{input_idx}:a]atrim=duration={total_duration},"
            f"volume=0.28,"
            f"afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={fade_start}:d=3[bg]"
        )
        mix_inputs.append("[bg]")
        input_idx += 1

    for ai, ap in enumerate(ambient_paths):
        cmd += ["-stream_loop", "-1", "-i", str(ap)]
        tag = f"atm{ai}"
        filter_parts.append(
            f"[{input_idx}:a]atrim=duration={total_duration},"
            f"volume=0.12,"
            f"afade=t=in:st=0:d=2.0,"
            f"afade=t=out:st={fade_start}:d=3[{tag}]"
        )
        mix_inputs.append(f"[{tag}]")
        input_idx += 1

    mix_str = "".join(mix_inputs)
    filter_parts.append(
        f"{mix_str}amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=2[aout]"
    )

    af = ";".join(filter_parts)
    cmd += [
        "-filter_complex", af,
        "-map", "[aout]",
        "-t", str(total_duration + 2),
        "-c:a", "aac", "-b:a", "192k",
        str(mixed_path),
    ]

    def _run_mix() -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, timeout=180)

    try:
        result = await asyncio.to_thread(_run_mix)
        rc     = result.returncode
        stderr = result.stderr
    except Exception as e:
        print(f"[audio_gen] audio mix subprocess error: {type(e).__name__}: {e}")
        rc, stderr = 1, str(e).encode()

    if rc != 0:
        err = stderr.decode(errors="replace")[-400:] if isinstance(stderr, bytes) else str(stderr)
        print(f"[audio_gen] audio mix error: {err}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Audio mix failed — falling back to voiceover-only AAC",
                    "level": "warning"})
        await _encode_vo_to_aac(
            ffmpeg, vo_path, mixed_path, pad_seconds=max(0, playback_padding_seconds),
        )

    return True


async def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    total_duration: int,
    emit: Callable,
) -> Optional[str]:
    """Mux a pre-mixed audio track into a video file. Returns output path on success.

    The audio fed here is already encoded to AAC by _mix_audio_tracks (fade-out
    already applied). We therefore:
      - Copy the video stream bit-for-bit (-c:v copy) to avoid re-encoding
      - Copy the audio stream as-is (-c:a copy) — no second fade, no re-encode
      - Use -shortest to stop at the shorter of video / audio
      - Add faststart for web playback
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        await emit({"type": "log", "stage": "9",
                    "message": "FFmpeg not found — cannot mux audio into video.",
                    "level": "error"})
        return None

    await emit({"type": "log", "stage": "9",
                "message": "Muxing audio into final video with FFmpeg...",
                "level": "info"})
    print(f"[audio_gen] muxing: {video_path.name} + {audio_path.name} -> {output_path.name}")

    def _run_mux() -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                # audio is already AAC from _mix_audio_tracks — copy to avoid
                # re-encoding and the previous double-fade bug
                "-c:a", "copy",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ],
            capture_output=True,
            timeout=180,
        )

    try:
        result = await asyncio.to_thread(_run_mux)
        rc = result.returncode
        stderr = result.stderr
    except Exception as e:
        print(f"[audio_gen] mux subprocess error: {type(e).__name__}: {e}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Mux subprocess error: {type(e).__name__}: {e}",
                    "level": "error"})
        return None

    if rc != 0:
        err = stderr.decode(errors="replace")[-400:]
        print(f"[audio_gen] mux error: {err}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Mux error: {err[-200:]}", "level": "error"})
        return None

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[audio_gen] mux done: {output_path} ({size_mb:.1f} MB)")
    await emit({"type": "log", "stage": "9",
                "message": f"Final video with audio: {total_duration}s, {size_mb:.1f} MB",
                "level": "success"})
    return str(output_path)


# ── Voice selection ───────────────────────────────────────────────────────
def _score_voices(brief: dict, platform: str = "") -> dict:
    """Return a {voice: score} map driven by all available brief signals.

    Scoring covers:
      • Target audience gender/age skew
      • Product category
      • Emotional tone & visual style
      • Narrative arc
      • Platform (TikTok/Reels skews Leslie; LinkedIn skews Noah)
      • Music mood / pacing

    Higher score → better fit. Ties broken by this priority: Leslie > Noah > Maya
    (to reduce the historical over-use of Maya as default).
    """
    scores = {"Maya": 0, "Noah": 0, "Leslie": 0}

    def text(*keys) -> str:
        parts = [str(brief.get(k, "") or "") for k in keys]
        return " ".join(parts).lower()

    # ── audience / demographic signals ──────────────────────────────────
    audience = text("target_audience", "demographic_insights")
    if any(w in audience for w in ["women", "female", "mothers", "ladies", "girl", "her"]):
        scores["Maya"] += 3
    if any(w in audience for w in ["men", "male", "fathers", "guys", "his", "gentleman"]):
        scores["Noah"] += 3
    if any(w in audience for w in ["gen-z", "gen z", "youth", "teen", "student", "millennial"]):
        scores["Leslie"] += 3
    if any(w in audience for w in ["professional", "executive", "b2b", "enterprise", "corporate"]):
        scores["Noah"] += 2
    if any(w in audience for w in ["parent", "family", "household"]):
        scores["Maya"] += 1

    # ── product category ────────────────────────────────────────────────
    product = text("product_name", "campaign_title", "unique_selling_point")
    maya_products = [
        "beauty", "skincare", "cosmetic", "fragrance", "perfume",
        "wellness", "yoga", "luxury", "jewel", "fashion", "organic",
        "baby", "food", "recipe", "home", "decor",
    ]
    noah_products = [
        "tech", "software", "app", "saas", "finance", "bank", "invest",
        "automotive", "car", "truck", "suv", "security", "insurance",
        "sports", "gym", "fitness", "protein", "supplement",
        "gaming", "hardware", "electric", "energy",
    ]
    leslie_products = [
        "startup", "mobile", "social", "streaming", "subscription",
        "travel", "adventure", "sneaker", "streetwear", "music", "podcast",
        "e-commerce", "delivery", "marketplace", "ai", "crypto",
    ]
    for kw in maya_products:
        if kw in product:
            scores["Maya"] += 2
    for kw in noah_products:
        if kw in product:
            scores["Noah"] += 2
    for kw in leslie_products:
        if kw in product:
            scores["Leslie"] += 2

    # ── emotional tone ───────────────────────────────────────────────────
    tone = text("emotional_tone", "tone", "creative_direction")
    maya_tones  = ["warm", "intimate", "nurturing", "gentle", "soft", "emotional", "aspirational", "romantic", "sensual", "luxury"]
    noah_tones  = ["authoritative", "bold", "powerful", "strong", "serious", "confident", "premium", "epic", "cinematic", "commanding"]
    leslie_tones = ["upbeat", "energetic", "playful", "fun", "fresh", "vibrant", "youthful", "quirky", "punchy", "witty"]
    for kw in maya_tones:
        if kw in tone:
            scores["Maya"] += 1
    for kw in noah_tones:
        if kw in tone:
            scores["Noah"] += 1
    for kw in leslie_tones:
        if kw in tone:
            scores["Leslie"] += 1

    # ── narrative arc ────────────────────────────────────────────────────
    arc = text("narrative_arc")
    if any(w in arc for w in ["emotional", "aspirational", "day-in-life", "before-after"]):
        scores["Maya"] += 1
    if any(w in arc for w in ["problem", "solution", "authority", "hero", "journey"]):
        scores["Noah"] += 1
    if any(w in arc for w in ["reveal", "social proof", "trend", "fast"]):
        scores["Leslie"] += 1

    # ── visual style ─────────────────────────────────────────────────────
    style = text("visual_style", "color_grade", "pacing")
    if any(w in style for w in ["dark-luxury", "editorial", "minimalist", "slow"]):
        scores["Maya"] += 1
    if any(w in style for w in ["hyperreal", "documentary", "dramatic", "slow-burn"]):
        scores["Noah"] += 1
    if any(w in style for w in ["vibrant", "lo-fi", "fast-cut", "tiktok"]):
        scores["Leslie"] += 1

    # ── music mood / pacing ──────────────────────────────────────────────
    music = text("music_mood", "pacing")
    if any(w in music for w in ["orchestral", "string", "piano", "classical", "ambient", "slow"]):
        scores["Maya"] += 1
    if any(w in music for w in ["rock", "epic", "bass", "hip-hop", "trap", "aggressive"]):
        scores["Noah"] += 1
    if any(w in music for w in ["edm", "electronic", "pop", "upbeat", "fast", "synth"]):
        scores["Leslie"] += 1

    # ── platform overrides ───────────────────────────────────────────────
    plat = (platform or "").lower()
    if plat in ("tiktok", "instagram_reels"):
        scores["Leslie"] += 2   # short-form platforms skew youthful/energetic
    elif plat == "linkedin":
        scores["Noah"] += 2     # professional/B2B context
    elif plat in ("youtube",):
        pass                    # no override — let content signals decide

    return scores


def _select_voice(brief: dict, platform: str = "") -> tuple[str, str]:
    """Return (voice_name, reason) using multi-signal scoring.

    GPT-4o's recommendation (from generate_voiceover_script) is preferred
    over this heuristic — call this only as a fallback.
    """
    scores = _score_voices(brief, platform)
    # Sort descending by score; tiebreak order Leslie > Noah > Maya
    tiebreak = ["Leslie", "Noah", "Maya"]
    ranked = sorted(scores.keys(), key=lambda v: (scores[v], tiebreak.index(v) * -1), reverse=True)
    winner = ranked[0]
    score_summary = ", ".join(f"{v}={scores[v]}" for v in ["Maya", "Noah", "Leslie"])
    reason = f"heuristic ({score_summary})"
    return winner, reason


# ── Main entry point ──────────────────────────────────────────────────────
async def run(
    state: PipelineState,
    session_dir: Path,
    emit: Callable,
) -> Optional[str]:
    """Full audio production pipeline.
    Returns path to final_with_audio.mp4, or None if skipped/failed.
    """
    final_video = session_dir / "final.mp4"
    if not final_video.exists():
        await emit({"type": "log", "stage": "9",
                    "message": "No final.mp4 found — skipping audio production.",
                    "level": "warning"})
        print("[audio_gen] no final.mp4, skipping")
        return None

    # Quick heuristic pre-selection for the log message (GPT-4o may override below)
    _pre_voice, _pre_reason = _select_voice(state.enriched_brief, state.platform)
    await emit({"type": "log", "stage": "9",
                "message": f"Audio production starting — RunwayML TTS + SFX (pre-selected: {_pre_voice})",
                "level": "info"})

    # ── 1. Script + music description + voice casting ─────────────────
    try:
        script_data = await generate_voiceover_script(state, emit)
    except Exception as exc:
        print(f"[audio_gen] script generation failed: {exc}")
        await emit({"type": "log", "stage": "9",
                    "message": f"Script generation error: {str(exc)[:120]}", "level": "error"})
        return None

    script          = script_data["script"]
    music_prompt    = script_data["music_prompt"]
    ambient_prompts = script_data.get("ambient_prompts", [])

    # ── Resolve final voice ───────────────────────────────────────────
    if script_data.get("gpt_voice"):
        voice        = script_data["gpt_voice"]
        voice_source = "GPT-4o cast"
    else:
        voice, _reason = _select_voice(state.enriched_brief, state.platform)
        voice_source   = f"heuristic ({_reason})"

    print(f"[audio_gen] final voice={voice} source={voice_source}")
    await emit({"type": "log", "stage": "9",
                "message": f"Voice selected: {voice} ({voice_source})",
                "level": "info"})

    if not script:
        await emit({"type": "log", "stage": "9",
                    "message": "Empty voiceover script — skipping audio.", "level": "warning"})
        return None

    state.voiceover_script = script

    # Guarantee a music prompt if GPT-4o returned nothing useful
    if not music_prompt or len(music_prompt.split()) < 10:
        tone         = state.enriched_brief.get("emotional_tone", "warm")
        music_mood   = state.enriched_brief.get("music_mood", "")
        arc          = state.enriched_brief.get("narrative_arc", "")
        base         = music_mood if music_mood else tone
        music_prompt = (
            f"{base}. Acoustic guitar, Rhodes piano, upright bass. 85 bpm. "
            f"Cinematic nostalgia. Sparse opening, gentle swell midpoint, quiet close. "
            f"Bittersweet, warm. Analog warmth."
        )
        await emit({"type": "log", "stage": "9",
                    "message": f"Music brief (fallback): {music_prompt[:80]}...",
                    "level": "info"})

    # Guarantee ambient prompts if none provided
    if not ambient_prompts:
        environments = [s.environment for s in state.storyboard if s.environment][:2]
        ambient_prompts = [
            f"Ambient atmosphere: {env[:200]}" for env in environments
        ] or ["Soft ambient room tone with gentle background activity"]

    # ── 2. Generate TTS + music bed + ambient SFX concurrently ───────────
    vo_path    = session_dir / "voiceover.mp3"
    music_path = session_dir / "music.mp3"

    tts_task     = synthesize_tts(script, voice, vo_path, emit)
    music_task   = synthesize_music_bed(music_prompt, music_path, emit)
    ambient_task = synthesize_ambient_sfx(ambient_prompts, session_dir, emit)

    (tts_ok, sfx_ok, ambient_paths) = await asyncio.gather(
        tts_task, music_task, ambient_task
    )

    if not tts_ok:
        await emit({"type": "log", "stage": "9",
                    "message": "Voiceover generation failed — skipping audio mux.",
                    "level": "error"})
        return None

    # ── 2b. Multilingual dubbing (optional) ───────────────────────────────
    dub_languages = getattr(state, "dub_languages", None) or state.enriched_brief.get("dub_languages", [])
    dub_results = {}
    if dub_languages and vo_path.exists():
        vo_url = None
        # Upload voiceover to get a URL for the dubbing API
        try:
            import base64
            vo_bytes = vo_path.read_bytes()
            vo_url = f"data:audio/mpeg;base64,{base64.b64encode(vo_bytes).decode()}"
        except Exception as exc:
            print(f"[audio_gen] Failed to encode voiceover for dubbing: {exc}")

        if vo_url:
            dub_results = await synthesize_multilingual_dubs(
                voiceover_url=vo_url,
                target_languages=dub_languages,
                session_dir=session_dir,
                emit=emit,
            )
            if dub_results:
                await emit({"type": "log", "stage": "9",
                            "message": f"Multilingual dubs ready: {', '.join(dub_results.keys())}",
                            "level": "success"})

    # ── 3. Mix all audio layers ───────────────────────────────────────────
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        await emit({"type": "log", "stage": "9",
                    "message": "FFmpeg not found — cannot mix/mux audio.",
                    "level": "error"})
        return None

    mixed_path = session_dir / "mixed_audio.aac"
    mix_duration = state.total_duration + int(getattr(state, "playback_padding_seconds", 0))
    await _mix_audio_tracks(
        ffmpeg=ffmpeg,
        vo_path=vo_path,
        music_path=music_path if sfx_ok else None,
        mixed_path=mixed_path,
        total_duration=mix_duration,
        emit=emit,
        ambient_paths=ambient_paths if ambient_paths else None,
        playback_padding_seconds=int(getattr(state, "playback_padding_seconds", 0)),
    )

    # ── 4. Mux into final video ───────────────────────────────────────────
    audio_video_path = session_dir / "final_with_audio.mp4"
    result_path = await mux_audio_video(
        video_path=final_video,
        audio_path=mixed_path if mixed_path.exists() else vo_path,
        output_path=audio_video_path,
        total_duration=mix_duration,
        emit=emit,
    )

    if result_path:
        n_sfx = (1 if sfx_ok else 0) + len(ambient_paths)
        dub_sources = {
            lang: f"/output/{state.session_id}/voiceover_{lang}.mp3"
            for lang in dub_results.keys()
        }
        await emit({
            "type": "audio_ready",
            "audio_video_src": f"/output/{state.session_id}/final_with_audio.mp4",
            "audio_src":       f"/output/{state.session_id}/voiceover.mp3",
            "music_src":       f"/output/{state.session_id}/music.mp3" if sfx_ok else None,
            "script":          script,
            "voice":           voice,
            "has_music":       sfx_ok,
            "sfx_count":       n_sfx,
            "multilingual_dubs": dub_sources if dub_sources else None,
        })
        layers = ["voiceover"]
        if sfx_ok:
            layers.append("music bed")
        if ambient_paths:
            layers.append(f"{len(ambient_paths)} ambient SFX")
        if dub_results:
            layers.append(f"{len(dub_results)} multilingual dubs")
        await emit({"type": "log", "stage": "9",
                    "message": f"Audio complete: {' + '.join(layers)} mixed into final video",
                    "level": "success"})

    return result_path
