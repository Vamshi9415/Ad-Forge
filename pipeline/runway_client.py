"""Async RunwayML API client using httpx."""

import asyncio
import os
import ssl
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

API_BASE = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
POLL_INTERVAL = 8
POLL_TIMEOUT = 600
MAX_RETRIES = 3


def _ssl_context():
    """Return the best available SSL context for this environment.

    Priority:
      1. truststore  — uses the OS/Windows certificate store, which already
                       trusts corporate proxy / VPN root CAs.
      2. verify=False — fallback for environments where truststore is not
                        installed and SSL inspection causes CERTIFICATE_VERIFY_FAILED.
                        Controlled by RUNWAY_SSL_VERIFY=0 env var.

    If neither is needed (normal internet, no SSL inspection) httpx's default
    certifi-based verification is used.
    """
    # Explicit opt-out via env var
    if os.environ.get("RUNWAY_SSL_VERIFY", "1") == "0":
        return False

    # Try truststore (Windows / macOS / Linux system cert store)
    try:
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ctx
    except ImportError:
        pass

    # Standard certifi bundle — works on most networks
    return True   # httpx default


_SSL = _ssl_context()


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": API_VERSION,
        "Content-Type": "application/json",
    }


async def _post(api_key: str, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    headers = _headers(api_key)
    delays = [5, 15, 45]
    async with httpx.AsyncClient(timeout=60, verify=_SSL) as client:
        for attempt in range(MAX_RETRIES + 1):
            r = await client.post(f"{API_BASE}{endpoint}", headers=headers, json=body)
            if r.is_success:
                return r.json()
            if r.status_code in (429, 502, 503, 504) and attempt < MAX_RETRIES:
                await asyncio.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise RuntimeError(
                f"RunwayML API error {r.status_code}: {r.text[:500]}"
            )
    raise RuntimeError("Max retries exceeded")


async def _get(api_key: str, path: str) -> Dict[str, Any]:
    headers = _headers(api_key)
    async with httpx.AsyncClient(timeout=30, verify=_SSL) as client:
        r = await client.get(f"{API_BASE}{path}", headers=headers)
        r.raise_for_status()
        return r.json()


async def poll_task(
    api_key: str,
    task_id: str,
    log: Optional[Callable] = None,
    timeout: int = POLL_TIMEOUT,
) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout:
        data = await _get(api_key, f"/v1/tasks/{task_id}")
        status = data.get("status", "")
        progress = data.get("progress", 0)

        if status == "SUCCEEDED":
            return data
        if status == "FAILED":
            failure = data.get("failure", "Unknown error")
            raise RuntimeError(f"Task FAILED: {failure}")
        if status == "CANCELLED":
            raise RuntimeError("Task was CANCELLED")

        elapsed = int(time.time() - start)
        if log:
            await log(f"Task [{task_id[:10]}] {status} {int(progress*100)}% ({elapsed}s)", "info")
        await asyncio.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Task timed out after {timeout}s")


IMAGE_PROMPT_MAX  = 1000   # RunwayML /v1/text_to_image hard limit
VIDEO_PROMPT_MAX  = 1000   # RunwayML /v1/image_to_video hard limit
SFX_PROMPT_MAX    = 450    # RunwayML /v1/sound_effect hard limit


def _truncate(text: str, max_len: int) -> str:
    """Truncate at a word boundary so the prompt stays grammatically sensible."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated


async def generate_image(
    api_key: str,
    prompt_text: str,
    ratio: str = "1280:720",
    model: str = "gen4_image",
    reference_images: Optional[List[Dict]] = None,
    log: Optional[Callable] = None,
) -> str:
    """Generate an image and return the output URL."""
    safe_prompt = _truncate(prompt_text, IMAGE_PROMPT_MAX)
    if len(safe_prompt) < len(prompt_text) and log:
        await log(
            f"Image prompt truncated {len(prompt_text)}→{len(safe_prompt)} chars "
            f"(RunwayML 1000-char limit)", "warning"
        )
    body: Dict[str, Any] = {
        "model": model,
        "promptText": safe_prompt,
        "ratio": ratio,
    }
    if reference_images:
        body["referenceImages"] = reference_images

    if log:
        await log(f"Submitting image generation task (model={model})", "info")

    task = await _post(api_key, "/v1/text_to_image", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in response: {task}")

    if log:
        await log(f"Image task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URLs in image task result")
    return outputs[0]


async def generate_video(
    api_key: str,
    prompt_image: str,
    prompt_text: str,
    model: str = "gen4_turbo",
    ratio: str = "1280:720",
    duration: int = 5,
    seed: Optional[int] = None,
    log: Optional[Callable] = None,
    timeout: Optional[int] = None,
) -> str:
    """Generate a video clip and return the output URL.

    timeout: poll timeout in seconds. Defaults to POLL_TIMEOUT (600s) but
    callers should scale this by clip duration since longer clips + throttled
    queues can easily need 12-15 minutes.
    """
    safe_motion = _truncate(prompt_text, VIDEO_PROMPT_MAX)
    if len(safe_motion) < len(prompt_text) and log:
        await log(
            f"Video prompt truncated {len(prompt_text)}→{len(safe_motion)} chars "
            f"(RunwayML 1000-char limit)", "warning"
        )
    body: Dict[str, Any] = {
        "model": model,
        "promptImage": prompt_image,
        "promptText": safe_motion,
        "ratio": ratio,
        "duration": duration,
    }
    if seed is not None:
        body["seed"] = seed

    if log:
        await log(
            f"Submitting video task (model={model}, {duration}s, {ratio})", "info"
        )

    task = await _post(api_key, "/v1/image_to_video", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in response: {task}")

    if log:
        await log(f"Video task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log, timeout=timeout or POLL_TIMEOUT)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URLs in video task result")
    return outputs[0]


async def download_bytes(url: str) -> bytes:
    """Download bytes from a URL."""
    async with httpx.AsyncClient(timeout=120, verify=_SSL) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


# ── RunwayML Audio API ────────────────────────────────────────────────────
# Powered by ElevenLabs models surfaced through the RunwayML API.
# Voices (TTS presets): Maya · Noah · Leslie
# Credits: TTS ≈ 1cr/50 chars   SFX ≈ 1-2cr per generation

TTS_MODEL = "eleven_multilingual_v2"
SFX_MODEL = "eleven_text_to_sound_v2"
STS_MODEL = "eleven_multilingual_sts_v2"
DUB_MODEL = "eleven_voice_dubbing"
ISOLATION_MODEL = "eleven_voice_isolation"

SUPPORTED_LANGUAGES = [
    "en", "ja", "zh", "de", "hi", "fr", "ko", "pt", "it", "es",
    "id", "nl", "tr", "fil", "pl", "sv", "bg", "ro", "ar", "cs",
    "el", "fi", "hr", "ms", "sk", "da", "ta", "uk", "ru",
]


async def generate_tts(
    api_key: str,
    text: str,
    voice_preset: str = "Maya",
    log: Optional[Callable] = None,
) -> str:
    """Generate speech from text using RunwayML TTS (ElevenLabs eleven_multilingual_v2).

    Returns a temporary URL to the generated MP3/audio file.
    voice_preset: 'Maya' (female, warm), 'Noah' (male, authoritative), 'Leslie' (neutral, clear)
    """
    body: Dict[str, Any] = {
        "model": TTS_MODEL,
        "promptText": text,
        "voice": {"type": "runway-preset", "presetId": voice_preset},
    }

    if log:
        await log(f"Submitting TTS task (voice={voice_preset}, ~{len(text)} chars)", "info")

    task = await _post(api_key, "/v1/text_to_speech", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in TTS response: {task}")

    if log:
        await log(f"TTS task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URL in TTS task result")
    return outputs[0]


async def generate_sfx(
    api_key: str,
    prompt_text: str,
    log: Optional[Callable] = None,
) -> str:
    """Generate a sound effect / music bed from a text prompt using RunwayML SFX
    (ElevenLabs eleven_text_to_sound_v2).

    Returns a temporary URL to the generated audio file.
    Tip: describe mood, instruments, tempo, and feel for best results.
    """
    safe_sfx = _truncate(prompt_text, SFX_PROMPT_MAX)
    if len(safe_sfx) < len(prompt_text) and log:
        await log(
            f"SFX prompt truncated {len(prompt_text)}→{len(safe_sfx)} chars "
            f"(RunwayML 450-char limit)", "warning"
        )
    body: Dict[str, Any] = {
        "model": SFX_MODEL,
        "promptText": safe_sfx,
    }

    if log:
        await log(f"Submitting SFX task: {safe_sfx[:60]}...", "info")

    task = await _post(api_key, "/v1/sound_effect", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in SFX response: {task}")

    if log:
        await log(f"SFX task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URL in SFX task result")
    return outputs[0]


async def generate_sts(
    api_key: str,
    audio_url: str,
    voice_preset: str = "Maya",
    remove_background_noise: bool = False,
    log: Optional[Callable] = None,
) -> str:
    """Convert speech to another voice using RunwayML STS (eleven_multilingual_sts_v2).

    Preserves the original language, intonation, and pacing while applying
    the target voice. Supports all 29 languages of the multilingual model.

    Returns a temporary URL to the generated audio file.
    """
    body: Dict[str, Any] = {
        "model": STS_MODEL,
        "media": {"type": "audio", "uri": audio_url},
        "voice": {"type": "runway-preset", "presetId": voice_preset},
    }
    if remove_background_noise:
        body["removeBackgroundNoise"] = True

    if log:
        await log(f"Submitting STS task (voice={voice_preset})", "info")

    task = await _post(api_key, "/v1/speech_to_speech", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in STS response: {task}")

    if log:
        await log(f"STS task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URL in STS task result")
    return outputs[0]


async def generate_dubbing(
    api_key: str,
    audio_url: str,
    target_language: str,
    log: Optional[Callable] = None,
) -> str:
    """Dub audio into another language using RunwayML (eleven_voice_dubbing).

    Translates and re-synthesizes speech in the target language while
    preserving the speaker's voice characteristics.

    target_language: ISO language code (e.g. "es", "fr", "de", "hi", "ja")
    Returns a temporary URL to the dubbed audio file.
    """
    body: Dict[str, Any] = {
        "model": DUB_MODEL,
        "audioUri": audio_url,
        "targetLang": target_language,
    }

    if log:
        await log(f"Submitting dubbing task (target={target_language})", "info")

    task = await _post(api_key, "/v1/voice_dubbing", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in dubbing response: {task}")

    if log:
        await log(f"Dubbing task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URL in dubbing task result")
    return outputs[0]


async def generate_voice_isolation(
    api_key: str,
    audio_url: str,
    log: Optional[Callable] = None,
) -> str:
    """Isolate voice from noisy audio using RunwayML (eleven_voice_isolation).

    Removes background noise and extracts clean speech.
    Returns a temporary URL to the cleaned audio file.
    """
    body: Dict[str, Any] = {
        "model": ISOLATION_MODEL,
        "audioUri": audio_url,
    }

    if log:
        await log("Submitting voice isolation task", "info")

    task = await _post(api_key, "/v1/voice_isolation", body)
    task_id = task.get("id")
    if not task_id:
        raise RuntimeError(f"No task ID in isolation response: {task}")

    if log:
        await log(f"Voice isolation task created: {task_id}", "info")

    result = await poll_task(api_key, task_id, log=log)
    outputs = result.get("output", [])
    if not outputs:
        raise RuntimeError("No output URL in voice isolation task result")
    return outputs[0]
