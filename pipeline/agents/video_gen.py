"""Stage 7: Temporal Media Synthesis — Image-to-Video via RunwayML."""

import asyncio
import os
from pathlib import Path
from typing import Callable

from pipeline.state import PipelineState, CREDIT_COST, VIDEO_MODEL_DURATIONS
from pipeline.runway_client import generate_video, download_bytes

# Submit all clips simultaneously — RunwayML throttles on their end anyway.
# Submitting early means clips start running sooner instead of waiting for
# the previous clip to fully complete before even entering the queue.
CONCURRENCY = 5
MAX_VIDEO_RETRIES = 2          # attempts after first failure (total = 3)
RETRY_DELAYS = [30, 60]        # exponential backoff: 30s then 60s between retries


async def run(state: PipelineState, session_dir: Path, emit: Callable) -> None:
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    # Proceed if we have a base image — VLM failures are advisory, not blocking
    eligible = [s for s in state.storyboard if s.image_base64 and s.image_status not in ("pending", "generating", "failed")]
    skipped = len(state.storyboard) - len(eligible)

    await emit({
        "type": "log",
        "stage": "7",
        "message": f"Synthesizing {len(eligible)} video clips via RunwayML {state.video_model}...",
        "level": "info",
    })
    if skipped > 0:
        await emit({
            "type": "log",
            "stage": "7",
            "message": f"{skipped} shot(s) skipped — no valid base frame",
            "level": "warning",
        })

    semaphore = asyncio.Semaphore(CONCURRENCY)

    def _clamp_duration(desired: int, model: str) -> int:
        """Return the best allowed duration for this model, >= desired if possible."""
        allowed = VIDEO_MODEL_DURATIONS.get(model)
        if not allowed:
            return desired
        longer = [d for d in allowed if d >= desired]
        return min(longer) if longer else max(allowed)

    def _poll_timeout(duration_secs: int) -> int:
        """Scale poll timeout by clip duration.

        A 5s clip on a busy queue can take ~6 minutes.
        A 10s clip can take ~12-15 minutes (was timing out at 10min hard cap).
        Formula: 480s base + 60s per second of clip duration.
        """
        return 480 + duration_secs * 60   # 5s→540s, 10s→1080s, 15s→1380s

    async def generate_shot_video(shot):
        async with semaphore:
            async def shot_log(msg, level="info"):
                await emit({"type": "log", "stage": "7", "message": f"[{shot.shot_id}] {msg}", "level": level})

            shot.video_status = "generating"
            duration = _clamp_duration(shot.duration, state.video_model)
            if duration != shot.duration:
                await shot_log(
                    f"Duration clamped {shot.duration}s → {duration}s "
                    f"(model {state.video_model} allows {VIDEO_MODEL_DURATIONS.get(state.video_model)})", "info"
                )

            await shot_log(f"Starting {duration}s video generation...", "info")

            video_url = None
            last_exc  = None

            # ── Retry loop with exponential backoff ──────────────────────────
            for attempt in range(MAX_VIDEO_RETRIES + 1):
                if attempt > 0:
                    delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
                    await shot_log(
                        f"Retry {attempt}/{MAX_VIDEO_RETRIES} — waiting {delay}s (backoff)...",
                        "warning"
                    )
                    await asyncio.sleep(delay)

                try:
                    video_url = await generate_video(
                        api_key=api_key,
                        prompt_image=shot.image_base64,
                        prompt_text=shot.motion_prompt,
                        model=state.video_model,
                        ratio=state.ratio,
                        duration=duration,
                        seed=shot.seed,
                        log=shot_log,
                        timeout=_poll_timeout(duration),
                    )
                    last_exc = None
                    break   # success — exit retry loop
                except Exception as exc:
                    last_exc = exc
                    await shot_log(
                        f"Attempt {attempt + 1}/{MAX_VIDEO_RETRIES + 1} failed: {str(exc)[:120]}",
                        "warning" if attempt < MAX_VIDEO_RETRIES else "error",
                    )

            # ── All retries failed — try shorter duration on timeout ──────────
            if last_exc is not None:
                timed_out = "timed out" in str(last_exc).lower()
                allowed   = VIDEO_MODEL_DURATIONS.get(state.video_model, [5, 10])
                shorter   = [d for d in sorted(allowed) if d < duration]
                fallback_dur = shorter[-1] if shorter else None

                if timed_out and fallback_dur:
                    await shot_log(
                        f"All retries exhausted (timeout) — last fallback at {fallback_dur}s...",
                        "warning"
                    )
                    try:
                        video_url = await generate_video(
                            api_key=api_key,
                            prompt_image=shot.image_base64,
                            prompt_text=shot.motion_prompt,
                            model=state.video_model,
                            ratio=state.ratio,
                            duration=fallback_dur,
                            seed=shot.seed,
                            log=shot_log,
                            timeout=_poll_timeout(fallback_dur),
                        )
                        duration  = fallback_dur
                        last_exc  = None
                    except Exception as exc2:
                        await shot_log(f"Duration fallback also failed: {exc2}", "error")

            if video_url is None:
                shot.video_status = "failed"
                await shot_log(f"Video generation failed after all retries.", "error")
                return

            # ── Success ────────────────────────────────────────────────────
            try:
                video_bytes = await download_bytes(video_url)
            except Exception as exc:
                shot.video_status = "failed"
                await emit({"type": "log", "stage": "7",
                            "message": f"[{shot.shot_id}] Download failed: {exc}", "level": "error"})
                return

            video_path = session_dir / "videos" / f"{shot.shot_id}.mp4"
            video_path.write_bytes(video_bytes)

            shot.video_path = str(video_path)
            shot.video_status = "complete"

            credits_per_sec = CREDIT_COST.get(state.video_model, 5)
            clip_credits = credits_per_sec * duration
            shot.cost_credits += clip_credits
            state.total_cost_credits += clip_credits
            state.total_cost_usd = state.total_cost_credits * 0.01

            await emit({
                "type": "video_ready",
                "shot_id": shot.shot_id,
                "src": f"/output/{state.session_id}/videos/{shot.shot_id}.mp4",
                "duration": duration,
                "cost_credits": state.total_cost_credits,
                "cost_usd": round(state.total_cost_usd, 2),
            })
            await emit({
                "type": "log", "stage": "7",
                "message": (
                    f"[{shot.shot_id}] {duration}s clip complete "
                    f"(+{clip_credits} credits, ${clip_credits*0.01:.2f})"
                ),
                "level": "success",
            })

    tasks = [generate_shot_video(shot) for shot in eligible]
    await asyncio.gather(*tasks)

    completed = sum(1 for s in state.storyboard if s.video_status == "complete")
    await emit({
        "type": "log",
        "stage": "7",
        "message": f"Video synthesis complete: {completed}/{len(eligible)} clips rendered | Total cost: {state.total_cost_credits} credits (${state.total_cost_usd:.2f})",
        "level": "success" if completed > 0 else "error",
    })
