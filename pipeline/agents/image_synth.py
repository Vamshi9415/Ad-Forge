"""Stage 6: Primary Visual Synthesis via RunwayML text-to-image.

CONTINUITY ARCHITECTURE — three-layer lock:

  Layer 1 — Visual world reference
    The anchor shot's base64 image is passed to RunwayML as a `referenceImages`
    entry tagged `@anchor`. RunwayML uses it to lock the VISUAL WORLD (color grade,
    lighting signature, environment feel) — NOT necessarily the same person.

  Layer 2 — GPT-4o Vision world extraction
    After the anchor is generated, GPT-4o Vision reads it and produces a precise
    80-120 word "world document" describing the color grade, lighting direction,
    environment, lens character, and overall visual tone. This is injected verbatim
    into every subsequent shot's text prompt so RunwayML has BOTH a visual and a
    textual blueprint for the WORLD to follow.

  Layer 3 — World consistency retry loop
    If a generated shot drifts from the anchor world (different grade, different
    lighting feel), it is regenerated with a more explicit world-lock prefix,
    up to MAX_CONSISTENCY_RETRIES additional attempts.

Two-phase generation:
  Phase A: shot_01 alone → establishes visual world DNA
  Phase B: shots 2-N in parallel, each receiving:
           • @anchor reference image (world reference)
           • @hero_ref user-upload (if any)
           • GPT-4o extracted world description in the text prompt
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Callable, List, Optional

from openai import AsyncOpenAI

from pipeline.state import PipelineState, CREDIT_COST
from pipeline.runway_client import generate_image, download_bytes

CONCURRENCY = 5
MAX_CONSISTENCY_RETRIES = 2   # extra regen attempts specifically for consistency failures


def _get_openai():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Identity extraction ───────────────────────────────────────────────────

WORLD_SYSTEM = """You are a visual continuity supervisor on a high-end advertising shoot.
Your job is to write a precise WORLD LOCK description of a reference frame so that
AI image generators can replicate the same VISUAL UNIVERSE across all shots.
The world lock is about COLOR, LIGHT, and ENVIRONMENT — NOT about forcing the same person in frame.

Describe exactly what you see in ONE dense paragraph (80-120 words). Cover:
  • Color grade: dominant palette, shadow colour, highlight colour, split tone (e.g. "warm teal shadows, cream highlights")
  • Lighting: direction (e.g. "warm key light from camera-right at 45°"), quality, colour temperature
  • Environment/setting: location type, time of day, depth of field character, background elements
  • Lens character: focal length feel, aperture feel, bokeh shape, any flares or grain
  • Overall cinematic tone: adjectives that describe the mood and visual world

Return ONLY the plain-text description — no JSON, no bullet points, no preamble."""


async def _extract_anchor_world(anchor_base64: str, emit: Callable) -> str:
    """Use GPT-4o Vision to produce a precise WORLD document from the anchor image.

    Returns a ~100 word description of the visual world (color grade, lighting,
    environment) that is injected verbatim into every subsequent shot's text prompt.
    """
    try:
        resp = await _get_openai().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": WORLD_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Write the WORLD LOCK description for this advertising anchor frame:"},
                        {"type": "image_url",
                         "image_url": {"url": anchor_base64, "detail": "high"}},
                    ],
                },
            ],
            max_tokens=250,
        )
        world_doc = (resp.choices[0].message.content or "").strip()
        print(f"[image_synth] anchor world extracted: {len(world_doc.split())} words")
        return world_doc
    except Exception as exc:
        print(f"[image_synth] world extraction failed: {exc} — using generic prefix")
        return ""


def _build_continuity_prompt(shot_prompt: str, world_doc: str) -> str:
    """Inject the world document into a shot prompt so RunwayML has both a visual
    reference (@anchor image) and a precise textual world blueprint to follow.
    """
    if not world_doc:
        if "@anchor" in shot_prompt:
            return shot_prompt
        return (
            "@anchor — same visual world, color grade, and lighting as the anchor frame. "
            + shot_prompt
        )

    world_prefix = f"@anchor — SAME VISUAL WORLD as anchor: {world_doc}"
    if "@anchor" in shot_prompt:
        return shot_prompt.replace("@anchor", world_prefix, 1)
    return f"{world_prefix} {shot_prompt}"


# ── Single-shot generation helper ────────────────────────────────────────

async def _generate_one(
    shot,
    state: PipelineState,
    session_dir: Path,
    api_key: str,
    references: Optional[List[dict]],
    prompt_override: Optional[str],
    emit: Callable,
    suffix: str = "",   # e.g. "_r1" for regen files
):
    """Generate one shot image, persist to disk, and update shot state."""
    used_prompt = prompt_override if prompt_override is not None else shot.image_prompt

    async def shot_log(msg, level="info"):
        await emit({"type": "log", "stage": "6",
                    "message": f"[{shot.shot_id}] {msg}", "level": level})

    await emit({
        "type": "log", "stage": "6",
        "message": f"[{shot.shot_id}] Submitting image generation{' (retry)' if suffix else ''}...",
        "level": "info",
    })

    try:
        shot.image_status = "generating"

        image_url = await generate_image(
            api_key=api_key,
            prompt_text=used_prompt,
            ratio=state.ratio,
            model=state.image_model,
            reference_images=references if references else None,
            log=shot_log,
        )

        img_bytes = await download_bytes(image_url)
        img_path = session_dir / "images" / f"{shot.shot_id}{suffix}.jpg"
        img_path.write_bytes(img_bytes)

        b64 = base64.b64encode(img_bytes).decode()
        shot.image_base64 = f"data:image/jpeg;base64,{b64}"
        shot.image_path = str(img_path)
        shot.image_url = image_url
        shot.image_status = "generated"
        shot.image_prompt = used_prompt   # keep the actual prompt used

        credits = CREDIT_COST.get(state.image_model, 6)
        shot.cost_credits += credits
        state.total_cost_credits += credits
        state.total_cost_usd = state.total_cost_credits * 0.01

        await emit({
            "type": "image_ready",
            "shot_id": shot.shot_id,
            "src": f"/output/{state.session_id}/images/{shot.shot_id}{suffix}.jpg",
            "prompt": used_prompt[:100],
            "status": "generated",
            "cost_credits": state.total_cost_credits,
            "cost_usd": round(state.total_cost_usd, 2),
        })
        await emit({
            "type": "log", "stage": "6",
            "message": f"[{shot.shot_id}] Frame generated (+{credits} credits)",
            "level": "success",
        })
        return True

    except Exception as exc:
        shot.image_status = "failed"
        await emit({
            "type": "log", "stage": "6",
            "message": f"[{shot.shot_id}] Image generation failed: {exc}",
            "level": "error",
        })
        return False


# ── Main entry point ──────────────────────────────────────────────────────

async def run(state: PipelineState, session_dir: Path, emit: Callable) -> None:
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")

    if not state.storyboard:
        await emit({"type": "log", "stage": "6",
                    "message": "No storyboard shots to render.", "level": "warning"})
        return

    await emit({
        "type": "log", "stage": "6",
        "message": (
            f"Generating {len(state.storyboard)} reference frames via "
            f"RunwayML {state.image_model} (visual world continuity architecture)..."
        ),
        "level": "info",
    })

    # Optional user-uploaded reference (always @hero_ref)
    user_hero_ref: Optional[dict] = None
    if state.image_data_uri:
        user_hero_ref = {"tag": "hero_ref", "uri": state.image_data_uri}
        await emit({"type": "log", "stage": "6",
                    "message": "User reference image → @hero_ref (every shot)",
                    "level": "info"})

    # ── PHASE A: Anchor shot ──────────────────────────────────────────────
    anchor_shot = state.storyboard[0]
    anchor_refs = [user_hero_ref] if user_hero_ref else None

    await emit({"type": "log", "stage": "6",
                "message": f"[{anchor_shot.shot_id}] Generating ANCHOR frame...",
                "level": "info"})

    await _generate_one(
        anchor_shot, state, session_dir, api_key,
        references=anchor_refs,
        prompt_override=None,
        emit=emit,
    )

    # ── World extraction from anchor ──────────────────────────────────────
    world_doc = ""
    if anchor_shot.image_status == "generated" and anchor_shot.image_base64:
        await emit({"type": "log", "stage": "6",
                    "message": "Extracting visual world document (GPT-4o Vision)...",
                    "level": "info"})
        world_doc = await _extract_anchor_world(anchor_shot.image_base64, emit)

        # Store on state so VLM validator can also use it
        state.enriched_brief["_anchor_world"] = world_doc
        # Keep backwards-compat key
        state.enriched_brief["_anchor_identity"] = world_doc

        if world_doc:
            await emit({"type": "log", "stage": "6",
                        "message": f"Visual world locked: {world_doc[:120]}...",
                        "level": "success"})
        else:
            await emit({"type": "log", "stage": "6",
                        "message": "World extraction failed — using visual-only anchor reference",
                        "level": "warning"})
    else:
        await emit({"type": "log", "stage": "6",
                    "message": "Anchor failed — falling back to independent generation.",
                    "level": "warning"})

    # Build references for shots 2-N
    if anchor_shot.image_status == "generated" and anchor_shot.image_base64:
        base_refs = [{"tag": "anchor", "uri": anchor_shot.image_base64}]
        if user_hero_ref:
            base_refs.append(user_hero_ref)
        await emit({"type": "log", "stage": "6",
                    "message": (
                        f"Anchor established — generating {len(state.storyboard) - 1} "
                        "story shots in parallel with @anchor world lock..."
                    ),
                    "level": "success"})
    else:
        base_refs = [user_hero_ref] if user_hero_ref else None
        world_doc = ""   # no anchor to enforce

    # ── PHASE B: Story shots in parallel ─────────────────────────────────
    other_shots = state.storyboard[1:]
    if not other_shots:
        successful = 1 if anchor_shot.image_status == "generated" else 0
        await emit({"type": "log", "stage": "6",
                    "message": f"Image synthesis complete: {successful}/1 frames",
                    "level": "success" if successful else "warning"})
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def gen_continuity_shot(shot, attempt: int = 0):
        """Generate with world-locked prompt; retry on failure up to MAX_CONSISTENCY_RETRIES."""
        async with semaphore:
            prompt = _build_continuity_prompt(shot.image_prompt, world_doc)
            suffix = f"_c{attempt}" if attempt > 0 else ""
            ok = await _generate_one(
                shot, state, session_dir, api_key,
                references=base_refs,
                prompt_override=prompt,
                emit=emit,
                suffix=suffix,
            )
            if not ok and attempt < MAX_CONSISTENCY_RETRIES:
                await emit({"type": "log", "stage": "6",
                            "message": f"[{shot.shot_id}] Retrying with stronger world lock "
                                       f"(attempt {attempt + 2}/{MAX_CONSISTENCY_RETRIES + 1})...",
                            "level": "warning"})
                stronger = (
                    f"@anchor — CRITICAL: replicate the anchor frame's VISUAL WORLD (color grade, "
                    f"lighting, environment atmosphere) EXACTLY. "
                    f"{world_doc} "
                    f"This is attempt {attempt + 2} — stay within the established visual universe. "
                    + shot.image_prompt
                )
                await _generate_one(
                    shot, state, session_dir, api_key,
                    references=base_refs,
                    prompt_override=stronger,
                    emit=emit,
                    suffix=f"_c{attempt + 1}",
                )

    await asyncio.gather(*[gen_continuity_shot(s) for s in other_shots])

    successful = sum(1 for s in state.storyboard if s.image_status == "generated")
    await emit({
        "type": "log", "stage": "6",
        "message": f"Image synthesis complete: {successful}/{len(state.storyboard)} frames generated",
        "level": "success" if successful == len(state.storyboard) else "warning",
    })
