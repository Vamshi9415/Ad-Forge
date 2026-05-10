"""Dedicated mini-pipeline: professional broadcast-style end card frame via GPT + RunwayML image.

Runs at the start of stage 8 (before stitch). Output:
  session_dir/images/promo_end_professional.jpg

The FFmpeg assembler prefers this asset (typography baked into the image) and skips
ASS subtitle overlays to avoid double text.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Callable, List, Optional

from openai import AsyncOpenAI

from pipeline.state import CREDIT_COST, PipelineState
from pipeline.runway_client import generate_image, download_bytes


OUTPUT_FILENAME = "promo_end_professional.jpg"

_PROMPT_SYSTEM = """You write dense RunwayML Gen-4 image prompts for premium TV/streaming advertisement END CARDS / CLOSING SLATES.

Hard rules:
- Output MUST be a single JSON object only (no markdown fences): {"runway_prompt":"..."}
- runway_prompt is ONE STRING under 950 characters, plain ASCII punctuation preferred.
- Describe one STATIC closing-frame graphic suitable as final broadcast slate — luxury minimal layout, clear typography hierarchy, broadcast-safe margins (~10%), generous negative space, subtle vignette or soft gradient acceptable.
- The slate MUST visibly incorporate EXACT headline text and EXACT CTA text words-for-word given by the user (spell precisely — viewer-readable faux typography integrated into the design).
- Photoreal UI-inspired typography OR premium matte poster aesthetic — NOT a cluttered collage.
- Composition should feel hero-centric (product / brand lockup ready for a gentle push-in in post — leave breathable framing).
- Mention ratio implicitly via composition clarity only — never contradict requested vibe/color."""

def _build_user_prompt(state: PipelineState, title: str, cta: str, world_lock: str) -> str:
    """Brace-safe assembly (titles may contain curly braces)."""
    brief_blob = _brief_blob(state)[:1200]
    return (
        f'Build runway_prompt for a closing slate matching ratio "{state.ratio}".\n\n'
        f"BRAND / CAMPAIGN TITLE (exact headline text — must appear verbatim on slate): {title}\n"
        f"CTA LINE (exact text — must appear verbatim): {cta}\n"
        f"PLATFORM: {state.platform}\n\n"
        "VISUAL BRIEF (follow tone and palette):\n"
        f"{brief_blob}\n\n"
        "WORLD LOCK (optional — align color/lighting language if provided):\n"
        f"{world_lock or '(none)'}\n\n"
        "Reference tags @anchor / @hero_ref (if present on API call) lock continuity — "
        "instruct prompt to harmonize WITH those references without copying unrelated subjects."
    )


def _get_openai():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _brief_blob(state: PipelineState) -> str:
    b = state.enriched_brief or {}
    parts = [
        f"visual_style: {b.get('visual_style', '')}",
        f"color_grade: {b.get('color_grade', '')}",
        f"emotional_tone: {b.get('emotional_tone', b.get('tone', ''))}",
        f"creative_direction: {str(b.get('creative_direction', ''))[:400]}",
        f"product_name: {b.get('product_name', '')}",
        f"tagline: {b.get('tagline', '')}",
    ]
    return "\n".join(parts)


def _anchor_uri(state: PipelineState) -> Optional[str]:
    if not state.storyboard:
        return None
    shot = state.storyboard[0]
    if shot.image_base64 and shot.image_base64.startswith("data:"):
        return shot.image_base64
    if shot.image_path and Path(shot.image_path).is_file():
        try:
            raw = Path(shot.image_path).read_bytes()
            b64 = base64.b64encode(raw).decode()
            return f"data:image/jpeg;base64,{b64}"
        except OSError:
            return None
    return None


def _reference_stack(state: PipelineState) -> Optional[List[dict]]:
    refs: List[dict] = []
    anchor = _anchor_uri(state)
    if anchor:
        refs.append({"tag": "anchor", "uri": anchor})
    if state.image_data_uri:
        refs.append({"tag": "hero_ref", "uri": state.image_data_uri})
    return refs or None


def _title_cta(state: PipelineState) -> tuple[str, str]:
    b = state.enriched_brief or {}
    title = (b.get("campaign_title") or b.get("brand_name") or "Your Brand").strip()
    cta = (
        b.get("primary_cta")
        or b.get("cta")
        or b.get("unique_selling_point")
        or "Shop now"
    )
    if isinstance(cta, str):
        cta = cta.strip()
    else:
        cta = str(cta).strip()
    return title, cta


async def run(state: PipelineState, session_dir: Path, emit: Callable) -> None:
    """Generate promo_end_professional.jpg when API keys allow; never raises."""
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")
    out_path = session_dir / "images" / OUTPUT_FILENAME

    async def log(msg: str, level: str = "info"):
        await emit({"type": "log", "stage": "8", "message": f"[End-card pipeline] {msg}", "level": level})

    if out_path.is_file() and out_path.stat().st_size > 2048:
        await log("Professional slate already present — skipping regeneration.", "info")
        return

    if not api_key:
        await log("RUNWAYML_API_SECRET missing — skipping AI end-card generation.", "warning")
        return

    if not state.storyboard:
        await log("No storyboard — skipping AI end-card generation.", "warning")
        return

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        await log("OPENAI_API_KEY missing — skipping AI end-card prompt planning.", "warning")
        return

    title, cta = _title_cta(state)
    world_lock = (state.enriched_brief.get("_anchor_world") or "")[:900]

    await log("Planning professional closing slate (GPT-4o → Runway Gen-4 image)...", "info")

    user_payload = _build_user_prompt(state, title, cta, world_lock)

    runway_prompt = ""
    try:
        resp = await _get_openai().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        runway_prompt = (raw.get("runway_prompt") or "").strip()
    except Exception as e:
        await log(f"Prompt planning failed ({type(e).__name__}) — {e}", "warning")
        return

    if len(runway_prompt) < 80:
        await log("GPT returned unusable runway_prompt — skipping AI end-card.", "warning")
        return

    refs = _reference_stack(state)

    async def rw_log(msg: str, level: str = "info"):
        await log(msg, level)

    try:
        url = await generate_image(
            api_key=api_key,
            prompt_text=runway_prompt,
            ratio=state.ratio,
            model=state.image_model,
            reference_images=refs,
            log=rw_log,
        )
        img_bytes = await download_bytes(url)
        if len(img_bytes) < 1024:
            await log("Downloaded image too small — discarding.", "warning")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_bytes)
        credits = CREDIT_COST.get(state.image_model, 6)
        state.total_cost_credits += credits
        state.total_cost_usd = state.total_cost_credits * 0.01
        await log(
            f"Professional end-card frame saved (+{credits} credits).",
            "success",
        )
        await emit({
            "type": "promo_end_professional_ready",
            "src": f"/output/{state.session_id}/images/{OUTPUT_FILENAME}",
        })
    except Exception as e:
        await log(f"Runway image generation failed ({type(e).__name__}) — {e}", "warning")
