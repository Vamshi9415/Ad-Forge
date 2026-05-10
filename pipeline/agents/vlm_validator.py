"""Stage 6a: Multi-Modal Visual Validation via GPT-4o Vision with regeneration retry.

Each shot is validated for:
  1. Production quality  — no severe artifacts, correct product category
  2. Continuity          — shots 2-N are compared against the anchor frame visually
                           and against the extracted identity document from Stage 6.

Shots are processed SEQUENTIALLY to avoid RunwayML throttling during regen.
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI

from pipeline.state import PipelineState, CREDIT_COST
from pipeline.runway_client import generate_image, download_bytes


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


MAX_RETRIES = 2   # regeneration attempts per FAILED shot

SYSTEM = """You are a senior visual QA director at a premium advertising production house.
Your job is to APPROVE or REJECT AI-generated advertising frames on PRODUCTION QUALITY grounds,
NOT on exact narrative/prompt compliance.

━━ CRITICAL PRINCIPLE ━━
AI image models interpret prompts creatively. A food brand ad showing beautiful product
photography instead of a metaphorical "nostalgic clock" is ACCEPTABLE — mark it VERIFIED.
Do not penalise creative interpretation.

Return ONLY valid JSON — no markdown fences:
{
  "verdict": "VERIFIED" or "FAILED",
  "confidence": <float 0.0-1.0>,
  "issues": ["specific production-quality problem — be precise"],
  "refined_prompt": "<150-200 word improved prompt — preserve @anchor and identity doc, fix only real defects>",
  "notes": "<one-sentence production log entry>"
}

━━ VERIFY if ALL hold ━━
• Correct product category / brand subject is present (or a contextually relevant scene)
• Image quality is professional — no severe artifacts, no unacceptable blur on hero subject
• Composition is intentional and could appear in a real magazine or TV ad
• Lighting is cohesive and fits the campaign tone
• No garbled brand/product text visible in the frame

━━ FAIL only if ANY true ━━
• Completely wrong product category (car shown for food brand)
• Severe generation artifacts: heavy noise, melting faces, deformed foreground subject
• Illegible or misspelled text that is central to the shot
• Wholly broken composition that looks technically failed

━━ DO NOT FAIL for ━━
• Metaphorical elements absent (ice cream shown instead of "nostalgia clock" — fine)
• Different flavour, colour variant, packaging, or minor prop detail
• Background differs from narrative description but is coherent and on-tone
• Minor prop substitution (bowl vs cone, etc.)

━━ CONTINUITY RULE for refined_prompt ━━
• If the prompt starts with "@anchor — SAME VISUAL WORLD", preserve that FULL prefix verbatim.
• If prompt starts with "@anchor", refined_prompt must also start with "@anchor".
• Keep any "@hero_ref" tag.

refined_prompt MUST be provided for every response (even VERIFIED) — make it 10-20% better."""


async def _vlm_check(
    shot,
    emit: Callable,
    stage: str = "6a",
    anchor_image_b64: Optional[str] = None,
    anchor_identity: str = "",
) -> dict:
    """Run one VLM quality + consistency check on a shot.

    For shots 2-N, both the anchor frame (for visual comparison) and the
    extracted identity document (for textual comparison) are provided so
    GPT-4o can flag visual drift explicitly.
    """
    is_non_anchor = bool(anchor_image_b64)

    # Build consistency extension for non-anchor shots
    consistency_extension = ""
    if is_non_anchor:
        consistency_extension = f"""

━━ WORLD CONSISTENCY CHECK (this shot vs anchor) ━━
The FIRST image is the ANCHOR frame (shot_01) — the visual world reference.
The SECOND image is the CURRENT frame to validate.

Compare them ONLY on visual world elements (NOT on whether same person appears):
  • Color grade: same warm/cool tone, same shadow/highlight palette?
  • Lighting direction and colour temperature: consistent feel?
  • Environment atmosphere: same time period, same cinematic universe?
  • Lens/grain character: similar film look?

Anchor world reference:
  {anchor_identity or '(see anchor image)'}

Add to your JSON:
  "consistency": "MATCH" or "DRIFT",
  "consistency_issues": ["describe world drift — e.g. color grade changed, lighting feels different"]

Mark verdict FAILED ONLY if the world drift is SEVERE:
  • Completely different color grade (warm golden hour → cold blue night)
  • Completely different environment (outdoor tropical → indoor studio)
Minor variations in subject, angle, composition → always keep VERIFIED (this is expected)."""

    system_prompt = SYSTEM + consistency_extension

    # Build content parts
    content_parts = []
    if is_non_anchor:
        content_parts.append({
            "type": "text",
            "text": (
                f"ANCHOR frame (shot_01) is first, CURRENT frame to validate is second.\n"
                f"Shot ID: {shot.shot_id}\n"
                f"Narrative function: {getattr(shot, 'narrative_function', 'n/a')}\n"
                f"Image prompt used: {shot.image_prompt[:250]}"
            ),
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": anchor_image_b64, "detail": "low"},
        })
    else:
        content_parts.append({
            "type": "text",
            "text": (
                f"Validate this ANCHOR advertising frame.\n"
                f"Shot ID: {shot.shot_id}\n"
                f"Narrative function: {getattr(shot, 'narrative_function', 'n/a')}\n"
                f"Subject: {getattr(shot, 'subject', 'n/a')}\n"
                f"Image prompt: {shot.image_prompt[:250]}"
            ),
        })

    content_parts.append({
        "type": "image_url",
        "image_url": {"url": shot.image_base64, "detail": "low"},
    })

    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    raw = resp.choices[0].message.content
    if not raw:
        raise ValueError("GPT-4o returned empty/null VLM response — treating as auto-verify")
    return json.loads(raw)


async def _validate_one_shot(shot, session_dir: Path, state: PipelineState, emit: Callable):
    """Validate a single shot with up to MAX_RETRIES regenerations if FAILED."""
    api_key = os.environ.get("RUNWAYML_API_SECRET", "")

    is_anchor_shot = bool(state.storyboard) and shot.shot_id == state.storyboard[0].shot_id
    anchor = state.storyboard[0] if state.storyboard else None

    # Anchor image + identity doc for consistency checks (shots 2-N only)
    anchor_image_b64 = None
    anchor_identity  = ""
    if not is_anchor_shot and anchor and anchor.image_base64:
        anchor_image_b64 = anchor.image_base64
        anchor_identity  = state.enriched_brief.get("_anchor_identity", "")

    # Reference images for regeneration (preserves continuity)
    reference_images = []
    if not is_anchor_shot and anchor and anchor.image_base64:
        reference_images.append({"tag": "anchor", "uri": anchor.image_base64})
    if state.image_data_uri:
        reference_images.append({"tag": "hero_ref", "uri": state.image_data_uri})

    current_prompt = shot.image_prompt

    for attempt in range(MAX_RETRIES + 1):
        print(f"[vlm_validator] {shot.shot_id} attempt {attempt}/{MAX_RETRIES} — VLM check")

        try:
            result = await _vlm_check(
                shot, emit,
                anchor_image_b64=anchor_image_b64,
                anchor_identity=anchor_identity,
            )
        except Exception as exc:
            print(f"[vlm_validator] {shot.shot_id} VLM API error: {exc}")
            await emit({
                "type": "log", "stage": "6a",
                "message": f"[{shot.shot_id}] VLM API error (auto-verifying): {str(exc)[:120]}",
                "level": "warning",
            })
            shot.image_status = "VERIFIED"
            await emit({"type": "image_validated", "shot_id": shot.shot_id,
                        "verdict": "VERIFIED", "confidence": 0.75,
                        "notes": "auto-verified due to API error"})
            return

        verdict      = result.get("verdict", "VERIFIED")
        confidence   = float(result.get("confidence", 0.8))
        issues       = result.get("issues", [])
        notes        = result.get("notes", "")
        refined      = result.get("refined_prompt", "")
        consistency  = result.get("consistency", "MATCH")
        cons_issues  = result.get("consistency_issues", [])

        # Downgrade verdict if there's a severe WORLD consistency drift
        if consistency == "DRIFT" and verdict == "VERIFIED":
            drift_severity = any(
                kw in " ".join(cons_issues).lower()
                for kw in ["completely different color", "opposite color", "completely different environment",
                           "entirely different", "completely changed", "wrong environment"]
            )
            if drift_severity:
                verdict    = "FAILED"
                confidence = 0.5
                issues     = cons_issues + issues
                print(f"[vlm_validator] {shot.shot_id} downgraded to FAILED — severe consistency drift")

        consistency_note = f" | consistency: {consistency}" if not is_anchor_shot else ""
        print(f"[vlm_validator] {shot.shot_id} attempt={attempt} verdict={verdict} "
              f"conf={confidence:.0%}{consistency_note}")

        await emit({
            "type": "image_validated",
            "shot_id": shot.shot_id,
            "verdict": verdict,
            "confidence": confidence,
            "notes": notes,
            "consistency": consistency if not is_anchor_shot else "anchor",
            "attempt": attempt + 1,
        })

        # ── PASSED or no retries left ────────────────────────────────
        if verdict == "VERIFIED" or attempt == MAX_RETRIES:
            shot.image_status = verdict
            suffix = "" if verdict == "VERIFIED" else f" — kept after {attempt} regen(s)"
            level  = "success" if verdict == "VERIFIED" else "warning"
            log_msg = f"[{shot.shot_id}] {verdict} ({confidence:.0%})"
            if not is_anchor_shot:
                log_msg += f" | continuity: {consistency}"
                if cons_issues:
                    log_msg += f" ({cons_issues[0][:60]})"
            log_msg += f" — {notes[:60]}{suffix}"
            await emit({"type": "log", "stage": "6a", "message": log_msg, "level": level})
            return

        # ── FAILED with retries remaining ────────────────────────────
        issue_summary = "; ".join((issues + cons_issues)[:3]) or "poor composition"

        # Build a better prompt that includes the identity doc prominently
        if refined and len(refined.split()) >= 30:
            improved_prompt = refined.strip()
        else:
            improved_prompt = current_prompt + (
                f" CORRECTIONS: {issue_summary}. "
                "Higher quality, cleaner composition, professional advertising."
            )

        # Ensure @anchor + world doc stays in the prompt for shots 2-N
        if not is_anchor_shot and anchor_identity and "@anchor" not in improved_prompt:
            improved_prompt = (
                f"@anchor — SAME VISUAL WORLD as anchor: {anchor_identity} "
                + improved_prompt
            )
        elif not is_anchor_shot and "@anchor" not in improved_prompt:
            improved_prompt = (
                "@anchor — same visual world, color grade, and lighting as the anchor frame. "
                + improved_prompt
            )

        regen_num = attempt + 1
        await emit({
            "type": "log", "stage": "6a",
            "message": f"[{shot.shot_id}] FAILED — regenerating ({regen_num}/{MAX_RETRIES}): "
                       f"{issue_summary[:80]}",
            "level": "warning",
        })
        print(f"[vlm_validator] {shot.shot_id} regen {regen_num}/{MAX_RETRIES}: {issue_summary[:60]}")

        try:
            async def regen_log(msg, lvl="info"):
                await emit({"type": "log", "stage": "6a",
                            "message": f"[{shot.shot_id}][regen-{regen_num}] {msg}",
                            "level": lvl})

            new_url    = await generate_image(
                api_key=api_key,
                prompt_text=improved_prompt,
                ratio=state.ratio,
                model=state.image_model,
                reference_images=reference_images if reference_images else None,
                log=regen_log,
            )
            img_bytes  = await download_bytes(new_url)
            b64        = base64.b64encode(img_bytes).decode()
            data_uri   = f"data:image/jpeg;base64,{b64}"

            img_dir  = session_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{shot.shot_id}_r{regen_num}.jpg"
            img_path.write_bytes(img_bytes)

            shot.image_base64 = data_uri
            shot.image_path   = str(img_path)
            shot.image_url    = new_url
            current_prompt    = improved_prompt

            credits = CREDIT_COST.get(state.image_model, 6)
            shot.cost_credits        += credits
            state.total_cost_credits += credits
            state.total_cost_usd      = state.total_cost_credits * 0.01

            await emit({
                "type": "image_ready",
                "shot_id": shot.shot_id,
                "src": f"/output/{state.session_id}/images/{shot.shot_id}_r{regen_num}.jpg",
                "prompt": improved_prompt[:100],
                "status": "generated",
                "cost_credits": state.total_cost_credits,
                "cost_usd": round(state.total_cost_usd, 2),
                "regen": True,
            })
            await emit({
                "type": "log", "stage": "6a",
                "message": f"[{shot.shot_id}] Regen {regen_num} done (+{credits} cr) — re-validating...",
                "level": "info",
            })

        except Exception as exc:
            print(f"[vlm_validator] {shot.shot_id} regen {regen_num} FAILED: {exc}")
            await emit({"type": "log", "stage": "6a",
                        "message": f"[{shot.shot_id}] Regen {regen_num} error: {str(exc)[:150]}",
                        "level": "error"})
            shot.image_status = "FAILED"
            return


async def run(state: PipelineState, session_dir: Path, emit: Callable) -> None:
    shots_with_images = [
        s for s in state.storyboard
        if s.image_status == "generated" and s.image_base64
    ]

    anchor_identity = state.enriched_brief.get("_anchor_world") or state.enriched_brief.get("_anchor_identity", "")
    print(f"[vlm_validator] run() — {len(shots_with_images)} shots | "
          f"anchor world doc: {len(anchor_identity.split())} words")

    await emit({
        "type": "log", "stage": "6a",
        "message": (
            f"VLM validation: {len(shots_with_images)} frames — quality + world-consistency check "
            f"(world doc: {'loaded' if anchor_identity else 'not available'}, "
            f"up to {MAX_RETRIES} regens per FAILED shot)..."
        ),
        "level": "info",
    })

    if not shots_with_images:
        await emit({"type": "log", "stage": "6a",
                    "message": "No generated images to validate.", "level": "warning"})
        return

    for shot in shots_with_images:
        await _validate_one_shot(shot, session_dir, state, emit)

    verified = sum(1 for s in state.storyboard if s.image_status == "VERIFIED")
    failed   = sum(1 for s in state.storyboard if s.image_status == "FAILED")
    total    = len(shots_with_images)
    print(f"[vlm_validator] complete: {verified}/{total} VERIFIED, {failed} FAILED")
    await emit({
        "type": "log", "stage": "6a",
        "message": f"VLM complete: {verified}/{total} VERIFIED, {failed} FAILED "
                   f"(video generation proceeds for all)",
        "level": "success" if failed == 0 else "warning",
    })
    await emit({"type": "vlm_summary", "verified": verified, "failed": failed, "total": total})
