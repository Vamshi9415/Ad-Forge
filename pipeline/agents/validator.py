"""Stage 5: Structural Logic and Syntactic Validation."""

import json
import os
from typing import Callable

from openai import AsyncOpenAI

from pipeline.state import PipelineState


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


VALID_DURATIONS = {5, 8, 10}

SYSTEM = """You are a senior QA engineer and prompt specialist for AI video generation pipelines.
You validate and elevate prompts to ensure maximum quality from RunwayML Gen-4 models.

VALIDATION RULES (non-negotiable):
1. Duration: must be exactly 5, 8, or 10 — integers only, nothing else
2. Image prompts: minimum 60 words; must contain subject + environment + lighting + quality tag
3. Motion prompts: must describe concrete physical movement (camera + subject + atmosphere), minimum 20 words
4. @hero_ref usage: only permitted when reference_image_available = true
5. Camera values: strictly -10.0 to +10.0; avoid values >7 unless intentional dramatic motion
6. No negative framing: prompts must not say "no X" or "without X" — describe what IS there

PROMPT UPGRADE RULES (apply when correcting):
- Weak: "a person walks" → Strong: "a young professional in a tailored charcoal blazer strides purposefully forward"
- Weak: "nice lighting" → Strong: "warm Rembrandt key light from camera-right with cool fill creating a 3:1 ratio"
- Weak: "camera moves in" → Strong: "smooth 4-second dolly-forward on a 50mm lens, subject grows from medium to close-up"
- Weak: "the product appears" → Strong: "@hero_ref product placed prominently in foreground left, sharp focus, f/2.8"

Return ONLY valid JSON:
{
  "is_valid": bool,
  "issues": [
    {
      "shot_id": "shot_01",
      "field": "duration|image_prompt|motion_prompt|camera",
      "severity": "critical|warning",
      "problem": "specific description of the issue",
      "fix": "what was changed and why"
    }
  ],
  "corrected_shots": [
    {
      "shot_id": "shot_01",
      "image_prompt": "upgraded full prompt",
      "motion_prompt": "upgraded motion description",
      "duration": 5
    }
  ]
}"""


async def run(state: PipelineState, emit: Callable) -> None:
    await emit({"type": "log", "stage": "5", "message": "Validating prompts against API schema requirements...", "level": "info"})

    has_ref = bool(state.image_data_uri)
    visual_style = state.enriched_brief.get("visual_style", "cinematic")
    color_grade  = state.enriched_brief.get("color_grade", "")
    lighting     = state.enriched_brief.get("lighting_style", "")

    shots_payload = [
        {
            "shot_id": s.shot_id,
            "duration": s.duration,
            "image_prompt": s.image_prompt,
            "image_prompt_word_count": len(s.image_prompt.split()),
            "motion_prompt": s.motion_prompt,
            "motion_prompt_word_count": len(s.motion_prompt.split()),
            "camera": {
                "pan": s.camera.pan, "tilt": s.camera.tilt,
                "zoom": s.camera.zoom, "static": s.camera.static,
            },
            "uses_hero_ref": "@hero_ref" in s.image_prompt,
        }
        for s in state.storyboard
    ]

    prompt = f"""Validate and upgrade this storyboard for RunwayML Gen-4 API.

CONTEXT:
- Reference image available: {has_ref}
- Visual style: {visual_style}
- Color grade: {color_grade}
- Lighting style: {lighting}
- Valid durations: [5, 8, 10] — integers only, no exceptions
- Platform ratio: {state.ratio}

STORYBOARD TO VALIDATE:
{json.dumps(shots_payload, indent=2)}

For every shot that fails any rule: provide a corrected_shot with the fixed/upgraded version.
Apply prompt upgrade rules even to technically valid prompts if they are weak or generic.
Only include a shot in corrected_shots if something actually changed."""

    resp = await _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=3000,
    )

    result = json.loads(resp.choices[0].message.content)
    issues   = result.get("issues", [])
    corrected = result.get("corrected_shots", [])

    if issues:
        critical = [i for i in issues if i.get("severity") == "critical"]
        warnings = [i for i in issues if i.get("severity") != "critical"]
        await emit({
            "type": "log", "stage": "5",
            "message": f"Found {len(issues)} issue(s) ({len(critical)} critical, {len(warnings)} warnings). Applying corrections...",
            "level": "warning",
        })
        corrections_map = {c["shot_id"]: c for c in corrected}
        for shot in state.storyboard:
            correction = corrections_map.get(shot.shot_id)
            if correction:
                if "image_prompt" in correction and correction["image_prompt"]:
                    shot.image_prompt = correction["image_prompt"]
                if "motion_prompt" in correction and correction["motion_prompt"]:
                    shot.motion_prompt = correction["motion_prompt"]
                if "duration" in correction and int(correction["duration"]) in VALID_DURATIONS:
                    shot.duration = int(correction["duration"])
    else:
        await emit({"type": "log", "stage": "5", "message": "All prompts passed validation — no issues detected", "level": "success"})

    # Hard guardrails — always enforced regardless of LLM output
    for shot in state.storyboard:
        if shot.duration not in VALID_DURATIONS:
            closest = min(VALID_DURATIONS, key=lambda d: abs(d - shot.duration))
            await emit({
                "type": "log", "stage": "5",
                "message": f"{shot.shot_id}: duration {shot.duration}s → clamped to {closest}s",
                "level": "warning",
            })
            shot.duration = closest

        if not has_ref and "@hero_ref" in shot.image_prompt:
            shot.image_prompt = shot.image_prompt.replace("@hero_ref", "").strip(", ").strip()

        # Clamp camera values
        for attr in ("pan", "tilt", "zoom", "roll"):
            val = getattr(shot.camera, attr)
            if not (-10.0 <= val <= 10.0):
                setattr(shot.camera, attr, max(-10.0, min(10.0, val)))

    await emit({
        "type": "log", "stage": "5",
        "message": f"Validation complete — {len(state.storyboard)} shots ready for synthesis",
        "level": "success",
    })
    await emit({"type": "stage_data", "stage": "5", "data": {"issues_found": len(issues), "corrected": len(corrected)}})
