"""Stage 4: Programmatic Prompt Engineering and Camera Choreography."""

import json
import os
import random
from typing import Callable

from openai import AsyncOpenAI

from pipeline.state import PipelineState, CameraParams


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SYSTEM = """You are a master AI prompt engineer specialising in RunwayML Gen-4 image and video generation.
Your prompts produce broadcast-quality, photorealistic advertising imagery where every shot
FEELS LIKE IT BELONGS TO THE SAME FILM — through consistent visual world, color grade, lighting,
and narrative purpose. Each shot must logically follow the previous one and set up the next.

━━ VISUAL WORLD LOCK (ABSOLUTE, NON-NEGOTIABLE) ━━
Shot_01 is the ANCHOR that establishes the visual universe. Its rendered image feeds back into
every subsequent shot via the @anchor tag. To lock the WORLD at the TEXT level:

1. Shot_01: establish the complete WORLD IDENTITY — environment family, color grade, lighting
   signature, lens character. This is the "look bible" for the entire spot.

2. Shots 2-N: START with "@anchor — same visual world, same color grade, same lighting signature."
   Then describe the NEW content (new subject, new environment beat, new action) while explicitly
   restating the locked world elements (grade name, lighting direction, lens choice).

3. LOCKED across all shots (must be verbatim repeated):
   • Color grade: exact descriptive name (e.g. "warm teal-and-amber with lifted shadows")
   • Lighting signature: direction + quality + temperature (e.g. "warm tungsten key right at 45°")
   • Lens character: focal length, aperture, bokeh quality

4. ALLOWED to vary — this is what creates the STORY:
   • Subjects (different people, objects, or environments per shot is fine)
   • Shot framing (ECU → WS → MCU → insert — vary every shot)
   • Location within the world (same city/time period, different specific spots)
   • Action and emotional beat

5. NARRATIVE AWARENESS — each prompt must SHOW what connects this shot to the story:
   • Reference the story context (e.g. "continuing the unboxing ritual from the previous shot")
   • Include a transition HINT so the motion ends in a way that leads to the next shot

━━ IMAGE PROMPT FORMULA — 9 LAYERS ━━
Layer 0  WORLD TAG       — "@anchor — [world restatement]" (shots 2-N only)
Layer 1  SHOT TYPE       — exact: "ECU", "MCU", "MS", "MWS", "WS", "XWS", "OTS"
                           + composition principle (rule of thirds, leading lines, negative space)
Layer 2  SUBJECT         — who/what is the visual subject of THIS shot (can differ between shots)
                           describe precisely: gender/age/identity OR object description
Layer 3  ENVIRONMENT     — specific location, time of day, weather, depth, foreground/background
Layer 4  LIGHTING        — direction, quality, temperature, fill ratio, practicals, shadows
Layer 5  LENS            — focal length, aperture, focus point, bokeh character, any artefacts
Layer 6  COLOR GRADE     — SAME grade name as shot_01, shadow/highlight hex values, split tone
Layer 7  STORY CONTEXT   — 1 sentence: what moment in the narrative this shot depicts
Layer 8  QUALITY TAGS    — "photorealistic 8K", "film grain", "anamorphic squeeze"

EXEMPLARY ANCHOR (shot_01) PROMPT:
"WS establishing shot of a rain-wet Bombay street at golden hour — bicycle rickshaws, flower
stalls with marigold garlands blurring past, warm amber practical lights beginning to glow.
Time of day: last light at 6pm, sun at horizon. Key light: warm golden hour from camera-left,
long horizontal shadows across wet cobblestones. 24mm f/2.8, deep focus, background bokeh hints
at crowds. Color grade: warm teal-and-amber, lifted shadows to deep ochre #5C3A1E, highlights
cream-warm #FFF8F0. Film grain, photorealistic 8K, anamorphic squeeze. Story context: opening
world-establishing hook that sets the scene of the city coming alive."

EXEMPLARY CONTINUITY (shot_02) PROMPT:
"@anchor — same visual world, same warm teal-and-amber color grade, same golden-hour lighting.
MCU of a street vendor's weathered hands arranging rose petals in a ceramic bowl, warm ochre
light pooling across his knuckles. Same golden hour from camera-left, 85mm f/1.4, focus on
the bowl's rim, background stall in smooth oval bokeh. Color grade: warm teal-and-amber,
lifted shadows #5C3A1E, highlights #FFF8F0. Story context: close-up reveal of the product in
its natural habitat, picking up the texture and detail promised by the wide shot.
Film grain, photorealistic 8K, anamorphic squeeze."

BAD PROMPT (never do this): "Beautiful shot of something in a nice setting"

━━ MOTION PROMPT FORMULA — 5 ELEMENTS ━━
Every motion prompt must specify ALL of:
  1. CAMERA MOVEMENT — axis, direction, speed, distance, duration
  2. SUBJECT MOVEMENT — body/object action, direction, speed
  3. ATMOSPHERIC — environmental animation (steam, dust, fabric, water, crowd movement)
  4. TIMING — "over the first 3 of 5 seconds", "last 2 seconds static"
  5. TRANSITION HINT — how this shot ENDS to lead naturally into the next one

EXEMPLARY MOTION PROMPT:
"Slow dolly-forward 0.4m over 4 seconds from WS to MS; the vendor's hands continue arranging
petals in slow deliberate movements; warm dust motes drift upward through the golden light;
over the final second the camera settles static on the bowl — hard cut to the ECU that follows."

BAD MOTION PROMPT: "The scene shows movement and energy"

━━ CAMERA PARAMETERS ━━
Scale: -10.0 (max negative) to +10.0 (max positive). Subtle: 1-3. Dramatic: 4-7. Extreme: 8-10.
- pan:  negative=pan left,  positive=pan right
- tilt: negative=tilt down, positive=tilt up
- zoom: negative=pull back, positive=push in (dolly/zoom)
- roll: rotation around optical axis (very rarely >2.0)
- static: true = locked tripod shot

━━ PROMPT LENGTH RULES ━━
• image_prompt: TARGET 140-160 words. Hard maximum: 950 characters (RunwayML API cap is 1000).
  Put @anchor tag + subject + environment + lighting + color grade FIRST.
  Put quality tags (photorealistic 8K, film grain, etc.) LAST — they will be trimmed if too long.
• motion_prompt: TARGET 60-80 words. Hard maximum: 950 characters.
• Shot framing MUST vary every single shot (no two consecutive shots with same framing).
• NEVER exceed 950 characters in image_prompt or motion_prompt.

Return a JSON object with a "prompts" key containing the array."""


async def run(state: PipelineState, emit: Callable) -> None:
    await emit({"type": "log", "stage": "4", "message": "Engineering cinematic prompts and camera choreography...", "level": "info"})

    has_ref = bool(state.image_data_uri)
    ref_note = (
        "A reference image has been uploaded. Append '@hero_ref' at the END of EVERY image_prompt "
        "to use it as a visual style reference."
        if has_ref else
        "No reference image — establish the visual world fully in text."
    )

    # Pull key cinematographic context from enriched brief
    campaign     = state.enriched_brief.get("campaign_title", "Ad Campaign")
    visual_style = state.enriched_brief.get("visual_style", "cinematic")
    color_grade  = state.enriched_brief.get("color_grade", "cinematic color grade")
    lighting     = state.enriched_brief.get("lighting_style", "natural lighting")
    lens         = state.enriched_brief.get("lens_aesthetic", "35mm cinematic lens")
    palette      = state.enriched_brief.get("color_palette", [])
    forbidden    = state.enriched_brief.get("forbidden_elements", [])
    product_rule = state.enriched_brief.get("product_placement_rule", "")
    usp          = state.enriched_brief.get("unique_selling_point", "")
    tone         = state.enriched_brief.get("emotional_tone", "")
    world_bible  = state.enriched_brief.get("_world_bible", "")

    palette_str   = ", ".join(palette) if isinstance(palette, list) else str(palette)
    forbidden_str = ", ".join(forbidden) if forbidden else "none specified"

    # Build an adjacency-aware storyboard: each shot knows prev/next
    shots_with_context = []
    for idx, s in enumerate(state.storyboard):
        prev_shot = state.storyboard[idx - 1] if idx > 0 else None
        next_shot = state.storyboard[idx + 1] if idx < len(state.storyboard) - 1 else None
        shots_with_context.append({
            "shot_id": s.shot_id,
            "position": f"shot {idx+1} of {len(state.storyboard)}",
            "narrative_function": s.narrative_function,
            "duration": s.duration,
            "subject": s.subject,
            "environment": s.environment,
            "action": s.action,
            "emotional_beat": s.emotional_beat,
            "transition_from_previous": prev_shot.transition_to_next if prev_shot else "OPENS the story — establish the world",
            "transition_to_next": s.transition_to_next if s.transition_to_next else ("CLOSES the story — final frame" if not next_shot else ""),
            "next_shot_preview": f"{next_shot.narrative_function}: {next_shot.subject}" if next_shot else "END OF SPOT",
        })

    storyboard_json = json.dumps(shots_with_context, indent=2)
    hero_ref_suffix = " @hero_ref" if has_ref else ""

    prompt = f"""Generate production-ready RunwayML prompts for this {len(state.storyboard)}-shot advertisement.
Shot_01 is the VISUAL WORLD ANCHOR — its rendered image feeds back into all other shots via @anchor
to lock the WORLD (color grade, lighting, environment feel) — not necessarily the same person/object.

━━ CAMPAIGN CONTEXT ━━
Campaign: {campaign}
Visual Style: {visual_style}
Color Grade to use in EVERY shot: {color_grade}
Lighting Signature to use in EVERY shot: {lighting}
Lens Aesthetic to use in EVERY shot: {lens}
Color Palette: {palette_str}
Emotional Tone: {tone}
USP to convey: {usp}
Product placement rule: {product_rule}
Forbidden elements: {forbidden_str}
Platform ratio: {state.ratio}
Reference image: {ref_note}
{f"World Bible from storyboard: {world_bible}" if world_bible else ""}

━━ STORYBOARD WITH ADJACENCY CONTEXT ━━
{storyboard_json}

━━ CONTINUITY RULES ━━
1. Shot_01's image_prompt: establish the WORLD — environment, color grade, lighting, lens. No @anchor.
2. Shots 2-N: START with "@anchor — same visual world, same {color_grade}, same {lighting}."
   Then describe the NEW scene content for THIS shot (new subject, moment, angle).
3. EVERY prompt must restate the color grade and lighting direction verbatim.
4. Motion prompts MUST end with a transition hint that matches the "transition_to_next" field.
5. Vary shot framing EVERY shot: no two consecutive shots with the same framing.

━━ DELIVERABLE FORMAT ━━
For EACH shot produce:
{{
  "shot_id": "shot_XX",
  "image_prompt": "<shot_01: world-establishing prompt | shots 2-N: @anchor + world restatement + new content>{hero_ref_suffix}",
  "motion_prompt": "camera movement + subject action + atmosphere + timing + TRANSITION HINT to next shot",
  "camera": {{
    "pan": <-10.0 to 10.0>, "tilt": <-10.0 to 10.0>,
    "zoom": <-10.0 to 10.0>, "roll": <-10.0 to 10.0>, "static": <true|false>
  }}
}}

QUALITY BAR:
• Each image_prompt: 140-160 words, MAX 950 characters. Put @anchor + core content FIRST, quality
  tags LAST. Lock color grade name and lighting direction in every prompt.
• Each motion_prompt: 60-80 words, MAX 950 characters. Include transition hint.
• Shot framing must vary every shot (WS → MCU → ECU → insert → WS etc).
Return {{"prompts": [...{len(state.storyboard)} objects...]}}"""

    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=7000,
    )

    raw = json.loads(resp.choices[0].message.content)
    if isinstance(raw, list):
        prompts = raw
    elif isinstance(raw, dict):
        prompts = (raw.get("prompts") or raw.get("shots") or raw.get("prompt_list")
                   or next((v for v in raw.values() if isinstance(v, list) and len(v) > 0), []))
    else:
        prompts = []

    prompt_map = {p["shot_id"]: p for p in prompts}

    for idx, shot in enumerate(state.storyboard):
        p = prompt_map.get(shot.shot_id, {})
        is_anchor = (idx == 0)
        world_anchor_prefix = (
            "" if is_anchor
            else f"@anchor — same visual world, same {color_grade}, same {lighting}. "
        )
        # Fallback: construct a rich prompt from storyboard data if missing
        fallback_img = (
            f"{world_anchor_prefix}{shot.narrative_function}. "
            f"{'WS' if idx == 0 else 'MCU'} of {shot.subject} in {shot.environment}, "
            f"{visual_style} style, {lighting}, {lens}, {color_grade}, "
            f"story context: {shot.emotional_beat}, "
            f"photorealistic 8K advertising photography{hero_ref_suffix}"
        )
        fallback_motion = (
            f"Slow {random.choice(['dolly forward','pan right','tilt up','static locked off'])} "
            f"as {shot.action}. Subtle atmospheric movement. "
            f"{shot.transition_to_next if shot.transition_to_next else 'Ends on a held static frame.'}"
        )
        chosen_img = p.get("image_prompt", fallback_img)
        # Belt-and-braces: enforce @anchor on shots 2..N if the LLM forgot
        if not is_anchor and "@anchor" not in chosen_img:
            chosen_img = world_anchor_prefix + chosen_img
        shot.image_prompt  = chosen_img
        shot.motion_prompt = p.get("motion_prompt", fallback_motion)

        cam = p.get("camera", {})
        if isinstance(cam, dict):
            shot.camera = CameraParams(
                pan=float(cam.get("pan", 0.0)),
                tilt=float(cam.get("tilt", 0.0)),
                zoom=float(cam.get("zoom", 0.0)),
                roll=float(cam.get("roll", 0.0)),
                static=bool(cam.get("static", False)),
            )
        shot.seed = random.randint(1, 4294967295)

    await emit({
        "type": "log", "stage": "4",
        "message": f"Generated {len(state.storyboard)} cinematic prompt sets with camera choreography",
        "level": "success",
    })

    preview = [{"shot_id": s.shot_id, "image_prompt": s.image_prompt[:120] + "..."} for s in state.storyboard]
    await emit({"type": "stage_data", "stage": "4", "data": {"prompts": preview}})
