"""Stage 3: Narrative Sequencing and Storyboarding."""

import json
import os
from typing import Callable

from openai import AsyncOpenAI

from pipeline.state import PipelineState, Shot, CameraParams


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


VALID_DURATIONS = [5, 8, 10]

SYSTEM = """You are an award-winning commercial film director (think Ridley Scott, Spike Jonze, Park Chan-wook for ads).
Design a precise, emotionally engineered shot-by-shot storyboard with strong STORY CONTINUITY.

━━ STORY CONTINUITY RULES ━━
Great ads feel like ONE continuous world, not a series of random beautiful images.
Achieve this by:

1. WORLD BIBLE — define a single visual universe all shots share:
   • One location family (e.g. "warm Bombay streets and interiors at golden hour")
   • One color palette locked across all shots
   • One recurring visual motif or prop that threads the story (e.g. the ice cream bowl, the coffee cup)
   • Consistent time of day and lighting direction

2. NARRATIVE THREAD — each shot must EARN its place by advancing the story:
   • What emotional state does this shot leave the viewer in?
   • What question does it pose that the NEXT shot answers?
   • Each shot ends on a visual or emotional beat that makes the audience want to see the next one.

3. TRANSITION LOGIC — write how each shot CONNECTS to the next:
   • Matching visual elements (shot ends on steam → next opens on steam)
   • Complementary camera moves (dolly-in ends → cut to static close-up)
   • Emotional escalation (quiet → tension → release → joy)

4. SHOT GRAMMAR — vary framing but maintain world coherence:
   • Do NOT repeat the same shot type twice in a row
   • Classic ad grammar: wide establish → medium reveal → close-up detail → insert → wide payoff

CRITICAL DURATION RULES:
- Each shot duration MUST be exactly 5, 8, or 10 (integers only)
- The sum of ALL shot durations MUST equal the total requested duration

Return ONLY a valid JSON object with a "shots" key:
{
  "shots": [
    {
      "shot_id": "shot_01",
      "narrative_function": "Hook — establish the longing/desire state",
      "duration": 10,
      "subject": "precise description of the main visual subject",
      "environment": "specific location, time of day, depth, atmosphere",
      "action": "physically observable action — steam rises, hand reaches, condensation drips",
      "emotional_beat": "what feeling this shot creates → what it makes the viewer want next",
      "transition_to_next": "what visual/emotional element bridges to the following shot"
    }
  ]
}"""


def _solve_duration(total: int) -> list[int]:
    """Find a valid shot duration breakdown summing to total."""
    solutions = []
    for n10 in range(total // 10 + 1):
        for n8 in range((total - n10 * 10) // 8 + 1):
            remainder = total - n10 * 10 - n8 * 8
            if remainder >= 0 and remainder % 5 == 0:
                n5 = remainder // 5
                durations = [10] * n10 + [8] * n8 + [5] * n5
                solutions.append(durations)
    if solutions:
        # Prefer 4-6 shots for best narrative pacing
        return min(solutions, key=lambda x: abs(len(x) - 5))
    # Fallback greedy
    shots, remaining = [], total
    while remaining >= 10:
        shots.append(10)
        remaining -= 10
    if remaining in (8, 5):
        shots.append(remaining)
    elif remaining > 0:
        shots.append(5)
    return shots


async def run(state: PipelineState, emit: Callable) -> None:
    await emit({"type": "log", "stage": "3", "message": f"Creating {state.total_duration}s narrative storyboard...", "level": "info"})

    valid_breakdown = _solve_duration(state.total_duration)
    num_shots = len(valid_breakdown)

    await emit({
        "type": "log", "stage": "3",
        "message": f"Optimal breakdown: {num_shots} shots × {valid_breakdown}s = {sum(valid_breakdown)}s total",
        "level": "info",
    })

    narrative_arc    = state.enriched_brief.get("narrative_arc", "Problem-Solution-CTA")
    visual_style     = state.enriched_brief.get("visual_style", "cinematic")
    color_grade      = state.enriched_brief.get("color_grade", "")
    lens_aesthetic   = state.enriched_brief.get("lens_aesthetic", "")
    key_scenes       = state.enriched_brief.get("key_scenes", [])
    product_rule     = state.enriched_brief.get("product_placement_rule", "Product appears in final third")
    shot_grammar     = state.enriched_brief.get("shot_grammar", "")
    transition_style = state.enriched_brief.get("transition_style", "hard cuts on beat")

    # If a reference image was uploaded, use it to anchor VISUAL STYLE only — not narrative
    image_analysis  = state.enriched_brief.get("_image_analysis", {})
    has_ref_image   = state.enriched_brief.get("has_reference_image", False)
    ref_style_notes = ""
    if has_ref_image and image_analysis:
        detected_style   = image_analysis.get("style", "")
        detected_palette = image_analysis.get("color_palette", "")
        detected_setting = image_analysis.get("setting", "")
        ref_style_notes = f"""
REFERENCE IMAGE — VISUAL STYLE ANCHOR (optional):
The user uploaded a reference image. Use it ONLY to anchor the visual world:
  • Style/aesthetic: {detected_style}
  • Color palette: {detected_palette}
  • Setting/world: {detected_setting}
The uploaded image informs the LOOK of the ad — not the narrative plot.
The story can show different subjects, environments, and moments as long as they feel like
they belong to the same visual world as the reference image.
"""

    prompt = f"""Create a {state.total_duration}-second advertisement storyboard that tells a STORY with clear visual continuity.

CREATIVE BRIEF:
{json.dumps({k: v for k, v in state.enriched_brief.items() if k in [
    'campaign_title','creative_direction','visual_style','color_grade',
    'lens_aesthetic','lighting_style','narrative_arc','product_placement_rule',
    'shot_grammar','key_scenes','emotional_tone','unique_selling_point',
]}, indent=2)}
{ref_style_notes}
MANDATORY SHOT DURATION SEQUENCE:
- Total shots: {num_shots}
- Duration per shot: {valid_breakdown}
- Assignment: {', '.join(f'shot_{i+1:02d}={d}s' for i, d in enumerate(valid_breakdown))}
- Sum check: {sum(valid_breakdown)}s = {state.total_duration}s ✓

NARRATIVE BLUEPRINT:
- Arc model: {narrative_arc}
- Visual style: {visual_style}{f' | Color grade: {color_grade}' if color_grade else ''}
- Lens aesthetic: {lens_aesthetic}
- Transition style: {transition_style}
- Product placement: {product_rule}

KEY SCENES TO INCORPORATE:
{chr(10).join(f'- {s}' for s in key_scenes[:5]) if key_scenes else '- Derive from the creative direction above'}

STORY CONTINUITY REQUIREMENTS:
1. Define ONE consistent visual world (environment family, time of day, color palette) for all shots.
2. Each shot must have a clear EMOTIONAL BEAT — what feeling it creates and what question it poses.
3. Each shot must have a TRANSITION NOTE — the specific visual or emotional element that bridges to the next shot.
4. Vary framing (wide → medium → close → insert) to create shot grammar.
5. Ensure the viewer can feel WHY each shot follows the previous one.

Build the emotional arc: {valid_breakdown[0]}s HOOK → develop tension → {valid_breakdown[-1]}s CTA CLOSE.
After export, a separate silent promo end card will overlay campaign title + primary CTA — keep the last shot visually brand-clear.

Return exactly {num_shots} shots with durations {valid_breakdown}.
Each shot must include: shot_id, narrative_function, duration, subject, environment, action, emotional_beat, transition_to_next."""

    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=3000,
    )

    raw = json.loads(resp.choices[0].message.content)
    if isinstance(raw, list):
        shots_data = raw
    elif isinstance(raw, dict):
        shots_data = (raw.get("shots") or raw.get("storyboard") or raw.get("scenes")
                      or raw.get("ad_shots") or next(
                          (v for v in raw.values() if isinstance(v, list) and len(v) > 0), []))
    else:
        shots_data = []

    shots = []
    for i, (shot_dict, dur) in enumerate(zip(shots_data, valid_breakdown)):
        shot = Shot(
            shot_id=f"shot_{i+1:02d}",
            narrative_function=shot_dict.get("narrative_function", f"Scene {i+1}"),
            duration=dur,
            subject=shot_dict.get("subject", ""),
            environment=shot_dict.get("environment", ""),
            action=shot_dict.get("action", ""),
            emotional_beat=shot_dict.get("emotional_beat", ""),
            transition_to_next=shot_dict.get("transition_to_next", ""),
        )
        shots.append(shot)

    state.storyboard = shots

    # Store the world-level continuity context for prompt_gen to use
    if shots_data and isinstance(shots_data[0], dict):
        state.enriched_brief["_world_bible"] = shots_data[0].get("world_bible", "")

    await emit({
        "type": "log", "stage": "3",
        "message": f"Storyboard complete: {len(shots)} shots, {sum(s.duration for s in shots)}s total runtime",
        "level": "success",
    })

    storyboard_summary = [
        {
            "shot_id": s.shot_id, "duration": s.duration,
            "function": s.narrative_function, "subject": s.subject,
            "transition_to_next": s.transition_to_next,
        }
        for s in shots
    ]
    await emit({"type": "stage_data", "stage": "3", "data": {"storyboard": storyboard_summary}})
