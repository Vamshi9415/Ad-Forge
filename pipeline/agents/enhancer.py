"""Stage 2: Contextual Enhancement and Creative Brief Generation."""

import json
import os
from typing import Callable

from openai import AsyncOpenAI

from pipeline.state import PipelineState


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SYSTEM = """You are a world-class advertising creative director, cinematographer, and media strategist.
Synthesize the parsed brief into a comprehensive, production-ready creative manifesto.
Think: David Fincher's visual precision + Wieden+Kennedy's emotional storytelling + MPC's technical craft.

Return ONLY valid JSON. Every field drives downstream visual generation — be cinematographically precise:
{
  "campaign_title": "punchy, memorable campaign name",
  "creative_direction": "2-sentence directorial vision that defines the entire visual world",
  "color_palette": ["4-6 entries — mix hex codes AND descriptive names, e.g. '#1a1a2e Deep Navy', 'Warm Amber #F59E0B'"],
  "color_grade": "specific grading style, e.g. 'Teal-Orange Hollywood blockbuster', 'Desaturated Scandinavian editorial', 'Warm analog film'",
  "visual_style": "one of: cinematic | documentary | editorial | minimalist | vibrant | dark-luxury | lo-fi | hyperreal",
  "lighting_style": "specific lighting setup, e.g. 'Rembrandt lighting with warm practicals', 'Overcast diffuse + rim light', 'Neon-backlit night scene'",
  "lens_aesthetic": "focal length & bokeh style, e.g. '35mm anamorphic with characteristic oval bokeh and lens flares', '85mm portrait f/1.4 shallow DOF'",
  "pacing": "fast-cut | moderate | slow-burn — with bpm equivalent",
  "music_mood": "specific genre + tempo descriptor, e.g. 'Cinematic hip-hop 90bpm with strings', 'Post-rock crescendo'",
  "platform_specs": {
    "aspect_ratio": "e.g. 16:9",
    "safe_zone": "e.g. center 80% for UI overlays",
    "pacing_norms": "platform-specific edit rhythm guidance"
  },
  "shot_grammar": "cinematographic language: how this campaign uses close-ups, wide shots, cuts",
  "demographic_insights": "specific psychographic + behavioral description of ideal viewer",
  "narrative_arc": "exact storytelling model, e.g. 'Problem-Agitate-Solve', 'Day-in-life aspirational', 'Before-After-Bridge'",
  "brand_safety_notes": "what to avoid — explicit visual guardrails",
  "key_scenes": ["5-7 specific scene ideas — each a complete visual sentence"],
  "transition_style": "how shots connect, e.g. 'hard cuts on beat', 'dissolves for time-passing', 'whip-pans for energy'",
  "product_placement_rule": "how and when the product appears — hero shot rules"
}"""


PLATFORM_CONTEXT = {
    "youtube":          "Landscape 16:9, 15-60s pre-roll, storytelling-first, moderate pacing, 3-second hook rule",
    "tiktok":           "Vertical 9:16, 15-30s, aggressive hook in 1s, fast-cut, authentic lo-fi aesthetic welcome, trending audio-driven",
    "instagram_reels":  "Vertical 9:16, 15-30s, visually arresting first frame, text overlays common, lifestyle and aesthetic-forward",
    "linkedin":         "Landscape 16:9, 30-60s, professional B2B, slower deliberate pacing, authority signaling",
    "twitter":          "Landscape 16:9, 6-30s, punchy, direct, news-feed scroll-stop, bold typography moments",
    "instagram_square": "Square 1:1, 15-60s, balanced center composition, no edges for UI chrome",
}


async def run(state: PipelineState, emit: Callable) -> None:
    await emit({"type": "log", "stage": "2", "message": "Enriching brief with platform specs, trends, and demographics...", "level": "info"})

    platform_hint = PLATFORM_CONTEXT.get(state.platform, "Standard 16:9 video content")

    prompt = f"""Develop a comprehensive creative manifesto for this campaign.

PARSED BRIEF:
{json.dumps(state.enriched_brief, indent=2)}

PLATFORM: {state.platform}
PLATFORM CONTEXT: {platform_hint}
DURATION: {state.total_duration}s
QUALITY MODE: {'Broadcast — premium gen4.5 model, photorealistic output' if state.quality_mode == 'broadcast' else 'Preview — gen4_turbo model, animatic/concept quality'}
REFERENCE IMAGE: {'YES — visual style must anchor to reference image aesthetics' if state.enriched_brief.get('has_reference_image') else 'NO — define the visual world from scratch'}

Your manifesto will directly drive RunwayML image and video generation.
Be specific enough that a cinematographer could shoot from your description alone."""

    resp = await _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1500,
    )

    brief = json.loads(resp.choices[0].message.content)
    state.enriched_brief.update(brief)

    await emit({
        "type": "log", "stage": "2",
        "message": f"Campaign: '{brief.get('campaign_title','Untitled')}' | Style: {brief.get('visual_style','N/A')} | Pacing: {brief.get('pacing','N/A')}",
        "level": "success",
    })
    await emit({"type": "stage_data", "stage": "2", "data": brief})
