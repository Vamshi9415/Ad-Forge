"""Stage 1: Intent Ingestion and Modality Normalization.

When a reference image is provided, it is treated as the PRIMARY creative input —
the story, characters, and visual world are built AROUND what's in the image.
The user's text prompt supplements and directs the narrative, but the image subject
is always the hero of the ad.
"""

import json
import os
from typing import Callable, Optional

from openai import AsyncOpenAI

from pipeline.state import PipelineState, PLATFORM_RATIOS


def _get_client():
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Image subject analysis ────────────────────────────────────────────────

IMAGE_ANALYSIS_SYSTEM = """You are a senior advertising creative director analyzing a reference image.
The user has uploaded this image and wants the ENTIRE AD to be built around what is shown in it.
Your job is to extract everything important so the story can revolve around THIS specific subject.

Analyze the image and return ONLY valid JSON:
{
  "subject_type": "person" | "product" | "place" | "scene" | "animal" | "abstract",
  "subject_description": "precise physical description of the main subject (80-120 words) — gender, age, appearance, clothing, expression for people; shape, colour, material, label for products; architecture, environment for places",
  "subject_identity": "one-sentence summary, e.g. 'A young South Indian woman in a red saree holding a ceramic cup'",
  "detected_product": "name/type of any product visible, or null",
  "detected_brand": "any visible brand name or logo, or null",
  "setting": "where the image is set — indoor/outdoor, location type, time of day",
  "visual_style": "photographic style: documentary | studio | lifestyle | cinematic | editorial | product-shot",
  "color_palette": ["3-5 dominant colours as descriptive names, e.g. 'deep crimson', 'warm ivory'"],
  "mood": "the emotional feel of the image in 3-5 words",
  "story_hooks": ["2-3 narrative directions this image could anchor — what stories could revolve around this subject"],
  "must_include_in_every_shot": "what must appear in EVERY shot of the ad to maintain identity (e.g. 'the woman in the red saree', 'the ceramic coffee cup')"
}"""


async def _analyze_reference_image(image_data_uri: str, emit: Callable) -> Optional[dict]:
    """Run GPT-4o Vision on the uploaded image to extract subject/story information.

    Returns a dict with the analysis, or None if the API call fails.
    The result drives the entire downstream pipeline — the story will be
    built AROUND this subject, not alongside it.
    """
    await emit({"type": "log", "stage": "1",
                "message": "Analyzing reference image — extracting subject and story potential...",
                "level": "info"})
    try:
        resp = await _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Analyze this reference image. The entire ad will be built around what you see here."},
                        {"type": "image_url",
                         "image_url": {"url": image_data_uri, "detail": "high"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        analysis = json.loads(resp.choices[0].message.content or "{}")
        subject_id = analysis.get("subject_identity", "the subject in the reference image")
        await emit({"type": "log", "stage": "1",
                    "message": f"Image subject: {subject_id}",
                    "level": "success"})
        return analysis
    except Exception as exc:
        await emit({"type": "log", "stage": "1",
                    "message": f"Image analysis failed ({str(exc)[:80]}) — using text prompt only",
                    "level": "warning"})
        return None


# ── Brief extraction ──────────────────────────────────────────────────────

SYSTEM_TEXT_ONLY = """You are a senior advertising creative strategist and brand consultant.
Extract a rich, structured creative brief from a raw advertising concept.
Think like a combination of creative director + media strategist + brand analyst.

Return ONLY valid JSON — be exhaustive in every field:
{
  "product_name": "exact brand/product name",
  "product_category": "specific category (e.g. 'Cold Brew Coffee', 'SaaS Productivity Tool')",
  "emotional_tone": "primary emotion to evoke (e.g. 'empowering curiosity', 'serene confidence')",
  "secondary_tone": "supporting emotional layer",
  "primary_cta": "verbatim call-to-action text",
  "key_visual_elements": ["3-6 specific objects, textures, settings that must appear visually"],
  "visual_metaphors": ["2-3 symbolic visual ideas that represent the brand promise"],
  "target_audience": "highly specific demographic + psychographic description",
  "brand_personality": "3 brand adjectives forming a personality archetype",
  "unique_selling_point": "single most differentiating product claim",
  "competitive_context": "what this brand does differently from category norm",
  "desired_feeling": "what a viewer should feel 5 seconds after the ad ends",
  "forbidden_elements": ["visuals or themes to avoid — infer from context"],
  "reference_image_notes": null
}"""

SYSTEM_IMAGE_FIRST = """You are a senior advertising creative strategist.
The user has uploaded a reference image AND written a concept text.
The IMAGE IS THE STAR — build the entire creative brief so the story revolves around
the specific subject shown in the image. The text prompt provides direction and intent;
the image provides the hero character/product/scene that EVERY shot must feature.

Return ONLY valid JSON:
{
  "product_name": "exact brand/product name — infer from image if visible",
  "product_category": "specific category",
  "emotional_tone": "primary emotion to evoke",
  "secondary_tone": "supporting emotional layer",
  "primary_cta": "verbatim call-to-action",
  "key_visual_elements": ["must include the reference image subject as element #1, then 3-5 more"],
  "visual_metaphors": ["2-3 symbolic ideas anchored to the image subject"],
  "target_audience": "highly specific demographic + psychographic",
  "brand_personality": "3 adjectives",
  "unique_selling_point": "single differentiating claim",
  "competitive_context": "what this brand does differently",
  "desired_feeling": "what viewer feels 5 seconds after the ad ends",
  "forbidden_elements": ["visuals/themes to avoid"],
  "reference_image_notes": "detailed description of the image subject — this will appear in EVERY shot of the ad",
  "image_subject_must_appear": "explicit instruction for every shot, e.g. 'The woman in the red saree must appear in every single shot'"
}"""


async def run(state: PipelineState, emit: Callable) -> None:
    await emit({"type": "log", "stage": "1",
                "message": "Parsing user intent and normalizing inputs...", "level": "info"})

    # ── Step 1: Analyze the reference image (if provided) ─────────────────
    image_analysis: Optional[dict] = None
    if state.image_data_uri:
        image_analysis = await _analyze_reference_image(state.image_data_uri, emit)
        # Store on state so all downstream agents can access it
        if image_analysis:
            state.enriched_brief["_image_analysis"] = image_analysis

    # ── Step 2: Build creative brief (image-first or text-only) ──────────
    if state.image_data_uri and image_analysis:
        system_prompt = SYSTEM_IMAGE_FIRST
        subject_desc  = image_analysis.get("subject_description", "")
        subject_id    = image_analysis.get("subject_identity", "the uploaded subject")
        must_include  = image_analysis.get("must_include_in_every_shot", subject_id)
        story_hooks   = image_analysis.get("story_hooks", [])
        hooks_str     = "\n".join(f"  • {h}" for h in story_hooks) if story_hooks else ""

        user_content = [
            {
                "type": "text",
                "text": (
                    f"Build a creative brief where THE IMAGE IS THE HERO of the story.\n\n"
                    f"USER CONCEPT:\n{state.original_prompt}\n\n"
                    f"REFERENCE IMAGE SUBJECT (already analyzed):\n"
                    f"  Identity: {subject_id}\n"
                    f"  Description: {subject_desc}\n"
                    f"  Must appear in every shot: {must_include}\n"
                    f"  Possible story directions:\n{hooks_str}\n\n"
                    f"IMPORTANT: The story must be built AROUND this specific subject. "
                    f"Every shot in the ad must feature or revolve around: {must_include}. "
                    f"The user's concept text tells you the DIRECTION and PURPOSE of the ad; "
                    f"the image subject IS the character/product/hero."
                ),
            },
            {"type": "image_url", "image_url": {"url": state.image_data_uri, "detail": "high"}},
        ]
    elif state.image_data_uri:
        # Image provided but analysis failed — fall back to style-anchor approach
        system_prompt = SYSTEM_IMAGE_FIRST
        user_content  = [
            {
                "type": "text",
                "text": (
                    f"Build a creative brief where the uploaded image is the HERO of the story.\n\n"
                    f"USER CONCEPT:\n{state.original_prompt}\n\n"
                    f"The story must revolve around whoever/whatever is in this image. "
                    f"Every shot should feature the main subject from this image."
                ),
            },
            {"type": "image_url", "image_url": {"url": state.image_data_uri, "detail": "high"}},
        ]
    else:
        system_prompt = SYSTEM_TEXT_ONLY
        user_content  = (
            f"Extract a full creative brief from this advertising concept.\n\n"
            f"CONCEPT:\n{state.original_prompt}\n\n"
            f"Be specific — infer visual style, forbidden clichés, and emotional subtext even if not stated."
        )

    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
    )

    parsed = json.loads(resp.choices[0].message.content)
    state.enriched_brief.update(parsed)
    state.enriched_brief.update({
        "platform":            state.platform,
        "ratio":               PLATFORM_RATIOS.get(state.platform, "1280:720"),
        "quality_mode":        state.quality_mode,
        "total_duration":      state.total_duration,
        "has_reference_image": bool(state.image_data_uri),
    })
    state.ratio = state.enriched_brief["ratio"]

    mode = "image-first" if state.image_data_uri else "text-only"
    await emit({
        "type": "log", "stage": "1",
        "message": (
            f"Identified [{mode}]: {parsed.get('product_name','Product')} | "
            f"Tone: {parsed.get('emotional_tone','N/A')} | "
            f"CTA: {parsed.get('primary_cta','N/A')}"
            + (f" | Hero: {image_analysis.get('subject_identity','')}" if image_analysis else "")
        ),
        "level": "success",
    })
    await emit({"type": "stage_data", "stage": "1", "data": parsed})
