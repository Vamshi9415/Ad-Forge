"""Test 04 — LLM Agent Chain: Stages 1-5 (real GPT-4o calls).

Uses session-level fixture (runs once per test session) to avoid redundant API calls.
Validates the output contracts of each stage deeply.
"""

import re
from pipeline.state import PipelineState, Shot

import pytest

pytestmark = [pytest.mark.api, pytest.mark.slow]


# ─── Stage 1: Ingestion ───────────────────────────────────────────────────
class TestIngestion:
    def test_enriched_brief_not_empty(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief, "enriched_brief is empty after Stage 1"

    def test_product_name_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("product_name"), f"product_name missing. Brief: {brief}"

    def test_emotional_tone_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("emotional_tone") or brief.get("tone"), (
            "emotional_tone/tone missing from brief"
        )

    def test_cta_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("primary_cta") or brief.get("cta"), "CTA missing"

    def test_brief_has_enough_fields(self, llm_state):
        """Optimized ingestion should return 8+ fields."""
        brief = llm_state["state"].enriched_brief
        assert len(brief) >= 5, f"Brief too sparse ({len(brief)} fields): {list(brief.keys())}"

    def test_stage_timing(self, llm_state):
        t = llm_state["timings"]["ingestion"]
        assert t < 30, f"Ingestion took {t}s — too slow"
        print(f"\n  Ingestion: {t}s")


# ─── Stage 2: Enhancement ─────────────────────────────────────────────────
class TestEnhancement:
    def test_campaign_title_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("campaign_title"), "campaign_title missing after enhancement"

    def test_visual_style_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("visual_style"), "visual_style missing"

    def test_pacing_set(self, llm_state):
        brief = llm_state["state"].enriched_brief
        assert brief.get("pacing"), "pacing missing"

    def test_brief_further_enriched(self, llm_state):
        """Enhancement should bring brief to 10+ fields."""
        brief = llm_state["state"].enriched_brief
        assert len(brief) >= 8, (
            f"Enhancement didn't enrich enough ({len(brief)} fields): {list(brief.keys())}"
        )

    def test_stage_timing(self, llm_state):
        t = llm_state["timings"]["enhancer"]
        assert t < 30, f"Enhancement took {t}s"
        print(f"\n  Enhancement: {t}s")


# ─── Stage 3: Storyboard ─────────────────────────────────────────────────
class TestStoryboard:
    def test_shots_generated(self, llm_state):
        shots = llm_state["state"].storyboard
        assert len(shots) > 0, "No shots in storyboard"

    def test_shot_ids_unique(self, llm_state):
        ids = [s.shot_id for s in llm_state["state"].storyboard]
        assert len(ids) == len(set(ids)), f"Duplicate shot IDs: {ids}"

    def test_shot_ids_sequential(self, llm_state):
        ids = [s.shot_id for s in llm_state["state"].storyboard]
        for i, sid in enumerate(ids):
            assert re.match(r"shot_\d+", sid), f"Bad shot_id format: {sid}"

    def test_all_durations_valid(self, llm_state):
        shots = llm_state["state"].storyboard
        bad = [s for s in shots if s.duration not in (5, 8, 10)]
        assert not bad, f"Shots with invalid durations: {[(s.shot_id, s.duration) for s in bad]}"

    def test_total_duration_matches(self, llm_state):
        state = llm_state["state"]
        total = sum(s.duration for s in state.storyboard)
        assert total == state.total_duration, (
            f"Storyboard total {total}s ≠ requested {state.total_duration}s"
        )

    def test_each_shot_has_narrative_function(self, llm_state):
        empty = [s for s in llm_state["state"].storyboard if not s.narrative_function]
        assert not empty, f"Shots missing narrative_function: {[s.shot_id for s in empty]}"

    def test_each_shot_has_subject(self, llm_state):
        empty = [s for s in llm_state["state"].storyboard if not s.subject]
        assert not empty, f"Shots missing subject: {[s.shot_id for s in empty]}"

    def test_stage_timing(self, llm_state):
        t = llm_state["timings"]["story"]
        assert t < 40, f"Story took {t}s"
        print(f"\n  Storyboard: {t}s | {len(llm_state['state'].storyboard)} shots")


# ─── Stage 4: Prompt Generation ──────────────────────────────────────────
class TestPromptGeneration:
    def test_all_shots_have_image_prompt(self, llm_state):
        empty = [s for s in llm_state["state"].storyboard if not s.image_prompt]
        assert not empty, f"Shots missing image_prompt: {[s.shot_id for s in empty]}"

    def test_all_shots_have_motion_prompt(self, llm_state):
        empty = [s for s in llm_state["state"].storyboard if not s.motion_prompt]
        assert not empty, f"Shots missing motion_prompt: {[s.shot_id for s in empty]}"

    def test_image_prompts_min_length(self, llm_state):
        """Image prompts must be substantial — at least 50 chars."""
        short = [
            s for s in llm_state["state"].storyboard
            if len(s.image_prompt) < 50
        ]
        assert not short, (
            f"Short image prompts: {[(s.shot_id, len(s.image_prompt)) for s in short]}"
        )

    def test_motion_prompts_min_length(self, llm_state):
        """Motion prompts must be meaningful — at least 20 chars."""
        short = [s for s in llm_state["state"].storyboard if len(s.motion_prompt) < 20]
        assert not short, (
            f"Short motion prompts: {[(s.shot_id, len(s.motion_prompt)) for s in short]}"
        )

    def test_camera_params_in_range(self, llm_state):
        """Camera values must be within RunwayML's ±10 range."""
        for s in llm_state["state"].storyboard:
            for field in ("pan", "tilt", "zoom", "roll"):
                val = getattr(s.camera, field)
                assert -10 <= val <= 10, (
                    f"{s.shot_id} camera.{field}={val} out of ±10 range"
                )

    def test_seeds_set(self, llm_state):
        missing = [s for s in llm_state["state"].storyboard if s.seed is None]
        assert not missing, f"Shots missing seed: {[s.shot_id for s in missing]}"

    def test_stage_timing(self, llm_state):
        t = llm_state["timings"]["prompt_gen"]
        assert t < 60, f"PromptGen took {t}s"
        print(f"\n  PromptGen: {t}s")

    def test_print_sample_prompts(self, llm_state):
        """Informational: print first two prompts (not an assertion)."""
        for s in llm_state["state"].storyboard[:2]:
            print(f"\n  {s.shot_id}:")
            print(f"    image:  {s.image_prompt[:120]}...")
            print(f"    motion: {s.motion_prompt[:80]}")
            print(f"    camera: pan={s.camera.pan}, tilt={s.camera.tilt}, zoom={s.camera.zoom}")


# ─── Stage 5: Validator ──────────────────────────────────────────────────
class TestValidator:
    def test_all_durations_still_valid(self, llm_state):
        bad = [s for s in llm_state["state"].storyboard if s.duration not in (5, 8, 10)]
        assert not bad, f"Validator changed durations to invalid values: {bad}"

    def test_all_prompts_still_non_empty(self, llm_state):
        empty = [s for s in llm_state["state"].storyboard if not s.image_prompt]
        assert not empty, "Validator wiped some image_prompts"

    def test_camera_values_still_in_range(self, llm_state):
        for s in llm_state["state"].storyboard:
            for field in ("pan", "tilt", "zoom", "roll"):
                val = getattr(s.camera, field)
                assert -10 <= val <= 10, (
                    f"Validator failed to clamp {s.shot_id} camera.{field}={val}"
                )

    def test_stage_timing(self, llm_state):
        t = llm_state["timings"]["validator"]
        assert t < 40, f"Validator took {t}s"
        print(f"\n  Validator: {t}s")

    def test_pipeline_total_timing(self, llm_state):
        total = sum(llm_state["timings"].values())
        assert total < 180, f"Entire LLM chain took {total:.1f}s — may be too slow"
        print(f"\n  Total LLM chain: {total:.1f}s")
        for stage, t in llm_state["timings"].items():
            print(f"    {stage}: {t}s")
