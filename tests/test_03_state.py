"""Test 03 — PipelineState, Shot, CameraParams, Constants.

Purely unit tests — no API calls or file I/O.
Traces every field, default, constraint, and constant in state.py.
"""

import pytest
from pipeline.state import (
    CameraParams, Shot, PipelineState,
    PLATFORM_RATIOS, CREDIT_COST,
)

pytestmark = pytest.mark.unit


# ─── CameraParams ─────────────────────────────────────────────────────────
class TestCameraParams:
    def test_defaults(self):
        cam = CameraParams()
        assert cam.pan  == 0.0
        assert cam.tilt == 0.0
        assert cam.zoom == 0.0
        assert cam.roll == 0.0
        assert cam.static is False

    def test_custom_values(self):
        cam = CameraParams(pan=2.5, tilt=-1.0, zoom=5.0, roll=0.5, static=True)
        assert cam.pan  == 2.5
        assert cam.tilt == -1.0
        assert cam.zoom == 5.0
        assert cam.roll == 0.5
        assert cam.static is True

    def test_float_fields(self):
        cam = CameraParams(pan=3, tilt=2)  # int → accepted
        assert isinstance(cam.pan, (int, float))
        assert isinstance(cam.tilt, (int, float))

    @pytest.mark.parametrize("field,value", [
        ("pan",  -10.0), ("pan",  10.0),
        ("tilt", -10.0), ("tilt", 10.0),
        ("zoom", -10.0), ("zoom", 10.0),
        ("roll", -10.0), ("roll", 10.0),
    ])
    def test_boundary_values_accepted(self, field, value):
        """Fields should accept ±10 (RunwayML range)."""
        cam = CameraParams(**{field: value})
        assert getattr(cam, field) == value


# ─── Shot ─────────────────────────────────────────────────────────────────
class TestShot:
    def test_minimal_creation(self):
        s = Shot(
            shot_id="shot_01",
            narrative_function="Establish environment",
            duration=5,
            subject="Coffee barista",
            environment="Dimly lit café",
            action="pouring coffee",
        )
        assert s.shot_id == "shot_01"
        assert s.duration == 5
        assert s.image_status == "pending"
        assert s.video_status == "pending"
        assert s.cost_credits == 0
        assert s.image_base64 is None
        assert s.video_path is None

    def test_default_camera(self):
        s = Shot("s1", "func", 5, "sub", "env", "act")
        assert isinstance(s.camera, CameraParams)
        assert s.camera.pan == 0.0

    def test_image_status_transitions(self):
        s = Shot("s1", "func", 5, "sub", "env", "act")
        for status in ("pending", "generating", "generated", "VERIFIED", "FAILED"):
            s.image_status = status
            assert s.image_status == status

    def test_video_status_transitions(self):
        s = Shot("s1", "func", 5, "sub", "env", "act")
        for status in ("pending", "generating", "complete", "failed"):
            s.video_status = status
            assert s.video_status == status

    @pytest.mark.parametrize("duration", [5, 8, 10])
    def test_valid_durations(self, duration):
        s = Shot("s1", "func", duration, "sub", "env", "act")
        assert s.duration == duration

    def test_cost_accumulates(self):
        s = Shot("s1", "func", 5, "sub", "env", "act")
        s.cost_credits += 6    # image
        s.cost_credits += 60   # 5s video
        assert s.cost_credits == 66

    def test_prompt_fields_default_empty(self):
        s = Shot("s1", "func", 5, "sub", "env", "act")
        assert s.image_prompt == ""
        assert s.motion_prompt == ""


# ─── PipelineState ────────────────────────────────────────────────────────
class TestPipelineState:
    def test_required_fields(self):
        ps = PipelineState(session_id="abc123", original_prompt="test prompt")
        assert ps.session_id == "abc123"
        assert ps.original_prompt == "test prompt"

    def test_defaults(self):
        ps = PipelineState(session_id="x", original_prompt="y")
        assert ps.platform == "youtube"
        assert ps.quality_mode == "preview"
        assert ps.total_duration == 30
        assert ps.ratio == "1280:720"
        assert ps.video_model == "gen4_turbo"
        assert ps.image_model == "gen4_image"
        assert ps.total_cost_credits == 0
        assert ps.total_cost_usd == 0.0
        assert ps.final_video_path is None
        assert ps.final_audio_video_path is None
        assert ps.voiceover_script is None
        assert ps.storyboard == []
        assert ps.enriched_brief == {}

    def test_cost_tracking(self):
        ps = PipelineState(session_id="x", original_prompt="y")
        ps.total_cost_credits = 100
        ps.total_cost_usd = ps.total_cost_credits * 0.01
        assert ps.total_cost_usd == 1.0

    def test_storyboard_accumulates(self):
        ps = PipelineState(session_id="x", original_prompt="y")
        for i in range(5):
            ps.storyboard.append(Shot(f"shot_0{i+1}", "f", 5, "s", "e", "a"))
        assert len(ps.storyboard) == 5

    def test_enriched_brief_arbitrary_keys(self):
        ps = PipelineState(session_id="x", original_prompt="y")
        ps.enriched_brief = {
            "campaign_title": "Test Campaign",
            "visual_style": "minimalist",
            "pacing": "moderate",
        }
        assert ps.enriched_brief["campaign_title"] == "Test Campaign"

    def test_new_audio_fields_exist(self):
        """Verify that audio fields added to state.py are present."""
        ps = PipelineState(session_id="x", original_prompt="y")
        assert hasattr(ps, "final_audio_video_path")
        assert hasattr(ps, "voiceover_script")
        ps.voiceover_script = "Slow is bold. Every drop counts."
        assert ps.voiceover_script is not None


# ─── PLATFORM_RATIOS ──────────────────────────────────────────────────────
class TestPlatformRatios:
    EXPECTED = {
        "youtube":          "1280:720",
        "instagram_reels":  "720:1280",
        "tiktok":           "720:1280",
        "linkedin":         "1280:720",
        "twitter":          "1280:720",
        "instagram_square": "1024:1024",
    }

    @pytest.mark.parametrize("platform,ratio", EXPECTED.items())
    def test_ratio_values(self, platform, ratio):
        assert PLATFORM_RATIOS[platform] == ratio

    def test_all_expected_platforms_present(self):
        for p in self.EXPECTED:
            assert p in PLATFORM_RATIOS, f"Platform '{p}' missing from PLATFORM_RATIOS"

    def test_ratio_format(self):
        """All ratios must be in 'width:height' format."""
        for platform, ratio in PLATFORM_RATIOS.items():
            parts = ratio.split(":")
            assert len(parts) == 2, f"Bad ratio format for {platform}: {ratio}"
            w, h = parts
            assert w.isdigit() and h.isdigit(), f"Non-integer ratio for {platform}: {ratio}"


# ─── CREDIT_COST ─────────────────────────────────────────────────────────
class TestCreditCost:
    def test_all_models_have_cost(self):
        expected_models = ["gen4_turbo", "gen4.5", "gen4_image"]
        for model in expected_models:
            assert model in CREDIT_COST, f"Model '{model}' missing from CREDIT_COST"
            assert CREDIT_COST[model] > 0

    def test_gen4_image_cost(self):
        """gen4_image = 6 credits ($0.06)."""
        assert CREDIT_COST["gen4_image"] == 6

    def test_video_model_costs(self):
        """gen4_turbo = 5cr/s ($0.05), gen4.5 = 12cr/s ($0.12)."""
        assert CREDIT_COST["gen4_turbo"] == 5
        assert CREDIT_COST["gen4.5"] == 12

    def test_cost_usd_calculation(self):
        """5s gen4_turbo video = 25 credits = $0.25."""
        cost = CREDIT_COST["gen4_turbo"] * 5  # 5 second video
        assert cost == 25
        assert cost * 0.01 == pytest.approx(0.25)

    def test_30s_campaign_estimate(self):
        """30s campaign with 5 shots × 5s gen4_turbo + 5 images."""
        img_cost   = CREDIT_COST["gen4_image"] * 5
        video_cost = CREDIT_COST["gen4_turbo"] * 30
        total      = img_cost + video_cost
        assert total == 30 + 150   # 180 credits
        assert total * 0.01 == pytest.approx(1.80)
