from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CameraParams:
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    roll: float = 0.0
    static: bool = False


@dataclass
class Shot:
    shot_id: str
    narrative_function: str
    duration: int
    subject: str
    environment: str
    action: str

    emotional_beat: str = ""
    transition_to_next: str = ""

    image_prompt: str = ""
    motion_prompt: str = ""
    camera: CameraParams = field(default_factory=CameraParams)

    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_status: str = "pending"

    video_path: Optional[str] = None
    video_status: str = "pending"
    seed: Optional[int] = None

    cost_credits: int = 0


@dataclass
class PipelineState:
    session_id: str
    original_prompt: str

    image_data_uri: Optional[str] = None
    platform: str = "youtube"
    quality_mode: str = "preview"
    total_duration: int = 30
    ratio: str = "1280:720"

    video_model: str = "gen4_turbo"
    image_model: str = "gen4_image"

    enriched_brief: Dict[str, Any] = field(default_factory=dict)
    storyboard: List[Shot] = field(default_factory=list)

    total_cost_credits: int = 0
    total_cost_usd: float = 0.0

    final_video_path: Optional[str] = None
    final_audio_video_path: Optional[str] = None
    voiceover_script: Optional[str] = None

    # Silent promo still appended after stitch (stage 8); audio mux extends to match.
    playback_padding_seconds: int = 0

    # Interactive mode: pause for user approval after key stages
    interactive: bool = True
    dub_languages: List[str] = field(default_factory=list)


PLATFORM_RATIOS: Dict[str, str] = {
    "youtube": "1280:720",
    "instagram_reels": "720:1280",
    "tiktok": "720:1280",
    "linkedin": "1280:720",
    "twitter": "1280:720",
    "instagram_square": "1024:1024",
}

CREDIT_COST: Dict[str, int] = {
    # ── Video models (credits per second) ────────────────────────────────
    "gen3a_turbo":  5,   # Gen-3 Alpha Turbo, image_to_video, $0.05/s (Runway API pricing)
    "gen4_turbo":   5,   # Fast image-driven, $0.05/s
    "gen4.5":      12,   # High quality general purpose, $0.12/s
    "seedance2":   36,   # Reference image+video, up to 15s, $0.36/s
    "veo3":        40,   # Google premium, 8s only, $0.40/s
    "veo3.1":      30,   # Google high quality, 4/6/8s, $0.30/s
    "veo3.1_fast": 12,   # Google fast, 4/6/8s, $0.12/s
    # ── Image models (credits per generation) ────────────────────────────
    "gen4_image":        6,   # Highest quality, $0.06
    "gen4_image_turbo":  2,   # Fast & cheap, $0.02
    "gemini_2.5_flash":  5,   # Google Gemini, $0.05
    # ── Audio models (RunwayML / ElevenLabs) ─────────────────────────────
    "eleven_multilingual_v2":   1,   # TTS: ~1 credit per 50 chars
    "eleven_text_to_sound_v2":  2,   # SFX: ~1-2 credits per generation
}

# Video model constraints: allowed durations (None = any duration)
VIDEO_MODEL_DURATIONS: Dict[str, list] = {
    "gen3a_turbo":  [5, 10],
    "gen4_turbo":   [5, 10],
    "gen4.5":       [5, 10],
    "seedance2":    [5, 8, 10, 15],
    "veo3":         [8],
    "veo3.1":       [4, 6, 8],
    "veo3.1_fast":  [4, 6, 8],
}

VIDEO_MODEL_INFO: Dict[str, Dict] = {
    "gen3a_turbo":  {"label": "Gen3 Alpha Turbo", "cost_label": "$0.05/s", "tag": "🧬 Gen-3"},
    "gen4_turbo":   {"label": "Gen4 Turbo",    "cost_label": "$0.05/s", "tag": "⚡ Fast"},
    "gen4.5":       {"label": "Gen4.5",         "cost_label": "$0.12/s", "tag": "🎬 Quality"},
    "seedance2":    {"label": "Seedance 2",     "cost_label": "$0.36/s", "tag": "🎭 Long-form"},
    "veo3":         {"label": "Veo 3",          "cost_label": "$0.40/s", "tag": "✨ Premium"},
    "veo3.1":       {"label": "Veo 3.1",        "cost_label": "$0.30/s", "tag": "🌟 Google HQ"},
    "veo3.1_fast":  {"label": "Veo 3.1 Fast",   "cost_label": "$0.12/s", "tag": "🚀 Google Fast"},
}

IMAGE_MODEL_INFO: Dict[str, Dict] = {
    "gen4_image":        {"label": "Gen4 Image",         "cost_label": "$0.06/frame", "tag": "🎨 Best"},
    "gen4_image_turbo":  {"label": "Gen4 Image Turbo",   "cost_label": "$0.02/frame", "tag": "⚡ Fast"},
    "gemini_2.5_flash":  {"label": "Gemini 2.5 Flash",   "cost_label": "$0.05/frame", "tag": "🌐 Google"},
}
