"""Fingerprint-based checkpoints for resume and same-prompt reuse.

Scans sibling session folders under output/ for checkpoint.json with a
matching input fingerprint. Restores serialized state and copied media so
the pipeline can skip completed stages or short-circuit a fully finished run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.state import CameraParams, PipelineState, Shot

# Ordered pipeline stage ids (must match orchestrator)
STAGE_ORDER: Tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "6a", "7", "8", "9")

STAGE_LABELS: Dict[str, str] = {
    "1": "Intent Ingestion",
    "2": "Contextual Enhancement",
    "3": "Narrative Sequencing",
    "4": "Prompt Engineering",
    "5": "Structural Validation",
    "6": "Visual Synthesis",
    "6a": "VLM Validation",
    "7": "Temporal Synthesis",
    "8": "Non-Linear Stitching",
    "9": "Audio Production",
}


def stage_index(stage_id: str) -> int:
    try:
        return STAGE_ORDER.index(stage_id)
    except ValueError:
        return -1


def compute_fingerprint(
    prompt: str,
    image_data_uri: Optional[str],
    platform: str,
    quality_mode: str,
    total_duration: int,
    video_model: str,
    image_model: str,
    ratio: str,
) -> str:
    """Stable hash of all inputs that affect generation."""
    img_digest = ""
    if image_data_uri:
        h = hashlib.sha256()
        step = 65536
        for i in range(0, len(image_data_uri), step):
            h.update(image_data_uri[i : i + step].encode("utf-8", errors="replace"))
        img_digest = h.hexdigest()[:48]
    payload = {
        "prompt": (prompt or "").strip(),
        "image_digest": img_digest,
        "platform": platform,
        "quality_mode": quality_mode,
        "total_duration": int(total_duration),
        "video_model": video_model,
        "image_model": image_model,
        "ratio": ratio,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _shot_to_dict(shot: Shot) -> dict:
    cam = shot.camera
    img_base = ""
    if shot.image_path:
        try:
            img_base = Path(shot.image_path).name
        except Exception:
            img_base = ""
    vid_base = ""
    if shot.video_path:
        try:
            vid_base = Path(shot.video_path).name
        except Exception:
            vid_base = ""
    return {
        "shot_id": shot.shot_id,
        "narrative_function": shot.narrative_function,
        "duration": shot.duration,
        "subject": shot.subject,
        "environment": shot.environment,
        "action": shot.action,
        "emotional_beat": shot.emotional_beat,
        "transition_to_next": shot.transition_to_next,
        "image_prompt": shot.image_prompt,
        "motion_prompt": shot.motion_prompt,
        "camera": {
            "pan": cam.pan,
            "tilt": cam.tilt,
            "zoom": cam.zoom,
            "roll": cam.roll,
            "static": cam.static,
        },
        "image_basename": img_base,
        "image_url": shot.image_url,
        "image_status": shot.image_status,
        "video_basename": vid_base,
        "video_status": shot.video_status,
        "seed": shot.seed,
        "cost_credits": shot.cost_credits,
    }


def _dict_to_shot(d: dict, session_dir: Path) -> Shot:
    camd = d.get("camera") or {}
    cam = CameraParams(
        pan=float(camd.get("pan", 0.0)),
        tilt=float(camd.get("tilt", 0.0)),
        zoom=float(camd.get("zoom", 0.0)),
        roll=float(camd.get("roll", 0.0)),
        static=bool(camd.get("static", False)),
    )
    shot_id = d.get("shot_id", "shot_01")
    ib = d.get("image_basename") or ""
    vb = d.get("video_basename") or ""

    img_path: Optional[str] = None
    if ib:
        p = session_dir / "images" / ib
        if p.is_file():
            img_path = str(p)
        else:
            matches = sorted(
                (session_dir / "images").glob(f"{shot_id}*.jpg"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if matches:
                img_path = str(matches[0])

    vid_path: Optional[str] = None
    if vb:
        vp = session_dir / "videos" / vb
        if vp.is_file():
            vid_path = str(vp)
    if not vid_path:
        vp2 = session_dir / "videos" / f"{shot_id}.mp4"
        if vp2.is_file():
            vid_path = str(vp2)

    return Shot(
        shot_id=shot_id,
        narrative_function=d.get("narrative_function", ""),
        duration=int(d.get("duration", 5)),
        subject=d.get("subject", ""),
        environment=d.get("environment", ""),
        action=d.get("action", ""),
        emotional_beat=d.get("emotional_beat", ""),
        transition_to_next=d.get("transition_to_next", ""),
        image_prompt=d.get("image_prompt", ""),
        motion_prompt=d.get("motion_prompt", ""),
        camera=cam,
        image_path=img_path,
        image_url=d.get("image_url"),
        image_status=d.get("image_status", "pending"),
        video_path=vid_path,
        video_status=d.get("video_status", "pending"),
        seed=d.get("seed"),
        cost_credits=int(d.get("cost_credits", 0)),
    )


def _load_image_b64(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except OSError:
        return None


def serialize_state(state: PipelineState) -> dict:
    return {
        "original_prompt": state.original_prompt,
        "platform": state.platform,
        "quality_mode": state.quality_mode,
        "total_duration": state.total_duration,
        "ratio": state.ratio,
        "video_model": state.video_model,
        "image_model": state.image_model,
        "enriched_brief": state.enriched_brief,
        "storyboard": [_shot_to_dict(s) for s in state.storyboard],
        "total_cost_credits": state.total_cost_credits,
        "total_cost_usd": state.total_cost_usd,
        "final_video_path": state.final_video_path,
        "final_audio_video_path": state.final_audio_video_path,
        "voiceover_script": state.voiceover_script,
        "playback_padding_seconds": state.playback_padding_seconds,
    }


def hydrate_shots_from_disk(state: PipelineState) -> None:
    """Fill image_base64 from image_path for each shot when present."""
    for shot in state.storyboard:
        shot.image_base64 = _load_image_b64(shot.image_path)


def merge_checkpoint_into_state(
    state: PipelineState,
    payload: dict,
    session_dir: Path,
) -> None:
    """Apply checkpoint 'state' dict onto live PipelineState (same session_id)."""
    state.enriched_brief = dict(payload.get("enriched_brief") or {})
    state.storyboard = [
        _dict_to_shot(sd, session_dir) for sd in (payload.get("storyboard") or [])
    ]
    state.total_cost_credits = int(payload.get("total_cost_credits", 0))
    state.total_cost_usd = float(payload.get("total_cost_usd", 0.0))
    # Never trust artifact paths from another session directory — use this session's files.
    fv = session_dir / "final.mp4"
    fa = session_dir / "final_with_audio.mp4"
    state.final_video_path = str(fv) if fv.is_file() else None
    state.final_audio_video_path = str(fa) if fa.is_file() else None
    state.voiceover_script = payload.get("voiceover_script")
    state.playback_padding_seconds = int(payload.get("playback_padding_seconds", 0))
    hydrate_shots_from_disk(state)


def copy_media_tree(src: Path, dst: Path) -> None:
    """Copy images/, videos/, and common outputs from a prior session."""
    for sub in ("images", "videos"):
        s, d = src / sub, dst / sub
        if not s.is_dir():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in s.iterdir():
            if f.is_file():
                shutil.copy2(f, d / f.name)
    for name in (
        "final.mp4",
        "final_with_audio.mp4",
        "voiceover.mp3",
        "music.mp3",
        "mixed_audio.aac",
        "concat.txt",
        "final_body.mp4",
        "promo_end_card.mp4",
        "promo_end_card.ass",
    ):
        sf = src / name
        if sf.is_file():
            shutil.copy2(sf, dst / name)
    for f in src.glob("ambient_*.mp3"):
        shutil.copy2(f, dst / f.name)
    for f in src.glob("images/*_r*.jpg"):
        shutil.copy2(f, dst / "images" / f.name)


def find_best_checkpoint(
    output_root: Path,
    fingerprint: str,
    exclude_session_id: str,
) -> Optional[Tuple[Path, dict]]:
    """Return (session_dir, checkpoint_dict) for best matching prior run.

    Prefers a fully finished run (pipeline_finished + stage 9) for cache replay;
    otherwise the run that progressed furthest (highest stage index, then newest).
    """
    if not output_root.is_dir():
        return None
    matches: List[Tuple[Path, dict, float]] = []
    for child in output_root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if child.name == exclude_session_id:
            continue
        cp_path = child / "checkpoint.json"
        if not cp_path.is_file():
            continue
        try:
            data = json.loads(cp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("fingerprint") != fingerprint:
            continue
        last = data.get("last_completed_stage") or ""
        if stage_index(last) < 0:
            continue
        mtime = cp_path.stat().st_mtime
        matches.append((child, data, mtime))

    if not matches:
        return None

    finished = [
        (d, cp, mt)
        for d, cp, mt in matches
        if cp.get("pipeline_finished") and cp.get("last_completed_stage") == "9"
    ]
    if finished:
        d, cp, _ = max(finished, key=lambda x: x[2])
        return d, cp

    def rank(t: Tuple[Path, dict, float]) -> Tuple[int, float]:
        cp = t[1]
        return (stage_index(cp.get("last_completed_stage") or ""), t[2])

    d, cp, _ = max(matches, key=rank)
    return d, cp


def write_checkpoint(
    session_dir: Path,
    fingerprint: str,
    state: PipelineState,
    last_completed_stage: str,
    pipeline_finished: bool = False,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "fingerprint": fingerprint,
        "last_completed_stage": last_completed_stage,
        "pipeline_finished": pipeline_finished,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": serialize_state(state),
    }
    path = session_dir / "checkpoint.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=True), encoding="utf-8")
