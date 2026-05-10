"""Product promotion end card: hero still + campaign title / CTA, appended after main stitch."""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.state import PipelineState

# Promo clip length (seconds): subtle Ken Burns + fades inside this segment.
END_CARD_CLIP_SEC = 5.0
# Legacy alias (total promo duration for callers/tests).
END_CARD_DURATION_SEC = END_CARD_CLIP_SEC
# Cross-dissolve from main spot into promo card (broadcast-style transition).
BODY_PROMO_CROSSFADE_SEC = 1.0
_PROMO_OUT_FADE_SEC = 0.45
_PROMO_IN_FADE_SEC = 0.55
_PROMO_FPS = 30

PROMO_END_PROFESSIONAL_FILE = "promo_end_professional.jpg"
_IMG_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _shot_still_matches(path: Path, shot_id: str) -> bool:
    stem = path.stem
    return stem == shot_id or stem.startswith(f"{shot_id}_")


def _pick_latest_shot_still(img_dir: Path, shot_id: str) -> Optional[Path]:
    """Newest keyframe file for a shot (any common extension; Windows-safe)."""
    if not img_dir.is_dir():
        return None
    candidates: List[Path] = []
    for p in img_dir.iterdir():
        if not p.is_file() or not _shot_still_matches(p, shot_id):
            continue
        if p.suffix.lower() in _IMG_SUFFIXES:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def _ass_time(d: float) -> str:
    """H:MM:SS.cc for ASS Dialogue lines."""
    d = max(0.04, float(d))
    cs = int(round(d * 100))
    s = cs // 100
    cent = cs % 100
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}.{cent:02d}"


def _promo_ass_body(width: int, height: int, duration: float, title: str, cta: str) -> str:
    dur = _ass_time(duration)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Italic, Alignment, MarginL, MarginR, MarginV, Outline, Shadow
Style: PromoTitle,Arial,54,&H00FFFFFF,&H00101010,-1,0,8,40,40,70,3,2
Style: PromoCTA,Arial,64,&H00FFEEDD,&H00101010,-1,0,2,40,40,90,4,2

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{dur},PromoTitle,,0,0,0,,{{\\fad(200,0)}}{title}
Dialogue: 0,0:00:00.15,{dur},PromoCTA,,0,0,0,,{{\\fad(400,0)}}{cta}
"""


def parse_ratio(ratio: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+):(\d+)$", (ratio or "").strip())
    if not m:
        return 1280, 720
    return int(m.group(1)), int(m.group(2))


def _ass_escape(text: str, max_len: int = 140) -> str:
    t = " ".join(text.split()).strip()
    if len(t) > max_len:
        t = t[: max_len - 1] + "…"
    return (
        t.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def _find_ffprobe(ffmpeg_exe: str) -> Optional[str]:
    """Locate ffprobe next to ffmpeg or on PATH (needed for crossfade timing)."""
    found = shutil.which("ffprobe")
    if found:
        return found
    if ffmpeg_exe:
        parent = Path(ffmpeg_exe).resolve().parent
        for name in ("ffprobe.exe", "ffprobe"):
            p = parent / name
            if p.is_file():
                return str(p)
    return None


def probe_media_duration(ffprobe: str, media_path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if r.returncode != 0:
            return None
        return float(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None


def _input_path_for_cwd(media_path: Path, cwd: Path) -> str:
    """Prefer repo-relative paths when media lives under cwd (avoids subtitles path bugs)."""
    mp = media_path.resolve()
    cw = cwd.resolve()
    try:
        return str(mp.relative_to(cw))
    except ValueError:
        return str(mp)


def decode_data_uri_to_file(data_uri: Optional[str], dest: Path) -> bool:
    if not data_uri or "," not in data_uri:
        return False
    try:
        _header, b64part = data_uri.split(",", 1)
        raw = base64.b64decode(b64part, validate=False)
        if len(raw) < 32:
            return False
        dest.write_bytes(raw)
        return dest.is_file() and dest.stat().st_size > 0
    except Exception:
        return False


def extract_last_frame(ffmpeg: str, video: Path, dest: Path) -> bool:
    """Grab a frame near the end of the last clip (hero fallback)."""
    for sseof in ("-0.4", "-0.15"):
        cmd = [
            ffmpeg,
            "-y",
            "-sseof",
            sseof,
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 100:
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass

    cmd_fb = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    try:
        r = subprocess.run(cmd_fb, capture_output=True, timeout=60)
        return r.returncode == 0 and dest.is_file() and dest.stat().st_size > 100
    except (subprocess.TimeoutExpired, OSError):
        return False


def resolve_promo_media(
    session_dir: Path,
    state: PipelineState,
    last_video: Path,
    ffmpeg_exe: str,
) -> Tuple[Optional[Path], bool]:
    """Pick hero media + whether to burn ASS title/cta (False when AI slate includes typography).

    Order: AI professional slate → last-shot still → upload → video frame grab.
    """
    professional = session_dir / "images" / PROMO_END_PROFESSIONAL_FILE
    if professional.is_file() and professional.stat().st_size > 2048:
        return professional, False

    completed = [s for s in state.storyboard if s.video_status == "complete"]
    if completed:
        last_shot = completed[-1]
        picked = _pick_latest_shot_still(session_dir / "images", last_shot.shot_id)
        if picked:
            return picked, True

    ref_path = session_dir / "promo_upload_decode.bin"
    if decode_data_uri_to_file(state.image_data_uri, ref_path):
        return ref_path, True

    frame_path = session_dir / "end_card_last_frame.jpg"
    if ffmpeg_exe and extract_last_frame(ffmpeg_exe, last_video, frame_path):
        return frame_path, True

    return None, True


def build_end_card_mp4_with_text(
    ffmpeg: str,
    image_path: Path,
    out_mp4: Path,
    width: int,
    height: int,
    title: str,
    cta: str,
    duration: float = END_CARD_CLIP_SEC,
    burn_ass: bool = True,
) -> bool:
    """Animate still → promo clip: slow zoom + fades (+ optional ASS); cinema-style product card."""
    session_dir = out_mp4.parent.resolve()
    dur = max(1.2, float(duration))
    n_frames = max(int(round(dur * _PROMO_FPS)), int(_PROMO_FPS))

    fade_out_st = max(0.0, dur - _PROMO_OUT_FADE_SEC)

    base = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=#141428,"
        f"zoompan=z='min(zoom+0.0022,1.14)':d={n_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={_PROMO_FPS}"
    )

    if burn_ass:
        session_ass = session_dir / "promo_end_card.ass"
        body = _promo_ass_body(
            width,
            height,
            dur,
            _ass_escape(title or " "),
            _ass_escape(cta or "Learn more"),
        )
        session_ass.write_text(body, encoding="utf-8")
        vf = (
            base
            + ",subtitles=promo_end_card.ass"
            + f",fade=t=in:st=0:d={_PROMO_IN_FADE_SEC},"
            + f"fade=t=out:st={fade_out_st:.3f}:d={_PROMO_OUT_FADE_SEC}"
        )
    else:
        vf = (
            base
            + f",fade=t=in:st=0:d={_PROMO_IN_FADE_SEC},"
            + f"fade=t=out:st={fade_out_st:.3f}:d={_PROMO_OUT_FADE_SEC}"
        )

    img_in = _input_path_for_cwd(image_path, session_dir)
    out_name = out_mp4.name

    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(_PROMO_FPS),
        "-i",
        img_in,
        "-vf",
        vf,
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        out_name,
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            cwd=str(session_dir),
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")[-500:]
            print(f"[end_card] build_end_card_mp4 failed: {err}")
            return False
        return out_mp4.is_file() and out_mp4.stat().st_size > 100
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[end_card] build_end_card_mp4 error: {type(e).__name__}: {e}")
        return False


def append_end_card_hard_cut(ffmpeg: str, body_mp4: Path, end_mp4: Path, final_mp4: Path) -> bool:
    """Concatenate video streams with a hard cut (fallback when crossfade unavailable)."""
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(body_mp4),
        "-i",
        str(end_mp4),
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")[-500:]
            print(f"[end_card] append concat failed: {err}")
            return False
        return final_mp4.is_file() and final_mp4.stat().st_size > 100
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[end_card] append concat error: {type(e).__name__}: {e}")
        return False


def merge_body_promo_crossfade(
    ffmpeg: str,
    body_mp4: Path,
    promo_mp4: Path,
    final_mp4: Path,
    body_duration_sec: float,
    crossfade_sec: float,
) -> bool:
    """Dissolve last moments of main spot into the animated promo card (no audio)."""
    xf = max(0.05, float(crossfade_sec))
    offset = max(0.0, float(body_duration_sec) - xf)
    fc = (
        f"[0:v][1:v]xfade=transition=fade:duration={xf:.4f}:offset={offset:.4f}[outv]"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(body_mp4),
        "-i",
        str(promo_mp4),
        "-filter_complex",
        fc,
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace")[-600:]
            print(f"[end_card] crossfade merge failed: {err}")
            return False
        return final_mp4.is_file() and final_mp4.stat().st_size > 100
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[end_card] crossfade merge error: {type(e).__name__}: {e}")
        return False


def brief_lines(state: PipelineState) -> Tuple[str, str]:
    b = state.enriched_brief or {}
    title = (b.get("campaign_title") or b.get("brand_name") or "Your Brand").strip()
    cta = (
        b.get("primary_cta")
        or b.get("cta")
        or b.get("unique_selling_point")
        or "Shop now"
    )
    if isinstance(cta, str):
        cta = cta.strip()
    else:
        cta = str(cta).strip()
    return title, cta


def try_build_and_append_end_card(
    ffmpeg: str,
    session_dir: Path,
    state: PipelineState,
    body_mp4: Path,
    final_mp4: Path,
    video_paths: List[Path],
) -> Tuple[bool, Optional[str], int]:
    """Return (appended_ok, warning_message, playback_padding_seconds_for_mux).

    playback_padding_seconds approximates net runtime added to the main spot
    (promo length minus crossfade overlap).
    """
    if not body_mp4.is_file() or not video_paths:
        return False, None, 0

    last_video = Path(video_paths[-1])
    img, burn_ass = resolve_promo_media(session_dir, state, last_video, ffmpeg)
    if not img:
        return False, "Promo end card skipped — no hero image or frame extract.", 0

    w, h = parse_ratio(state.ratio)
    title, cta = brief_lines(state)

    end_mp4 = session_dir / "promo_end_card.mp4"
    if not build_end_card_mp4_with_text(
        ffmpeg,
        img,
        end_mp4,
        w,
        h,
        title=title,
        cta=cta,
        duration=END_CARD_CLIP_SEC,
        burn_ass=burn_ass,
    ):
        return False, "Promo end card skipped — FFmpeg could not build promo motion clip.", 0

    ffprobe = _find_ffprobe(ffmpeg)
    promo_dur = (
        probe_media_duration(ffprobe, end_mp4) if ffprobe else None
    ) or END_CARD_CLIP_SEC
    body_dur = (
        probe_media_duration(ffprobe, body_mp4) if ffprobe else None
    )

    tmp_final = session_dir / "final_with_promo_tmp.mp4"
    if tmp_final.exists():
        tmp_final.unlink()

    crossfade_used = 0.0
    merged = False
    if ffprobe and body_dur and body_dur > 0.35:
        xf_target = min(
            BODY_PROMO_CROSSFADE_SEC,
            body_dur * 0.33,
            promo_dur * 0.4,
        )
        xf_target = max(0.35, xf_target)
        xf_target = min(xf_target, body_dur - 0.05, promo_dur - 0.15)
        merged = merge_body_promo_crossfade(
            ffmpeg,
            body_mp4,
            end_mp4,
            tmp_final,
            body_dur,
            xf_target,
        )
        if merged:
            crossfade_used = xf_target

    if not merged:
        merged = append_end_card_hard_cut(ffmpeg, body_mp4, end_mp4, tmp_final)
        crossfade_used = 0.0

    if merged:
        shutil.move(str(tmp_final), str(final_mp4))
        pad = max(1, int(round(promo_dur - crossfade_used)))
        return True, None, pad

    if tmp_final.exists():
        tmp_final.unlink()
    return False, "Promo end card skipped — FFmpeg merge failed.", 0
