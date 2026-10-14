"""Main pipeline orchestrator — coordinates all 9 stages with optional interactive approval gates."""

import asyncio
import glob as _glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pipeline import checkpoint as ckpt
from pipeline.end_card import try_build_and_append_end_card
from pipeline.state import PipelineState, PLATFORM_RATIOS, VIDEO_MODEL_DURATIONS
from pipeline.agents import (
    ingestion, enhancer, story, prompt_gen, validator,
    image_synth, vlm_validator, video_gen, audio_gen, end_card_gen,
)

APPROVAL_STAGES = {"2", "3", "4", "6a"}


def _find_ffmpeg() -> Optional[str]:
    """Robustly locate ffmpeg on PATH and common Windows install locations."""
    found = shutil.which("ffmpeg")
    if found:
        print(f"[orchestrator] ffmpeg via PATH: {found}")
        return found

    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.EXE"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg*\bin\ffmpeg.exe"),
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expandvars(r"%USERPROFILE%\scoop\shims\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for pattern in candidates:
        matches = _glob.glob(pattern)
        if matches:
            print(f"[orchestrator] ffmpeg via glob: {matches[0]}")
            return matches[0]
        if os.path.isfile(pattern):
            print(f"[orchestrator] ffmpeg found: {pattern}")
            return pattern

    print("[orchestrator] ffmpeg NOT found in any known location")
    return None


class PipelineOrchestrator:
    def __init__(self, session_id: str, session_dir: Path, send: Callable):
        self.session_id  = session_id
        self.session_dir = Path(session_dir).resolve()
        self.send        = send
        self.wait_for_approval: Optional[Callable] = None
        self._skip_all_reviews = False

    # ── Approval gate ──────────────────────────────────────────────────

    async def _approval_gate(
        self,
        stage_id: str,
        stage_name: str,
        state: PipelineState,
    ) -> dict:
        """Pause pipeline and wait for user to review, edit, or approve stage output.
        Returns the user's response dict with action and optional edits.
        """
        if not state.interactive or self._skip_all_reviews or not self.wait_for_approval:
            return {"action": "approve"}

        review_data = self._build_review_data(stage_id, state)

        await self.send({
            "type": "approval_request",
            "stage_id": stage_id,
            "stage_name": stage_name,
            "review_data": review_data,
        })

        await self.send({
            "type": "log", "stage": stage_id,
            "message": f"Waiting for your review of {stage_name}...",
            "level": "info",
        })

        print(f"[orchestrator] Stage {stage_id} awaiting approval...")
        response = await self.wait_for_approval()

        action = response.get("action", "approve")
        if action == "skip_all":
            self._skip_all_reviews = True
            await self.send({
                "type": "log", "stage": "sys",
                "message": "Auto-approve enabled — remaining stages will run without review.",
                "level": "info",
            })
            return {"action": "approve"}

        await self.send({
            "type": "log", "stage": stage_id,
            "message": f"Review complete — action: {action}",
            "level": "success",
        })

        if action == "edit" and response.get("edits"):
            self._apply_edits(stage_id, state, response["edits"])
            await self.send({
                "type": "log", "stage": stage_id,
                "message": "User edits applied.",
                "level": "success",
            })

        return response

    def _build_review_data(self, stage_id: str, state: PipelineState) -> dict:
        """Prepare the stage output data for the frontend review UI."""
        if stage_id == "2":
            b = state.enriched_brief
            return {
                "type": "brief",
                "fields": {
                    "campaign_title": b.get("campaign_title", ""),
                    "tagline": b.get("tagline", ""),
                    "tone": b.get("tone", b.get("emotional_tone", "")),
                    "visual_style": b.get("visual_style", ""),
                    "pacing": b.get("pacing", ""),
                    "music_mood": b.get("music_mood", ""),
                    "target_audience": b.get("target_audience", b.get("demographic_insights", "")),
                    "narrative_arc": b.get("narrative_arc", ""),
                    "creative_direction": b.get("creative_direction", ""),
                    "color_grade": b.get("color_grade", ""),
                },
            }

        if stage_id == "3":
            return {
                "type": "storyboard",
                "shots": [
                    {
                        "shot_id": s.shot_id,
                        "narrative_function": s.narrative_function,
                        "duration": s.duration,
                        "subject": s.subject,
                        "environment": s.environment,
                        "action": s.action,
                        "emotional_beat": s.emotional_beat,
                    }
                    for s in state.storyboard
                ],
            }

        if stage_id == "4":
            return {
                "type": "prompts",
                "shots": [
                    {
                        "shot_id": s.shot_id,
                        "narrative_function": s.narrative_function,
                        "image_prompt": s.image_prompt,
                        "motion_prompt": s.motion_prompt,
                    }
                    for s in state.storyboard
                ],
            }

        if stage_id == "6a":
            return {
                "type": "frames",
                "shots": [
                    {
                        "shot_id": s.shot_id,
                        "narrative_function": s.narrative_function,
                        "image_status": s.image_status,
                        "image_src": (
                            f"/output/{state.session_id}/images/{s.shot_id}.png"
                            if s.image_path else None
                        ),
                    }
                    for s in state.storyboard
                ],
            }

        return {}

    def _apply_edits(self, stage_id: str, state: PipelineState, edits: dict) -> None:
        """Merge user edits into pipeline state."""
        if stage_id == "2" and edits.get("fields"):
            for key, val in edits["fields"].items():
                if val is not None and str(val).strip():
                    state.enriched_brief[key] = str(val).strip()

        elif stage_id == "3" and edits.get("shots"):
            shot_map = {s.shot_id: s for s in state.storyboard}
            for edited in edits["shots"]:
                sid = edited.get("shot_id")
                if sid and sid in shot_map:
                    s = shot_map[sid]
                    if "narrative_function" in edited:
                        s.narrative_function = edited["narrative_function"]
                    if "duration" in edited:
                        s.duration = int(edited["duration"])
                    if "subject" in edited:
                        s.subject = edited["subject"]
                    if "environment" in edited:
                        s.environment = edited["environment"]
                    if "action" in edited:
                        s.action = edited["action"]

        elif stage_id == "4" and edits.get("shots"):
            shot_map = {s.shot_id: s for s in state.storyboard}
            for edited in edits["shots"]:
                sid = edited.get("shot_id")
                if sid and sid in shot_map:
                    s = shot_map[sid]
                    if "image_prompt" in edited:
                        s.image_prompt = edited["image_prompt"]
                    if "motion_prompt" in edited:
                        s.motion_prompt = edited["motion_prompt"]

    # ── Pipeline result builder ────────────────────────────────────────

    def _build_pipeline_result(
        self,
        state: PipelineState,
        final_path: Optional[str],
        audio_video_path: Optional[str],
    ) -> Dict[str, Any]:
        completed = [s for s in state.storyboard if s.video_status == "complete"]
        return {
            "final_video": (
                f"/output/{self.session_id}/final_with_audio.mp4"
                if audio_video_path
                else (f"/output/{self.session_id}/final.mp4" if final_path else None)
            ),
            "final_video_no_audio": f"/output/{self.session_id}/final.mp4" if final_path else None,
            "audio_src": f"/output/{self.session_id}/voiceover.mp3" if audio_video_path else None,
            "individual_clips": [
                f"/output/{self.session_id}/videos/{s.shot_id}.mp4"
                for s in completed
            ],
            "shots": [self._shot_summary(s) for s in state.storyboard],
            "total_cost_credits": state.total_cost_credits,
            "total_cost_usd": round(state.total_cost_usd, 2),
            "total_duration": sum(s.duration for s in completed) + state.playback_padding_seconds,
            "campaign_title": state.enriched_brief.get("campaign_title", "Ad Campaign"),
            "platform": state.platform,
            "quality_mode": state.quality_mode,
            "voiceover_script": state.voiceover_script,
        }

    async def _skip_stage(self, stage_id: str, stage_name: str, prior_session: Optional[str]) -> None:
        await self._stage_start(stage_id, stage_name)
        extra = f" (from session {prior_session})" if prior_session else ""
        await self.send({
            "type": "log",
            "stage": stage_id,
            "message": f"Skipped{extra} — same inputs, reused checkpoint.",
            "level": "info",
        })
        await self._stage_done(stage_id, stage_name)

    # ── Main run ───────────────────────────────────────────────────────

    async def run(
        self,
        prompt: str,
        image_data_uri: Optional[str],
        platform: str,
        quality_mode: str,
        total_duration: int,
        video_model: Optional[str] = None,
        image_model: Optional[str] = None,
        interactive: bool = True,
    ) -> Dict[str, Any]:

        resolved_video_model = video_model or (
            "gen4_turbo" if quality_mode == "preview" else "gen4.5"
        )
        resolved_image_model = image_model or "gen4_image"
        allowed_durations = VIDEO_MODEL_DURATIONS.get(resolved_video_model)

        state = PipelineState(
            session_id=self.session_id,
            original_prompt=prompt,
            image_data_uri=image_data_uri,
            platform=platform,
            quality_mode=quality_mode,
            total_duration=total_duration,
            ratio=PLATFORM_RATIOS.get(platform, "1280:720"),
            video_model=resolved_video_model,
            image_model=resolved_image_model,
            interactive=interactive,
        )
        state.enriched_brief["_allowed_durations"] = allowed_durations or []

        fingerprint = ckpt.compute_fingerprint(
            prompt=prompt,
            image_data_uri=image_data_uri,
            platform=platform,
            quality_mode=quality_mode,
            total_duration=total_duration,
            video_model=resolved_video_model,
            image_model=resolved_image_model,
            ratio=state.ratio,
        )

        output_root = self.session_dir.parent
        prior = ckpt.find_best_checkpoint(output_root, fingerprint, self.session_id)
        start_idx = 0
        prior_session_name: Optional[str] = None

        if prior:
            src_dir, cp = prior
            prior_session_name = src_dir.name
            await self.send({
                "type": "log",
                "stage": "sys",
                "message": (
                    f"Same inputs as session {prior_session_name} — "
                    "copying media and resuming after last good stage."
                ),
                "level": "info",
            })
            await asyncio.to_thread(ckpt.copy_media_tree, src_dir, self.session_dir)
            ckpt.merge_checkpoint_into_state(state, cp["state"], self.session_dir)

            if cp.get("pipeline_finished") and cp.get("last_completed_stage") == "9":
                await self.send({
                    "type": "log",
                    "stage": "sys",
                    "message": "Prior run fully completed — replaying outputs only (no API calls).",
                    "level": "success",
                })
                for sid in ckpt.STAGE_ORDER:
                    await self._skip_stage(sid, ckpt.STAGE_LABELS[sid], prior_session_name)
                ckpt.write_checkpoint(
                    self.session_dir, fingerprint, state, "9", pipeline_finished=True
                )
                return {
                    **self._build_pipeline_result(
                        state,
                        state.final_video_path,
                        state.final_audio_video_path,
                    ),
                    "resumed_from_session": prior_session_name,
                    "cache_hit": True,
                }

            last = cp.get("last_completed_stage") or ""
            start_idx = ckpt.stage_index(last) + 1

        for i, sid in enumerate(ckpt.STAGE_ORDER):
            if i < start_idx:
                await self._skip_stage(sid, ckpt.STAGE_LABELS[sid], prior_session_name)

        final_path: Optional[str] = state.final_video_path
        audio_video_path: Optional[str] = state.final_audio_video_path

        # ── Stage 1: Intent Ingestion ──────────────────────────────────
        if ckpt.stage_index("1") >= start_idx:
            await self._stage_start("1", "Intent Ingestion")
            await ingestion.run(state, self.send)
            await self._stage_done("1", "Intent Ingestion")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "1")

        # ── Stage 2: Contextual Enhancement + APPROVAL ─────────────────
        if ckpt.stage_index("2") >= start_idx:
            await self._stage_start("2", "Contextual Enhancement")
            await enhancer.run(state, self.send)
            await self._stage_done("2", "Contextual Enhancement")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "2")

            await self._approval_gate("2", "Creative Brief", state)

        # ── Stage 3: Narrative Sequencing + APPROVAL ───────────────────
        if ckpt.stage_index("3") >= start_idx:
            await self._stage_start("3", "Narrative Sequencing")
            await story.run(state, self.send)
            await self._stage_done("3", "Narrative Sequencing")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "3")

            await self._approval_gate("3", "Storyboard", state)

        # ── Stage 4: Prompt Engineering + APPROVAL ─────────────────────
        if ckpt.stage_index("4") >= start_idx:
            await self._stage_start("4", "Prompt Engineering")
            await prompt_gen.run(state, self.send)
            await self._stage_done("4", "Prompt Engineering")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "4")

            await self._approval_gate("4", "Image & Motion Prompts", state)

        # ── Stage 5: Structural Validation ─────────────────────────────
        if ckpt.stage_index("5") >= start_idx:
            await self._stage_start("5", "Structural Validation")
            await validator.run(state, self.send)
            await self._stage_done("5", "Structural Validation")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "5")

        # ── Stage 6: Visual Synthesis ──────────────────────────────────
        if ckpt.stage_index("6") >= start_idx:
            await self._stage_start("6", "Visual Synthesis")
            await image_synth.run(state, self.session_dir, self.send)
            await self._stage_done("6", "Visual Synthesis")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "6")

        # ── Stage 6a: VLM Validation + APPROVAL ───────────────────────
        if ckpt.stage_index("6a") >= start_idx:
            await self._stage_start("6a", "VLM Validation")
            await vlm_validator.run(state, self.session_dir, self.send)
            await self._stage_done("6a", "VLM Validation")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "6a")

            await self._approval_gate("6a", "Generated Frames", state)

        # ── Stage 7: Temporal Synthesis ────────────────────────────────
        if ckpt.stage_index("7") >= start_idx:
            await self._stage_start("7", "Temporal Synthesis")
            await video_gen.run(state, self.session_dir, self.send)
            await self._stage_done("7", "Temporal Synthesis")
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "7")

        # ── Stage 8: Non-Linear Stitching ──────────────────────────────
        if ckpt.stage_index("8") >= start_idx:
            await self._stage_start("8", "Non-Linear Stitching")
            await end_card_gen.run(state, self.session_dir, self.send)
            final_path = await self._stitch(state)
            await self._stage_done("8", "Non-Linear Stitching")
            state.final_video_path = final_path
            ckpt.write_checkpoint(self.session_dir, fingerprint, state, "8")

        # ── Stage 9: Audio Production ──────────────────────────────────
        if ckpt.stage_index("9") >= start_idx:
            await self._stage_start("9", "Audio Production")
            audio_video_path = await audio_gen.run(state, self.session_dir, self.send)
            await self._stage_done("9", "Audio Production")
            state.final_audio_video_path = audio_video_path
            ckpt.write_checkpoint(
                self.session_dir, fingerprint, state, "9", pipeline_finished=True
            )

        result = self._build_pipeline_result(state, final_path, audio_video_path)
        if prior_session_name:
            result["resumed_from_session"] = prior_session_name
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    async def _stitch(self, state: PipelineState) -> Optional[str]:
        state.playback_padding_seconds = 0
        video_paths = [
            self.session_dir / "videos" / f"{s.shot_id}.mp4"
            for s in state.storyboard
            if s.video_status == "complete"
        ]

        if not video_paths:
            await self.send({
                "type": "log", "stage": "8",
                "message": "No completed clips to stitch.", "level": "warning",
            })
            return None

        ffmpeg_exe = _find_ffmpeg()
        body_path = self.session_dir / "final_body.mp4"
        final_path = self.session_dir / "final.mp4"

        if len(video_paths) == 1:
            shutil.copy2(video_paths[0], body_path)
            await self.send({
                "type": "log", "stage": "8",
                "message": "Single clip — preparing final assembly.", "level": "success",
            })
        else:
            if not ffmpeg_exe:
                await self.send({
                    "type": "log", "stage": "8",
                    "message": (
                        "FFmpeg not found — individual clips are available for download. "
                        "Install FFmpeg (winget install Gyan.FFmpeg) and restart the server."
                    ),
                    "level": "warning",
                })
                return None

            concat_file = self.session_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for p in video_paths:
                    f.write(f"file '{p.resolve().as_posix()}'\n")

            await self.send({
                "type": "log", "stage": "8",
                "message": f"Stitching {len(video_paths)} clips with FFmpeg...", "level": "info",
            })
            print(f"[orchestrator] stitching {len(video_paths)} clips -> {body_path}")

            def _run_ffmpeg_concat() -> subprocess.CompletedProcess:
                return subprocess.run(
                    [
                        ffmpeg_exe, "-y",
                        "-f", "concat", "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy",
                        str(body_path),
                    ],
                    capture_output=True,
                    timeout=300,
                )

            try:
                result = await asyncio.to_thread(_run_ffmpeg_concat)
            except subprocess.TimeoutExpired:
                print("[orchestrator] FFmpeg stitch TIMEOUT after 300s")
                await self.send({
                    "type": "log", "stage": "8",
                    "message": "FFmpeg concat timed out after 5 minutes",
                    "level": "error",
                })
                return None
            except Exception as e:
                print(f"[orchestrator] FFmpeg stitch unexpected error: {type(e).__name__}: {e}")
                await self.send({
                    "type": "log", "stage": "8",
                    "message": f"FFmpeg concat failure: {type(e).__name__}: {e}",
                    "level": "error",
                })
                return None

            if result.returncode != 0:
                err = result.stderr.decode(errors="replace")[-400:]
                print(f"[orchestrator] FFmpeg stitch error: {err}")
                await self.send({
                    "type": "log", "stage": "8",
                    "message": f"FFmpeg concat error: {err[-200:]}", "level": "error",
                })
                return None

        appended = False
        if ffmpeg_exe:
            def _promo():
                return try_build_and_append_end_card(
                    ffmpeg_exe,
                    self.session_dir,
                    state,
                    body_path,
                    final_path,
                    video_paths,
                )

            appended, warn, pad_sec = await asyncio.to_thread(_promo)
            if warn:
                await self.send({"type": "log", "stage": "8", "message": warn, "level": "warning"})
            if appended:
                state.playback_padding_seconds = pad_sec
                await self.send({
                    "type": "log", "stage": "8",
                    "message": (
                        "Closing promo card merged — crossfade from spot into animated product "
                        "card (slow zoom + fade)."
                    ),
                    "level": "success",
                })

        if not appended:
            shutil.copy2(body_path, final_path)

        total_dur = sum(s.duration for s in state.storyboard if s.video_status == "complete")
        total_dur += state.playback_padding_seconds
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"[orchestrator] stitch complete: {final_path} ({size_mb:.1f} MB)")
        await self.send({
            "type": "log", "stage": "8",
            "message": f"Video assembled: {total_dur}s, {size_mb:.1f} MB",
            "level": "success",
        })
        await self.send({
            "type": "final_video_ready",
            "src": f"/output/{self.session_id}/final.mp4",
            "duration": total_dur,
            "size_mb": round(size_mb, 1),
        })
        return str(final_path)

    async def _stage_start(self, stage_id: str, name: str):
        print(f"[orchestrator] Stage {stage_id} start: {name}")
        await self.send({"type": "stage_start", "stage_id": stage_id, "stage_name": name})

    async def _stage_done(self, stage_id: str, name: str):
        print(f"[orchestrator] Stage {stage_id} done:  {name}")
        await self.send({"type": "stage_complete", "stage_id": stage_id, "stage_name": name})

    def _shot_summary(self, shot) -> dict:
        return {
            "shot_id":           shot.shot_id,
            "narrative_function": shot.narrative_function,
            "duration":          shot.duration,
            "subject":           shot.subject,
            "image_status":      shot.image_status,
            "video_status":      shot.video_status,
            "cost_credits":      shot.cost_credits,
        }
