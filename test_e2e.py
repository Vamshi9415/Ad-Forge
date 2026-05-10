"""End-to-end pipeline test — runs a full session via HTTP + WebSocket.

Usage:  python test_e2e.py
This will consume real API credits (RunwayML + OpenAI).
Set DRY_RUN=1 to stop after Stage 5 (no RunwayML calls).
"""

import asyncio
import json
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; X="\033[0m"
def ok(m): print(f"  {G}[OK]{X} {m}")
def err(m): print(f"  {R}[ERR]{X} {m}")
def info(m): print(f"  {C}[...]{X} {m}")
def warn(m): print(f"  {Y}[WARN]{X} {m}")
def hdr(m): print(f"\n{B}{C}{'='*55}{X}\n{B} {m}{X}\n{C}{'='*55}{X}")


async def run_e2e():
    import httpx, websockets

    BASE = "http://localhost:8000"
    WS_BASE = "ws://localhost:8000"

    hdr("End-to-End Pipeline Test")
    print(f"  Mode: {'DRY RUN (stages 1-5 only)' if DRY_RUN else 'FULL (all stages, real API calls)'}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Step 1: Create session ──────────────────────────────────────────
    hdr("Step 1: Create Pipeline Session")
    payload = {
        "prompt": "An eco-friendly insulated water bottle brand for outdoor adventurers. Features 24-hour temperature retention and BPA-free materials. CTA: Stay Cold. Stay Bold.",
        "platform": "youtube",
        "quality_mode": "preview",
        "total_duration": 10,  # Minimal — 2 x 5s clips
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{BASE}/api/sessions", json=payload)
        if not r.is_success:
            err(f"Session creation failed: {r.status_code} {r.text}")
            return False

        data = r.json()
        session_id = data["session_id"]
        ok(f"Session created: {session_id}")

    # ── Step 2: Connect WebSocket and watch pipeline ────────────────────
    hdr("Step 2: WebSocket Pipeline Stream")
    info(f"Connecting to ws://localhost:8000/api/ws/{session_id}")

    results = {
        "stages_started": [],
        "stages_complete": [],
        "stages_error": [],
        "images_generated": 0,
        "images_verified": 0,
        "videos_generated": 0,
        "final_video": None,
        "total_cost": 0.0,
        "campaign_title": None,
        "pipeline_ok": False,
    }

    t_start = time.time()
    stage_times = {}

    async with websockets.connect(f"{WS_BASE}/api/ws/{session_id}") as ws_conn:
        ok("WebSocket connected")
        print()

        dry_stopped = False
        while True:
            try:
                raw = await asyncio.wait_for(ws_conn.recv(), timeout=600)
            except asyncio.TimeoutError:
                err("WebSocket timed out after 600s")
                break
            except Exception as wse:
                if "1000" in str(wse) or "ConnectionClosed" in type(wse).__name__:
                    break  # Normal close
                err(f"WebSocket error: {wse}")
                break

            msg = json.loads(raw)
            t = msg.get("type", "")

            if t == "stage_start":
                sid = msg["stage_id"]
                results["stages_started"].append(sid)
                stage_times[sid] = time.time()
                print(f"  {C}> Stage {sid}: {msg.get('stage_name','')}{X}")

            elif t == "stage_complete":
                sid = msg["stage_id"]
                results["stages_complete"].append(sid)
                elapsed = time.time() - stage_times.get(sid, t_start)
                print(f"  {G}+ Stage {sid} complete ({elapsed:.1f}s){X}")

                # Dry-run: stop after Stage 5
                if DRY_RUN and sid == "5" and not dry_stopped:
                    dry_stopped = True
                    warn("DRY RUN mode — stopping after Stage 5")
                    results["pipeline_ok"] = True
                    break

            elif t == "stage_error":
                sid = msg["stage_id"]
                results["stages_error"].append(sid)
                err(f"Stage {sid} error: {msg.get('message', '')}")

            elif t == "log":
                level = msg.get("level", "info")
                stage = msg.get("stage", "?")
                text = msg.get("message", "")
                col = G if level == "success" else (R if level == "error" else (Y if level == "warning" else ""))
                print(f"    [{stage}] {col}{text}{X}")

            elif t == "image_ready":
                results["images_generated"] += 1
                cost = msg.get("cost_usd", 0)
                print(f"  {G}[IMG]{X} {msg['shot_id']} generated (cumulative: ${cost:.2f})")

            elif t == "image_validated":
                if msg.get("verdict") == "VERIFIED":
                    results["images_verified"] += 1
                    print(f"  {G}[VLM]{X} {msg['shot_id']} VERIFIED (confidence: {msg.get('confidence',0):.0%})")
                else:
                    print(f"  {Y}[VLM]{X} {msg['shot_id']} {msg.get('verdict','?')}")

            elif t == "video_ready":
                results["videos_generated"] += 1
                cost = msg.get("cost_usd", 0)
                print(f"  {G}[VID]{X} {msg['shot_id']} {msg.get('duration','?')}s clip ready (cumulative: ${cost:.2f})")

            elif t == "pipeline_complete":
                results["pipeline_ok"] = True
                results["final_video"] = msg.get("final_video")
                results["total_cost"] = msg.get("total_cost_usd", 0)
                results["campaign_title"] = msg.get("campaign_title", "")
                break

            elif t == "pipeline_error":
                err(f"Pipeline error: {msg.get('message','')}")
                if msg.get("detail"):
                    print(f"    Traceback:\n{msg['detail'][:500]}")
                break

            elif t == "done":
                break

            elif t == "ping":
                pass

    # ── Results ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hdr("Results")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"  Stages started:  {results['stages_started']}")
    print(f"  Stages complete: {results['stages_complete']}")
    if results["stages_error"]:
        print(f"  {R}Stages errored:  {results['stages_error']}{X}")
    print(f"  Images generated:{results['images_generated']}")
    print(f"  Images verified: {results['images_verified']}")
    print(f"  Videos generated:{results['videos_generated']}")
    print(f"  Final video:     {results['final_video'] or 'N/A'}")
    print(f"  Total cost:      ${results['total_cost']:.2f}")
    if results["campaign_title"]:
        print(f"  Campaign title:  {results['campaign_title']}")
    print()

    # ── Assertions ────────────────────────────────────────────────────
    hdr("Pass / Fail")
    checks = [
        ("Pipeline completed without error", results["pipeline_ok"]),
        ("Stages 1-5 completed", all(s in results["stages_complete"] for s in ["1","2","3","4","5"])),
    ]
    if not DRY_RUN:
        checks += [
            ("At least 1 image generated", results["images_generated"] > 0),
            ("At least 1 video generated", results["videos_generated"] > 0),
            ("No stage errors", len(results["stages_error"]) == 0),
        ]

    all_pass = True
    for desc, passed in checks:
        if passed:
            ok(desc)
        else:
            err(desc)
            all_pass = False

    print()
    if all_pass:
        print(f"  {G}{B}All checks passed!{X}")
        if results["final_video"]:
            print(f"  {G}Final video available at: http://localhost:8000{results['final_video']}{X}")
    else:
        print(f"  {R}{B}Some checks failed.{X}")

    return all_pass


if __name__ == "__main__":
    try:
        success = asyncio.run(run_e2e())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)
