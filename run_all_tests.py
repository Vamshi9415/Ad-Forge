#!/usr/bin/env python
"""
AdForge — Master Test Runner
=============================
Runs all tests in the tests/ folder in logical order:

  Tier 0 (unit)       — no API calls, instant
  Tier 1 (FFmpeg)     — no API calls, local subprocess
  Tier 2 (API)        — real RunwayML + OpenAI calls
  Tier 3 (mock)       — unit tests for VLM retry logic
  Tier 4 (audio)      — real OpenAI TTS + FFmpeg mux
  Tier 5 (server)     — FastAPI HTTP + WebSocket (requires server running)

Usage:
  python run_all_tests.py              # all tiers
  python run_all_tests.py --unit       # only unit tests (no API)
  python run_all_tests.py --no-server  # skip integration tests
  python run_all_tests.py --fast       # unit + FFmpeg only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def banner(text, color=BLUE):
    width = 70
    print(f"\n{color}{'━' * width}{RESET}")
    print(f"{BOLD}{color}  {text}{RESET}")
    print(f"{color}{'━' * width}{RESET}")


def section(text):
    print(f"\n{CYAN}{'─' * 50}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{CYAN}{'─' * 50}{RESET}")


def run_pytest(args: list[str], label: str) -> tuple[int, float]:
    """Run pytest with given args. Returns (exit_code, elapsed_seconds)."""
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"\n{DIM}  $ {' '.join(cmd)}{RESET}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    return result.returncode, elapsed


def check_server_running() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:8000/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="AdForge test runner")
    parser.add_argument("--unit",       action="store_true", help="Run unit tests only (no API)")
    parser.add_argument("--fast",       action="store_true", help="Unit + FFmpeg only")
    parser.add_argument("--no-server",  action="store_true", help="Skip integration tests")
    parser.add_argument("--no-slow",    action="store_true", help="Skip slow tests (>30s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    args = parser.parse_args()

    banner("AdForge — Comprehensive Test Suite", BLUE)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD:    {Path.cwd()}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}  # tier_name → (returncode, elapsed)
    verbose = ["-v"] if args.verbose else []

    # ─── Tier 0: Environment + Module Imports ────────────────────────────
    section("TIER 0 — Environment & Recursive Module Import Trace")
    code, elapsed = run_pytest(
        ["tests/test_00_env.py", "-m", "unit", "--tb=short"] + verbose,
        "environment"
    )
    results["T0 Environment"] = (code, elapsed)

    if code != 0:
        print(f"\n{RED}{BOLD}CRITICAL: Environment checks failed — fix imports before proceeding.{RESET}")
        _print_summary(results)
        return code

    # ─── Tier 1: FFmpeg ───────────────────────────────────────────────────
    section("TIER 1 — FFmpeg: Detection, Version, Concat, Mux")
    ffmpeg_args = ["tests/test_01_ffmpeg.py", "--tb=short"]
    if args.no_slow:
        ffmpeg_args += ["-m", "not slow"]
    code, elapsed = run_pytest(ffmpeg_args + verbose, "ffmpeg")
    results["T1 FFmpeg"] = (code, elapsed)

    if args.fast or args.unit:
        _print_summary(results)
        return max(r[0] for r in results.values())

    # ─── Tier 2: External API Connections ────────────────────────────────
    section("TIER 2 — API Connections (RunwayML + OpenAI)")
    api_args = ["tests/test_02_apis.py", "--tb=short"]
    if args.no_slow:
        api_args += ["-m", "not slow"]
    code, elapsed = run_pytest(api_args + verbose, "apis")
    results["T2 APIs"] = (code, elapsed)

    # ─── Tier 3a: State / Dataclasses ────────────────────────────────────
    section("TIER 3a — State: PipelineState, Shot, CameraParams, Constants")
    code, elapsed = run_pytest(
        ["tests/test_03_state.py", "--tb=short"] + verbose, "state"
    )
    results["T3a State"] = (code, elapsed)

    # ─── Tier 3b: LLM Agent Chain (Stages 1-5) ───────────────────────────
    section("TIER 3b — LLM Agent Chain: Ingestion → Enhancement → Story → PromptGen → Validator")
    code, elapsed = run_pytest(
        ["tests/test_04_agents_chain.py", "--tb=short"] + verbose, "agents"
    )
    results["T3b Agents (1-5)"] = (code, elapsed)

    if args.unit:
        _print_summary(results)
        return max(r[0] for r in results.values())

    # ─── Tier 4: VLM Retry Logic (mock-based) ────────────────────────────
    section("TIER 4 — VLM Validator: Retry Logic (mock-based, $0 cost)")
    code, elapsed = run_pytest(
        ["tests/test_05_vlm_retry.py", "--tb=short"] + verbose, "vlm_retry"
    )
    results["T4 VLM Retry"] = (code, elapsed)

    # ─── Tier 5: Audio Generation ─────────────────────────────────────────
    section("TIER 5 — Audio Generation: Script + TTS + FFmpeg Mux")
    audio_args = ["tests/test_06_audio_gen.py", "--tb=short"]
    if args.no_slow:
        audio_args += ["-m", "not slow"]
    code, elapsed = run_pytest(audio_args + verbose, "audio")
    results["T5 Audio Gen"] = (code, elapsed)

    # ─── Tier 6: Server Integration ───────────────────────────────────────
    if not args.no_server:
        section("TIER 6 — Server Integration: HTTP Endpoints + WebSocket")
        server_running = check_server_running()
        if server_running:
            code, elapsed = run_pytest(
                ["tests/test_07_server.py", "--tb=short"] + verbose, "server"
            )
            results["T6 Server"] = (code, elapsed)
        else:
            print(f"\n{YELLOW}  ⚠ Server not running on port 8000 — skipping server tests.{RESET}")
            print(f"  Start it with: {DIM}python -m uvicorn server:app --port 8000{RESET}")
            results["T6 Server"] = (-1, 0)  # -1 = skipped
    else:
        print(f"\n{YELLOW}  [SKIPPED] Server integration tests (--no-server){RESET}")

    _print_summary(results)
    return max(r[0] for r in results.values() if r[0] != -1)


def _print_summary(results: dict):
    banner("TEST SUMMARY", GREEN if all(r[0] == 0 for r in results.values() if r[0] != -1) else RED)

    total_time = sum(r[1] for r in results.values())
    all_passed = True

    for tier, (code, elapsed) in results.items():
        if code == -1:
            icon = f"{YELLOW}SKIP{RESET}"
            status_str = "skipped"
        elif code == 0:
            icon = f"{GREEN}PASS{RESET}"
            status_str = f"{elapsed:.1f}s"
        else:
            icon = f"{RED}FAIL{RESET}"
            status_str = f"{elapsed:.1f}s"
            all_passed = False

        print(f"  [{icon}] {tier:<28} {DIM}{status_str}{RESET}")

    print(f"\n  {DIM}Total wall time: {total_time:.1f}s{RESET}")

    if all_passed:
        print(f"\n  {GREEN}{BOLD}✓ All test tiers passed! Pipeline is ready.{RESET}\n")
    else:
        failed = [t for t, (c, _) in results.items() if c not in (0, -1)]
        print(f"\n  {RED}{BOLD}✗ Failed tiers: {', '.join(failed)}{RESET}\n")
        print(f"  Re-run a specific tier with:")
        for t in failed:
            file_map = {
                "T0 Environment":   "tests/test_00_env.py",
                "T1 FFmpeg":        "tests/test_01_ffmpeg.py",
                "T2 APIs":          "tests/test_02_apis.py",
                "T3a State":        "tests/test_03_state.py",
                "T3b Agents (1-5)": "tests/test_04_agents_chain.py",
                "T4 VLM Retry":     "tests/test_05_vlm_retry.py",
                "T5 Audio Gen":     "tests/test_06_audio_gen.py",
                "T6 Server":        "tests/test_07_server.py",
            }
            f = file_map.get(t, "tests/")
            print(f"    {DIM}pytest {f} -v{RESET}")
        print()


if __name__ == "__main__":
    sys.exit(main())
