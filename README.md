# AdForge — Generative Advertising Pipeline

AdForge is a **FastAPI** application that turns a **creative brief** (text plus optional reference image) into a **multi-shot advertisement video** with voiceover, music, ambient sound, and a **professional closing promo card**. It orchestrates **OpenAI** models for planning and copy, **RunwayML** for images, video, and audio (via ElevenLabs-backed endpoints), and **FFmpeg** for assembly, transitions, and muxing.

---

## Features

- **End-to-end pipeline**: brief ingestion → storyboard → prompts → reference frames → optional VLM checks → video clips → stitch → animated end card with crossfade → mixed audio → final MP4.
- **Platforms**: aspect ratios for YouTube, Reels, TikTok, LinkedIn, Twitter, Instagram square (`pipeline/state.py`).
- **Quality modes**: preview vs higher-quality video model defaults.
- **Checkpoints**: fingerprint-based resume and reuse of prior sessions under `output/`.
- **Web UI**: `static/index.html` with WebSocket progress.
- **REST API**: session creation, image upload, OpenAPI at `/docs`.

---

## Requirements

- **Python** 3.10+ recommended  
- **FFmpeg** and **ffprobe** on `PATH` (or a standard Windows install location scanned by the orchestrator). Required for concat, end-card motion, crossfade, and audio mux.  
- **API keys** (see below)

---

## Installation

```powershell
cd d:\0_placements\genai\runway
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create your local secrets file from the template (never commit real keys):

```powershell
copy .env.example .env
```

Edit `.env` and fill in `OPENAI_API_KEY` and `RUNWAYML_API_SECRET`. The app loads `.env` at startup via `python-dotenv` (`server.py` uses `load_dotenv(override=True)`).

---

## Environment variables

See **`.env.example`** in the repo root for a commented template (same variables as below).

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT stages: ingestion, enhancement, storyboard, prompts, validation, voiceover script, VLM copy, end-card prompt planning. |
| `RUNWAYML_API_SECRET` | RunwayML: images, video, TTS, SFX; also VLM-related calls where applicable. |
| `RUNWAY_SSL_VERIFY` | Set to `0` only if you hit TLS issues behind corporate SSL inspection (see `pipeline/runway_client.py`). |

Optional: `DRY_RUN` for certain scripts/tests (`test_e2e.py`).

Commit **`.env.example`** only; keep **`.env`** private (do not push secrets).

---

## Run the server

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

- App: `http://127.0.0.1:8000/`  
- OpenAPI: `http://127.0.0.1:8000/docs`  

On some Windows setups, ports **8000** or **8001** may return **WinError 10013**. Try another port (for example **8765**) or inspect excluded port ranges with:

`netsh interface ipv4 show excludedportrange protocol=tcp`

---

## HTTP and WebSocket overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (`static/index.html`) |
| `POST` | `/api/upload-image` | Upload image → `data_uri` |
| `POST` | `/api/sessions` | Start pipeline; body matches `GenerateRequest` |
| `GET` | `/api/sessions/{id}/status` | `pending` / `running` / `complete` / `error` |
| `WS` | `/api/ws/{session_id}` | Progress logs and final `pipeline_complete` |

Static renders and downloads are served under `/output/{session_id}/...`.

---

## Tests

```powershell
pytest tests -q --ignore=tests/test_08_e2e_pipeline.py
```

Slow end-to-end Runway tests live in `tests/test_08_e2e_pipeline.py`. Integration tests may expect the server on port **8000** (`tests/test_07_server.py`).

---

## Repository layout

| Path | Role |
|------|------|
| `.env.example` | Template for required env vars; copy to `.env` locally |
| `.gitignore` | Ignores `.env`, `output/`, caches, venvs — see file for full list |
| `server.py` | FastAPI app, sessions, WebSocket fan-out |
| `pipeline/orchestrator.py` | Stage scheduler, stitching, checkpoint hooks |
| `pipeline/state.py` | `PipelineState`, platforms, model credit hints |
| `pipeline/checkpoint.py` | Fingerprints, resume, media copy |
| `pipeline/runway_client.py` | Runway HTTP client |
| `pipeline/end_card.py` | Promo clip animation, crossfade, fallbacks |
| `pipeline/agents/*.py` | One module per stage (+ `end_card_gen.py`) |
| `static/` | Frontend |
| `output/` | Per-session artifacts (created at runtime) |

---

## Pipeline stages (ordered)

| ID | Name | Agent / module | Main outputs |
|----|------|----------------|--------------|
| **1** | Intent Ingestion | `ingestion` | Structured `enriched_brief` from user prompt |
| **2** | Contextual Enhancement | `enhancer` | Creative director fields added to brief |
| **3** | Narrative Sequencing | `story` | `storyboard` shots with durations summing to campaign length |
| **4** | Prompt Engineering | `prompt_gen` | Per-shot Runway image + motion prompts, camera params |
| **5** | Structural Validation | `validator` | LLM validation / fixes on structure |
| **6** | Visual Synthesis | `image_synth` | `images/{shot}_*.jpg`, anchor world lock |
| **6a** | VLM Validation | `vlm_validator` | Vision-language checks / retries |
| **7** | Temporal Synthesis | `video_gen` | `videos/{shot}.mp4` |
| **8** | Non-linear stitching | `end_card_gen`, `end_card`, orchestrator `_stitch` | `final_body.mp4`, optional AI slate, `final.mp4` |
| **9** | Audio production | `audio_gen` | Voiceover, music, ambients, `final_with_audio.mp4` |

---

## Flow diagram — full pipeline

Each subgraph matches one checkpoint stage. Data flows **downstream** through `PipelineState`.

```mermaid
flowchart TB
  subgraph IN["Inputs"]
    U[User prompt]
    IMG[Optional reference image data_uri]
  end

  subgraph S1["Stage 1 — Intent Ingestion"]
    S1a[GPT: parse brief]
    S1b["enriched_brief JSON"]
    S1a --> S1b
  end

  subgraph S2["Stage 2 — Contextual Enhancement"]
    S2a[GPT: creative director pass]
    S2b[Richer brief fields]
    S2a --> S2b
  end

  subgraph S3["Stage 3 — Narrative Sequencing"]
    S3a[GPT: shot list + durations]
    S3b["storyboard: Shot objects"]
    S3a --> S3b
  end

  subgraph S4["Stage 4 — Prompt Engineering"]
    S4a[GPT: image_prompt + motion_prompt per shot]
    S4b[Camera parameters]
    S4a --> S4b
  end

  subgraph S5["Stage 5 — Structural Validation"]
    S5a[GPT: validate / repair structure]
    S5b[Ready prompts + shots]
    S5a --> S5b
  end

  subgraph S6["Stage 6 — Visual Synthesis"]
    S6a[Runway: anchor frame]
    S6b[GPT-4o Vision: world document]
    S6c[Runway: remaining frames with @anchor + world lock]
    S6a --> S6b --> S6c
  end

  subgraph S6a2["Stage 6a — VLM Validation"]
    V1[GPT Vision + Runway checks]
    V2[Retried frames if needed]
    V1 --> V2
  end

  subgraph S7["Stage 7 — Temporal Synthesis"]
    VV[Runway: image-to-video per shot]
    CL["videos/*.mp4"]
    VV --> CL
  end

  subgraph S8["Stage 8 — Non-linear stitching"]
    EC[end_card_gen: GPT + Runway optional promo_end_professional.jpg]
    ST[FFmpeg: concat clips to final_body.mp4]
    PR[end_card: animated promo + crossfade into final.mp4]
    EC --> ST --> PR
  end

  subgraph S9["Stage 9 — Audio Production"]
    A1[GPT: VO script + music brief]
    A2[Runway TTS + SFX / music / ambients]
    A3[FFmpeg mix + mux to final_with_audio.mp4]
    A1 --> A2 --> A3
  end

  U --> S1a
  IMG --> S1a
  S1b --> S2a --> S3a --> S4a --> S5a --> S6a
  S6c --> V1 --> VV --> EC
  PR --> A1

  classDef stage fill:#1e293b,stroke:#64748b,color:#e2e8f0;
  class S1,S2,S3,S4,S5,S6,S6a2,S7,S8,S9 stage;
```

---

## Stage 8 detail (end card + assembly)

Stage 8 combines an **optional AI-generated closing slate**, **clip stitching**, and **cinematic merge** into the main timeline.

```mermaid
flowchart LR
  subgraph PRE["Prerequisites"]
    VIDS["videos/*.mp4 complete"]
    KEYS["RUNWAY + OPENAI for AI slate"]
  end

  subgraph GEN["End-card mini-pipeline"]
    G1[GPT-4o: Runway prompt JSON]
    G2[Runway text-to-image]
    G3["images/promo_end_professional.jpg"]
    G1 --> G2 --> G3
  end

  subgraph ASM["FFmpeg assembly"]
    F1["Concat → final_body.mp4"]
    F2["Resolve hero: AI slate → last still → upload → frame grab"]
    F3["Build promo clip: zoom + fades ± ASS overlay"]
    F4["Crossfade body tail into promo OR hard concat"]
    F5["final.mp4"]
    F1 --> F2 --> F3 --> F4 --> F5
  end

  subgraph PAD["state.playback_padding_seconds"]
    P1["Net extra duration for audio mux"]
  end

  VIDS --> F1
  KEYS -.->|optional| GEN
  GEN -.-> F2
  F4 --> PAD

  classDef muted fill:#334155,stroke:#94a3b8,color:#f8fafc;
  class PRE,PAD muted;
```

---

## Checkpoint and resume

Runs persist state under `output/{session_id}/checkpoint.json`. A **fingerprint** hashes prompt, image digest, platform, duration, models, and ratio. Matching prior sessions can **skip** completed stages or **replay** a finished run without new API calls.

```mermaid
flowchart TB
  FP[Compute fingerprint]
  SCAN[Scan sibling output/*/checkpoint.json]
  MATCH{Match found?}
  COPY[copy_media_tree + merge state]
  IDX[Compute start_idx from last_completed_stage]
  RUN[Execute stages start_idx .. 9]
  SHORT{Fully finished prior run?}
  REPLAY[Emit skipped logs only]

  FP --> SCAN --> MATCH
  MATCH -->|no| RUN
  MATCH -->|yes| SHORT
  SHORT -->|yes| REPLAY
  SHORT -->|no| COPY --> IDX --> RUN
```

---

## Per-stage Mermaid diagrams

Each diagram shows the **primary inputs → processing → outputs** for that checkpoint stage.

### Stage 1 — Intent Ingestion

```mermaid
flowchart LR
  P[User prompt] --> I[GPT JSON extraction]
  R[Optional ref image] --> I
  I --> B[enriched_brief]
```

### Stage 2 — Contextual Enhancement

```mermaid
flowchart LR
  B[enriched_brief] --> E[GPT creative director]
  E --> B2[Brief + tone / scenes / platform notes]
```

### Stage 3 — Narrative Sequencing

```mermaid
flowchart LR
  B[enriched_brief] --> S[GPT storyboard JSON]
  D[Fixed duration budget] --> S
  S --> SB[List of Shot + durations]
```

### Stage 4 — Prompt Engineering

```mermaid
flowchart LR
  SB[Storyboard shots] --> G[GPT prompt engineer]
  G --> IP[image_prompt per shot]
  G --> MP[motion_prompt per shot]
  G --> CP[camera params]
```

### Stage 5 — Structural Validation

```mermaid
flowchart LR
  SB[Shots + prompts] --> V[GPT validator]
  V --> OK[Repaired / confirmed structure]
```

### Stage 6 — Visual Synthesis

```mermaid
flowchart LR
  subgraph A["Phase A — anchor"]
    R1[Runway text-to-image] --> IMG1[shot_01.jpg]
  end
  subgraph W["World lock"]
    IMG1 --> V[GPT-4o Vision world doc]
  end
  subgraph B2["Phase B — others"]
    V --> R2[Runway + @anchor refs]
    H[@hero_ref if any] --> R2
    R2 --> IMGN[shot_02..N.jpg]
  end
```

### Stage 6a — VLM Validation

```mermaid
flowchart LR
  IM[Generated frames] --> VL[GPT Vision + policy]
  VL --> PASS[Accept or regenerate subset]
```

### Stage 7 — Temporal Synthesis

```mermaid
flowchart LR
  IM[Shot images] --> RV[Runway image-to-video]
  RV --> VP[videos/shot_XX.mp4]
```

### Stage 8 — Non-linear stitching

```mermaid
flowchart TB
  subgraph G["Optional AI slate"]
    GPT[GPT-4o prompt plan] --> RW[Runway image]
    RW --> SL[promo_end_professional.jpg]
  end
  VP[videos/*.mp4] --> C[FFmpeg concat copy]
  C --> FB[final_body.mp4]
  SL -.-> HERO[Hero resolve + motion promo MP4]
  IMGS[Fallback stills / frame] -.-> HERO
  FB --> XF[Crossfade or concat]
  HERO --> XF
  XF --> FIN[final.mp4]
```

### Stage 9 — Audio Production

```mermaid
flowchart LR
  B[enriched_brief + shots] --> SC[GPT VO script + music brief]
  SC --> TTS[Runway TTS]
  SC --> SFX[Runway SFX / music / ambients]
  TTS --> MIX[FFmpeg mix AAC]
  SFX --> MIX
  MIX --> MUX[FFmpeg mux + final_with_audio.mp4]
  FIN[final.mp4 video] --> MUX
```

---

## Credits and cost accounting

Rough credit estimates per model live in `pipeline/state.py` (`CREDIT_COST`). Totals accumulate on `PipelineState` during generation.

---

## License / extras

The `skills/` directory contains optional Cursor / Claude plugin assets for Runway documentation and helpers; it is separate from the core FastAPI pipeline.

---

## Troubleshooting

- **FFmpeg not found**: Install FFmpeg and ensure `ffmpeg` / `ffprobe` resolve (PATH or WinGet/Chocolatey paths per orchestrator).
- **Stage 8 promo skipped**: Usually missing hero image and failed frame extract; ensure ffprobe available for crossfade timing on Windows.
- **WebSocket disconnect noise on Windows**: Abrupt tab closes can log `ConnectionResetError`; the pipeline may still complete server-side.
