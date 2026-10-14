import asyncio
import base64
import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(override=True)  # .env always takes precedence over shell env vars

from pipeline.orchestrator import PipelineOrchestrator

log = logging.getLogger("adforge")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIST = STATIC_DIR / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = FRONTEND_DIST / "index.html"
    assets = FRONTEND_DIST / "assets"
    if not index.is_file():
        log.warning(
            "Frontend not built: missing %s — run `npm run build` in `frontend/`",
            index,
        )
    elif not assets.is_dir() or not any(assets.iterdir()):
        log.warning(
            "Frontend build incomplete: %s missing or empty — page may load without JS/CSS",
            assets,
        )
    yield


app = FastAPI(title="AdForge - Generative Advertising Pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Any localhost / 127.0.0.1 port (5173 Vite, 8000/8899 Uvicorn, etc.)
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

_vite_assets = FRONTEND_DIST / "assets"
if _vite_assets.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_vite_assets.resolve())), name="vite_assets")

sessions: dict = {}


@app.middleware("http")
async def log_unhandled_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        log.exception("Unhandled exception for %s %s", request.method, request.url.path)
        raise


class GenerateRequest(BaseModel):
    prompt: str
    image_data_uri: Optional[str] = None
    platform: str = "youtube"
    quality_mode: str = "preview"
    total_duration: int = 30
    video_model: Optional[str] = None   # explicit override; falls back to quality_mode mapping
    image_model: Optional[str] = None   # explicit override; defaults to gen4_image
    interactive: bool = True            # pause for user approval after key stages


@app.get("/")
async def root():
    index = (FRONTEND_DIST / "index.html").resolve()
    if not index.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built. Run `npm run build` in the `frontend/` directory.",
        )
    try:
        return FileResponse(
            path=str(index),
            media_type="text/html",
            filename="index.html",
        )
    except OSError as e:
        log.exception("Failed to read %s", index)
        raise HTTPException(
            status_code=500,
            detail=f"Could not serve frontend index: {e}",
        ) from e


@app.get("/health")
async def health():
    """Lightweight probe; does not touch Runway or OpenAI."""
    ok = (FRONTEND_DIST / "index.html").is_file()
    return {"status": "ok", "frontend_index": ok, "dist": str(FRONTEND_DIST)}


@app.get("/api/debug/config")
async def debug_config():
    """Optional: confirm paths when ADFORGE_DEBUG=1 (avoid leaking secrets)."""
    import os

    if os.environ.get("ADFORGE_DEBUG", "").strip() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "base_dir": str(BASE_DIR),
        "frontend_dist": str(FRONTEND_DIST.resolve()),
        "index_exists": (FRONTEND_DIST / "index.html").is_file(),
        "assets_mounted": _vite_assets.is_dir(),
    }


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    mime_type = file.content_type or "image/jpeg"
    b64 = base64.b64encode(content).decode()
    data_uri = f"data:{mime_type};base64,{b64}"
    return {"data_uri": data_uri, "filename": file.filename, "size": len(content)}


@app.post("/api/sessions")
async def create_session(request: GenerateRequest):
    session_id = str(uuid.uuid4())[:8]
    session_dir = OUTPUT_DIR / session_id
    (session_dir / "images").mkdir(parents=True, exist_ok=True)
    (session_dir / "videos").mkdir(parents=True, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue()
    approval_event = asyncio.Event()
    sessions[session_id] = {
        "status": "pending",
        "queue": queue,
        "approval_event": approval_event,
        "approval_response": None,
    }

    asyncio.create_task(_run_pipeline(session_id, request, session_dir, queue))
    return {"session_id": session_id}


async def _run_pipeline(
    session_id: str,
    request: GenerateRequest,
    session_dir: Path,
    queue: asyncio.Queue,
):
    async def send(msg: dict):
        await queue.put(msg)

    try:
        sessions[session_id]["status"] = "running"
        orchestrator = PipelineOrchestrator(session_id, session_dir, send)

        async def wait_for_approval() -> dict:
            """Block pipeline until user approves/edits via WebSocket."""
            session = sessions[session_id]
            session["approval_event"].clear()
            session["approval_response"] = None
            await session["approval_event"].wait()
            return session["approval_response"] or {"action": "approve"}

        orchestrator.wait_for_approval = wait_for_approval

        result = await orchestrator.run(
            prompt=request.prompt,
            image_data_uri=request.image_data_uri,
            platform=request.platform,
            quality_mode=request.quality_mode,
            total_duration=request.total_duration,
            video_model=request.video_model,
            image_model=request.image_model,
            interactive=request.interactive,
        )
        sessions[session_id]["status"] = "complete"
        await queue.put({"type": "pipeline_complete", **result})
    except Exception as exc:
        log.exception("Pipeline session %s failed", session_id)
        sessions[session_id]["status"] = "error"
        await queue.put(
            {
                "type": "pipeline_error",
                "message": str(exc),
                "detail": traceback.format_exc(),
            }
        )
    finally:
        await queue.put(None)


@app.websocket("/api/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    session = sessions[session_id]
    queue = session["queue"]

    async def sender():
        """Push pipeline events to the client."""
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                    continue

                if msg is None:
                    await websocket.send_json({"type": "done"})
                    return

                await websocket.send_json(msg)
        except (WebSocketDisconnect, Exception):
            pass

    async def receiver():
        """Listen for client messages (approval responses)."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                if msg.get("type") == "approval_response":
                    session["approval_response"] = msg
                    session["approval_event"].set()
                elif msg.get("type") == "skip_all_reviews":
                    session["approval_response"] = {"action": "skip_all"}
                    session["approval_event"].set()
        except (WebSocketDisconnect, Exception):
            pass

    try:
        await asyncio.gather(sender(), receiver())
    except Exception:
        pass


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": sessions[session_id]["status"]}
