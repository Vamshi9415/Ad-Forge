import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(override=True)  # .env always takes precedence over shell env vars

from pipeline.orchestrator import PipelineOrchestrator

app = FastAPI(title="AdForge - Generative Advertising Pipeline")

# Resolve to an absolute path so all downstream paths (concat.txt, etc.) are
# absolute — avoids FFmpeg doubling the path when it resolves relative entries
# in concat.txt relative to the concat file's own directory.
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")

app.mount("/output", StaticFiles(directory="output"), name="output")

sessions: dict = {}


class GenerateRequest(BaseModel):
    prompt: str
    image_data_uri: Optional[str] = None
    platform: str = "youtube"
    quality_mode: str = "preview"
    total_duration: int = 30
    video_model: Optional[str] = None   # explicit override; falls back to quality_mode mapping
    image_model: Optional[str] = None   # explicit override; defaults to gen4_image


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


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
    sessions[session_id] = {"status": "pending", "queue": queue}

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
        result = await orchestrator.run(
            prompt=request.prompt,
            image_data_uri=request.image_data_uri,
            platform=request.platform,
            quality_mode=request.quality_mode,
            total_duration=request.total_duration,
            video_model=request.video_model,
            image_model=request.image_model,
        )
        sessions[session_id]["status"] = "complete"
        await queue.put({"type": "pipeline_complete", **result})
    except Exception as exc:
        import traceback
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

    queue = sessions[session_id]["queue"]

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if msg is None:
                await websocket.send_json({"type": "done"})
                break

            await websocket.send_json(msg)
    except (WebSocketDisconnect, Exception):
        pass


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": sessions[session_id]["status"]}
