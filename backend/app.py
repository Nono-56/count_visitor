from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .state import LineCounterState
from .video import VideoProcessor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

state = LineCounterState()
video_processor = VideoProcessor(state)

app = FastAPI(title="Visitor Counter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    loop = asyncio.get_running_loop()
    state.set_event_loop(loop)
    LOGGER.info("Starting video processor...")
    video_processor.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    LOGGER.info("Stopping video processor...")
    video_processor.stop()


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class LinePayload(BaseModel):
    start: Point
    end: Point


@app.get("/status")
async def get_status() -> JSONResponse:
    return JSONResponse(state.to_dict())


@app.put("/line")
async def update_line(payload: LinePayload) -> JSONResponse:
    start = (payload.start.x, payload.start.y)
    end = (payload.end.x, payload.end.y)
    state.update_line(start, end)
    return JSONResponse(state.to_dict())


@app.get("/stream")
async def stream() -> StreamingResponse:
    boundary = "frame"
    return StreamingResponse(
        video_processor.mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = state.register_listener()
    try:
        await websocket.send_json(state.to_dict())
        while True:
            data = await queue.get()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("WebSocket error: %s", exc)
        await websocket.close(code=1011)
    finally:
        state.unregister_listener(queue)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": "Visitor counter backend running"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
