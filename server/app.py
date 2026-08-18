"""FastAPI server exposing the agent (agent/orchestrator.py) over HTTP with
token-by-token streaming (SSE), plus the single-page frontend at "/" --
the proper UI layer for the malayalam-llm project, as opposed to
app/streamlit_app.py's debug-only demo. Not tied to any particular
checkpoint; point CHECKPOINT at whichever one you want to serve.

Usage (inside the docker container, on Shannon):
    uvicorn server.app:app --host 0.0.0.0 --port 8000

Config via environment variables (so `docker compose up api` doesn't need
argv threading through a fixed container command):
    CHECKPOINT -- default checkpoints/instruct/malayalam_assistant.pt
    TOKENIZER  -- default tokenizer/malayalam_tokenizer.json
    DEVICE     -- default "cuda" if available else "cpu"; force "cpu" to
                  test this server without touching a GPU a training run is
                  already using, e.g.:
                      docker compose run --rm -e DEVICE=cpu -p 8000:8000 \
                          cpu uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.orchestrator import run_turn_stream  # noqa: E402
from agent.tools import AGENT_SYSTEM_PROMPT  # noqa: E402
from model.checkpoint import load_model  # noqa: E402

from tokenizers import Tokenizer

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/instruct/malayalam_assistant.pt")
TOKENIZER_PATH = os.environ.get("TOKENIZER", "tokenizer/malayalam_tokenizer.json")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

print(f"[server] loading checkpoint={CHECKPOINT} tokenizer={TOKENIZER_PATH} device={DEVICE}")
tok = Tokenizer.from_file(TOKENIZER_PATH)
model, cfg, ckpt_system_prompt = load_model(CHECKPOINT, device=DEVICE)
system_prompt = ckpt_system_prompt or AGENT_SYSTEM_PROMPT
print("[server] model loaded, ready")

app = FastAPI(title="Ammini")


class Message(BaseModel):
    role: str  # "user" | "assistant" -- system prompt is server-controlled, not client-supplied
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 40


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "checkpoint": CHECKPOINT}


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        return StreamingResponse(
            iter([_sse({"type": "error", "message": "last message must have role=user"})]),
            media_type="text/event-stream",
        )

    history = [(m.role, m.content) for m in req.messages[:-1]]
    user_text = req.messages[-1].content

    def event_stream():
        try:
            for event in run_turn_stream(model, tok, cfg, system_prompt, history, user_text,
                                          req.max_new_tokens, req.temperature, req.top_k):
                yield _sse(event)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
