import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import (
    BASE_DIR,
    FRONTEND_DIR,
    ENV_PATH,
    reload_env,
)
from backend.llm_service import get_llm_service
from backend.retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

app = FastAPI(
    title="Across AI - Enterprise HR Assistant",
    description="Conversational HR AI Assistant for Meridian Global Industries (MGI)",
    version="2.0.0",
)

# Enable CORS for local desktop usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatTurn(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Turn text content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or question")
    history: Optional[List[ChatTurn]] = Field(default=[], description="Multi-turn conversation history")

class ConfigUpdateRequest(BaseModel):
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None

# API Endpoints
@app.get("/api/health")
async def health_check():
    """System health status."""
    return {
        "status": "online",
        "app_name": "Across AI",
        "version": "2.0.0",
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main conversational endpoint:
    - Multi-turn history support
    - Hybrid RAG retrieval
    - Natural persona generation with Groq / Gemini
    - Clean citation metadata
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    llm_service = get_llm_service()
    history_dicts = [{"role": t.role, "content": t.content} for t in request.history]

    try:
        response_data = llm_service.generate_response(
            query=request.message,
            history=history_dicts,
        )
        return response_data
    except Exception as e:
        logger.error(f"Chat generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Real-time streaming conversational endpoint (Server-Sent Events):
    - Streams tokens as they are generated
    - Emits citations and metadata as first event
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    llm_service = get_llm_service()
    history_dicts = [{"role": t.role, "content": t.content} for t in request.history]

    def event_stream():
        try:
            for item in llm_service.stream_response(query=request.message, history=history_dicts):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/config")
async def update_config(config: ConfigUpdateRequest):
    """Update API keys or models dynamically at runtime."""
    env_updates = {}
    if config.llm_api_key is not None:
        env_updates["LLM_API_KEY"] = config.llm_api_key.strip()
    if config.llm_base_url is not None and config.llm_base_url.strip():
        env_updates["LLM_BASE_URL"] = config.llm_base_url.strip()
    if config.llm_model is not None and config.llm_model.strip():
        env_updates["LLM_MODEL"] = config.llm_model.strip()

    # Read existing .env
    existing_lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        match = False
        for k, v in env_updates.items():
            if line.startswith(f"{k}="):
                new_lines.append(f"{k}={v}\n")
                updated_keys.add(k)
                match = True
                break
        if not match:
            new_lines.append(line)

    for k, v in env_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Reload variables in memory
    reload_env()
    for k, v in env_updates.items():
        os.environ[k] = v

    llm_service = get_llm_service()
    new_provider = llm_service.get_active_provider()

    return {
        "status": "success",
        "message": "Configuration updated successfully.",
        "active_provider": new_provider,
    }

@app.get("/api/handbook/stats")
async def handbook_stats():
    """Metadata regarding handbook coverage."""
    return {
        "edition": "Global Edition 2027",
        "effective_date": "01 January 2027",
        "chapters_count": 12,
        "forms_count": 26,
        "faqs_count": 30,
        "annexures": [
            {"id": "A", "title": "Band and Grade Structure"},
            {"id": "B", "title": "Leave Entitlement Matrix"},
            {"id": "C", "title": "HR Delegation of Authority"},
            {"id": "D", "title": "HR Service Standards"},
            {"id": "E", "title": "Contacts and Escalation Guide"},
            {"id": "F", "title": "Standard HR Forms (HR-F01 to HR-F26)"},
            {"id": "G", "title": "Manager Checklists (G1 to G10)"},
            {"id": "H", "title": "Frequently Asked Questions (Q1 to Q30)"},
            {"id": "I", "title": "Glossary of Terms"},
            {"id": "J", "title": "Document Revision History"},
        ],
    }

# Serve static frontend files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>Across AI Frontend Under Construction</h1>")
