import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import spaces
import gradio as gr
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from backend.main import app as fastapi_app

@spaces.GPU
def gpu_warmup(msg: str = "ready") -> str:
    return f"Across AI ZeroGPU: {msg}"

# Patch Gradio App.create_app to ensure custom UI serves at root (/) even inside demo.launch()
_original_create_app = gr.routes.App.create_app

def _custom_create_app(*args, **kwargs):
    app = _original_create_app(*args, **kwargs)

    # Preserve Gradio's original index handler for /gradio
    original_main = None
    for r in app.router.routes:
        if getattr(r, "path", None) == "/" and "GET" in getattr(r, "methods", []):
            original_main = r.endpoint
            break

    # Remove Gradio's default '/' route so custom frontend takes priority
    app.router.routes = [
        r for r in app.router.routes
        if getattr(r, "path", None) not in ("/", "/index.html")
    ]

    # Mount all FastAPI routes (/api/chat, /api/chat/stream, /api/health, etc.)
    for route in fastapi_app.routes:
        if getattr(route, "path", "").startswith("/api"):
            app.router.routes.insert(0, route)

    # Serve custom obsidian-violet UI at root (/) supporting both GET and Gradio's HEAD health-probe
    @app.head("/", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    async def serve_custom_frontend():
        return FileResponse(BASE_DIR / "frontend" / "index.html")

    # Keep Gradio demo accessible at /gradio
    if original_main is not None:
        app.get("/gradio", include_in_schema=False, response_class=HTMLResponse)(original_main)

    # Mount static assets if present
    frontend_dir = BASE_DIR / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")

    return app

gr.routes.App.create_app = _custom_create_app

with gr.Blocks(title="Across AI") as demo:
    gr.Markdown("### Across AI Service Active")
    box = gr.Textbox(value="ready", visible=False)
    btn = gr.Button("Warmup", visible=False)
    btn.click(fn=gpu_warmup, inputs=box, outputs=box)
    demo.load(fn=gpu_warmup, inputs=box, outputs=box)

# Expose app symbol for ASGI runners
app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
