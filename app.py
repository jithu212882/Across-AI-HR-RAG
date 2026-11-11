import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import spaces
import gradio as gr
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.main import app as fastapi_app

@spaces.GPU
def gpu_warmup(msg: str = "ready") -> str:
    return f"Across AI ZeroGPU: {msg}"

with gr.Blocks(title="Across AI") as demo:
    gr.Markdown("### Across AI Service Active")
    box = gr.Textbox(value="ready", visible=False)
    btn = gr.Button("Warmup", visible=False)
    btn.click(fn=gpu_warmup, inputs=box, outputs=box)
    demo.load(fn=gpu_warmup, inputs=box, outputs=box)

def configure_app_instance(target_app):
    """Mount /api routes, static files, and HTTP middleware intercepting root /."""
    # Mount all FastAPI /api/* routes
    for route in fastapi_app.routes:
        if getattr(route, "path", "").startswith("/api"):
            if route not in target_app.router.routes:
                target_app.router.routes.insert(0, route)

    # Serve static assets from frontend/ if present
    frontend_dir = BASE_DIR / "frontend"
    if frontend_dir.exists():
        target_app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")

    # Middleware intercepts exact GET / before Gradio's default UI route
    @target_app.middleware("http")
    async def serve_custom_root_ui(request: Request, call_next):
        if request.method == "GET" and request.url.path in ("/", "/index.html"):
            return FileResponse(BASE_DIR / "frontend" / "index.html")
        return await call_next(request)

# Configure the initial demo.app
configure_app_instance(demo.app)

# Patch Gradio App.create_app so that when demo.launch() creates the runtime app,
# the HTTP middleware and /api routes are preserved
_orig_create_app = gr.routes.App.create_app

def _custom_create_app(*args, **kwargs):
    runtime_app = _orig_create_app(*args, **kwargs)
    configure_app_instance(runtime_app)
    return runtime_app

gr.routes.App.create_app = _custom_create_app

# Expose app symbol for ASGI runners
app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
