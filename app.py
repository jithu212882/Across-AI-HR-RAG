import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import spaces
import gradio as gr
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.main import app as fastapi_app

@spaces.GPU
def gpu_warmup(msg: str = "ready") -> str:
    return f"Across AI ZeroGPU: {msg}"

# Serve the custom obsidian-violet HTML frontend at /ui
@fastapi_app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

# Serve any static assets under /static if present
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    try:
        fastapi_app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")
    except Exception:
        pass

# Render /ui full-viewport inside Gradio's root page so Gradio never blocks /
with gr.Blocks(
    title="Across AI",
    css="footer {display:none !important;} .gradio-container {padding:0 !important; max-width:100% !important;}"
) as demo:
    gr.HTML('<iframe src="/ui" style="width:100vw; height:100vh; border:none; position:fixed; top:0; left:0; z-index:9999;"></iframe>')
    box = gr.Textbox(value="ready", visible=False)
    demo.load(fn=gpu_warmup, inputs=box, outputs=box)

def _mount_fastapi_routes(target_app):
    """Mount all FastAPI routes (/api/*, /ui, /static) onto Gradio's internal FastAPI app."""
    for route in fastapi_app.routes:
        if getattr(route, "path", "").startswith(("/api", "/ui", "/static")):
            if route not in target_app.router.routes:
                target_app.router.routes.insert(0, route)

_mount_fastapi_routes(demo.app)

_orig_create_app = gr.routes.App.create_app
def _patched_create_app(*args, **kwargs):
    runtime_app = _orig_create_app(*args, **kwargs)
    _mount_fastapi_routes(runtime_app)
    return runtime_app
gr.routes.App.create_app = _patched_create_app

app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
