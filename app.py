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

iframe_html = '<iframe src="/ui" style="width:100vw;height:100vh;border:none;position:fixed;inset:0;z-index:9999;"></iframe>'

with gr.Blocks(title="Across AI") as demo:
    gr.HTML(iframe_html)
    box = gr.Textbox(value="ready", visible=False)
    btn = gr.Button("Warmup", visible=False)
    btn.click(fn=gpu_warmup, inputs=box, outputs=box)

def _get_index_file():
    root_index = BASE_DIR / "index.html"
    frontend_index = BASE_DIR / "frontend" / "index.html"
    if root_index.exists():
        return root_index
    elif frontend_index.exists():
        return frontend_index
    return BASE_DIR / "index.html"

def _attach_routes(target_app):
    """Attach fastapi_app routes and /ui to the launched Gradio FastAPI app."""
    target_app.include_router(fastapi_app.router)

    @target_app.get("/ui", include_in_schema=False)
    async def serve_ui():
        return FileResponse(_get_index_file())

    frontend_dir = BASE_DIR / "frontend"
    if frontend_dir.exists():
        try:
            target_app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")
        except Exception:
            pass

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        ssr_mode=False,
        prevent_thread_lock=True,
    )
    _attach_routes(demo.app)
    demo.block_thread()
