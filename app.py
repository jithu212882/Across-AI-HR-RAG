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

# Include all /api routes from FastAPI app directly onto Gradio's internal FastAPI server
demo.app.include_router(fastapi_app.router)
# Also mount for sub-path fallback
demo.app.mount("/api", fastapi_app)

# Replace default Gradio root GET route with our custom obsidian-violet HTML UI
demo.app.router.routes = [
    r for r in demo.app.router.routes
    if not (hasattr(r, "path") and r.path == "/" and "GET" in r.methods)
]

@demo.app.get("/", include_in_schema=False)
async def serve_custom_frontend():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

# Expose app symbol for ASGI runners
app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
