import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gradio as gr
from backend.main import app as fastapi_app

# Optional Hugging Face ZeroGPU hook
try:
    import spaces
    @spaces.GPU
    def _hf_gpu_hook():
        return "ok"
except Exception:
    pass

with gr.Blocks(title="Across AI") as demo:
    gr.Markdown("### Across AI Service Active\nCustom glassmorphic interface is running at the root URL.")

# Mount Gradio onto /gradio while keeping custom HTML UI at root (/)
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    # Prevent [Errno 98] port collision on Hugging Face Spaces where runner binds port 7860
    if not os.environ.get("SPACE_ID"):
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
