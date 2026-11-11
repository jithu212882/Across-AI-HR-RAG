import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import spaces
import gradio as gr
from backend.main import app as fastapi_app

@spaces.GPU
def gpu_warmup():
    return "Across AI Ready"

with gr.Blocks(title="Across AI") as demo:
    gr.Markdown("### Across AI Service Active")

# Mount Gradio at /gradio while keeping our custom HTML UI at /
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    if not os.environ.get("SPACE_ID"):
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
