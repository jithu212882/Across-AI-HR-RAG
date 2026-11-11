import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gradio as gr
from backend.main import app as fastapi_app

# Minimal dummy Gradio block mounted at /gradio so HF Space healthcheck passes
with gr.Blocks(title="Across AI Service") as dummy_ui:
    gr.Markdown("# Across AI Service\nCustom glassmorphic interface is running at the root URL.")

# Mount Gradio onto a sub-path (/gradio) while keeping our custom HTML UI at root (/)
app = gr.mount_gradio_app(fastapi_app, dummy_ui, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
