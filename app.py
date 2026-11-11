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
def gpu_health_check(text: str = "ready") -> str:
    return f"Across AI ZeroGPU: {text}"

with gr.Blocks(title="Across AI") as demo:
    gr.Markdown("### Across AI Service")
    status_box = gr.Textbox(value="ready", visible=False)
    check_btn = gr.Button("Check", visible=False)
    check_btn.click(fn=gpu_health_check, inputs=status_box, outputs=status_box)
    demo.load(fn=gpu_health_check, inputs=status_box, outputs=status_box)

# Mount Gradio at /gradio while keeping custom HTML UI at root (/)
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    if not os.environ.get("SPACE_ID"):
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
