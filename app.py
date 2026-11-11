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
def gpu_warmup(msg: str = "ready") -> str:
    return f"Across AI ZeroGPU: {msg}"

# Read index.html directly in Python
html_path = BASE_DIR / "index.html"
if not html_path.exists():
    html_path = BASE_DIR / "frontend" / "index.html"

html_content = html_path.read_text(encoding="utf-8")

# Script shim to ensure scripts in innerHTML execute in modern browsers
script_shim = (
    '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" '
    'style="display:none;" onload="'
    'document.querySelectorAll(\'script\').forEach(function(s){'
    'if(!s.dataset.executed){'
    's.dataset.executed=\'true\';'
    'var ns=document.createElement(\'script\');'
    'Array.from(s.attributes).forEach(function(a){ns.setAttribute(a.name,a.value);});'
    'ns.text=s.text;'
    'document.body.appendChild(ns);'
    '}'
    '});">'
)

with gr.Blocks(title="Across AI") as demo:
    gr.HTML(html_content + script_shim)
    box = gr.Textbox(value="ready", visible=False)
    btn = gr.Button("Warmup", visible=False)
    btn.click(fn=gpu_warmup, inputs=box, outputs=box)

# Initialize demo config so App.create_app works cleanly
custom_css = "footer {display:none !important;} .gradio-container {padding:0 !important; max-width:100% !important; margin:0 !important;}"
demo.config = demo.get_config_file()
demo.theme = getattr(demo, "theme", None) or gr.themes.Default()
demo.css = custom_css
demo.js = getattr(demo, "js", "") or ""
demo.head = getattr(demo, "head", "") or ""
demo.head_paths = getattr(demo, "head_paths", []) or []
demo.css_paths = getattr(demo, "css_paths", []) or []
if hasattr(demo, "_set_html_css_theme_variables"):
    demo._set_html_css_theme_variables()

# Attach all /api/* routes from backend.main.app onto Gradio's FastAPI app
app = gr.routes.App.create_app(demo)
for route in fastapi_app.routes:
    if getattr(route, "path", "").startswith("/api"):
        if route not in app.router.routes:
            app.router.routes.insert(0, route)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
