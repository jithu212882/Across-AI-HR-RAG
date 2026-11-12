import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# File Paths
PDF_PATH = BASE_DIR / "MGI_HR_Policy_Handbook_150pages.pdf"
FRONTEND_DIR = BASE_DIR / "frontend"
CHUNKS_DATA_PATH = BASE_DIR / "backend" / "data" / "chunks.json"

# Database & Storage Settings (supports pre-indexed JSON or ChromaDB fallback)
CHROMA_DB_PATH = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", os.getenv("CHROMA_DB_PATH", "./mgi_chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", os.getenv("COLLECTION_NAME", "mgi_policy_handbook"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Hyperparameters
DENSE_K = int(os.getenv("DENSE_K", "6"))
BM25_K = int(os.getenv("BM25_K", "6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# LLM Configurations (Groq / OpenAI-compatible Router)
def _resolve_llm_provider(base_url: str, api_key: str) -> str:
    """Resolve LLM provider, defaulting to 'groq' for Groq URLs or gsk_ keys."""
    provider = os.getenv("LLM_PROVIDER", "").strip().strip('"\'')
    if provider:
        return provider
    if "groq.com" in base_url.lower() or api_key.startswith("gsk_"):
        return "groq"
    return "groq"

raw_base_url = (os.getenv("LLM_BASE_URL") or "").strip() or "https://api.groq.com/openai/v1"
# Strip markdown links if present in .env
if "[" in raw_base_url and "](" in raw_base_url:
    raw_base_url = raw_base_url.split("](")[1].rstrip(")")
LLM_BASE_URL = raw_base_url.strip().strip('"\'').strip() or "https://api.groq.com/openai/v1"

LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip().strip('"\'').strip()
LLM_PROVIDER = _resolve_llm_provider(LLM_BASE_URL, LLM_API_KEY)
LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip().strip('"\'').strip() or "llama-3.3-70b-versatile"

# Fallback Models list
raw_fallbacks = os.getenv("LLM_FALLBACK_MODELS", "llama-3.1-8b-instant,mixtral-8x7b-32768").strip('"\'')
FALLBACK_MODELS = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]
# Add working Groq model fallback in case Groq 404s the legacy name
if "qwen/qwen3.8-27b" not in FALLBACK_MODELS:
    FALLBACK_MODELS.append("qwen/qwen3.8-27b")

# Anti-Hallucination Generation Parameters
TEMPERATURE = 0.0
TOP_P = 0.1
FREQUENCY_PENALTY = 0.1
MAX_TOKENS = 650

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

def reload_env():
    """Reload environment variables if .env was updated at runtime without clobbering runtime secrets."""
    global LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, FALLBACK_MODELS, CHROMA_DB_PATH, COLLECTION_NAME
    try:
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=False)
    except Exception:
        pass

    raw_url = (os.getenv("LLM_BASE_URL") or "").strip() or "https://api.groq.com/openai/v1"
    if "[" in raw_url and "](" in raw_url:
        raw_url = raw_url.split("](")[1].rstrip(")")
    LLM_BASE_URL = raw_url.strip().strip('"\'').strip() or "https://api.groq.com/openai/v1"

    LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip().strip('"\'').strip()
    LLM_PROVIDER = _resolve_llm_provider(LLM_BASE_URL, LLM_API_KEY)
    LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip().strip('"\'').strip() or "llama-3.3-70b-versatile"

    raw_fb = os.getenv("LLM_FALLBACK_MODELS", "llama-3.1-8b-instant,mixtral-8x7b-32768").strip('"\'')
    FALLBACK_MODELS = [m.strip() for m in raw_fb.split(",") if m.strip()]
    if "qwen/qwen3.8-27b" not in FALLBACK_MODELS:
        FALLBACK_MODELS.append("qwen/qwen3.8-27b")

    CHROMA_DB_PATH = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", os.getenv("CHROMA_DB_PATH", "./mgi_chroma_db"))
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", os.getenv("COLLECTION_NAME", "mgi_policy_handbook"))
