# Across AI — Enterprise HR Assistant

Across AI is a production-grade, conversational HR assistant built for Meridian Global Industries (MGI). It delivers strict, zero-hallucination answers regarding company policies, leave matrices, band/grade structures, and employee guidelines through a hybrid RAG pipeline and an obsidian-violet glassmorphic web interface.

---

## Architecture Overview

- **Frontend:** Responsive vanilla HTML5/CSS3 glassmorphic interface with Markdown rendering via Marked.js and sanitization via DOMPurify.
- **Backend API:** FastAPI application providing REST and Server-Sent Events (SSE) streaming endpoints.
- **Retrieval Pipeline:** Hybrid search combining dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via persistent ChromaDB and lexical keyword scoring via BM25Okapi, with exact alphanumeric code boosting.
- **LLM Reasoning:** OpenAI SDK connected to high-performance inference providers (Groq / Hugging Face / OpenAI compatible routers) with automatic fallback chains and anti-hallucination system constraints.

---

## Directory Structure

```
├── backend/
│   ├── __init__.py
│   ├── config.py           # Environment variables & runtime settings
│   ├── ingest.py           # PDF parsing, structural chunking & vector indexing
│   ├── llm_service.py      # LLM orchestration, persona management & streaming
│   ├── main.py             # FastAPI REST and SSE streaming endpoints
│   ├── retriever.py        # Hybrid Dense + BM25Okapi retriever
│   └── test_suite.py       # End-to-end unit and integration test suite
├── frontend/
│   └── index.html          # Obsidian-violet glassmorphic desktop interface
├── .env.example            # Environment configuration template
├── .gitignore              # Sensitive files & secret prevention
├── vercel.json             # Vercel static site routing configuration
├── .vercelignore           # Excludes heavy Python/ML dependencies from Vercel
├── pyproject.toml          # Static project descriptor
├── render.yaml             # Render Blueprint web service definition
├── Dockerfile              # Containerized production runtime
├── .dockerignore           # Container build exclusions
├── MGI_HR_Policy_Handbook_150pages.pdf  # Source knowledge handbook
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Production Deployment (Vercel + Render)

### 1. Backend on Render
Deploy the API and ChromaDB vector engine to Render:
1. Connect your GitHub repository (`jithu212882/Across-AI-HR-RAG`) on [Render](https://render.com).
2. Choose **Web Service** or use the included Blueprint (`render.yaml`).
3. Set **Build Command:**
   ```bash
   pip install --upgrade pip && pip install -r requirements.txt && python -m backend.ingest
   ```
4. Set **Start Command:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add your environment variables (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`).
6. Copy your deployed Render service URL (e.g. `https://across-ai-backend.onrender.com`).

### 2. Frontend on Vercel
Deploy the static glassmorphic UI to Vercel:
1. Import your GitHub repository (`jithu212882/Across-AI-HR-RAG`) on [Vercel](https://vercel.com).
2. Vercel automatically uses `vercel.json` and `.vercelignore` to serve `frontend/` as static assets without building heavy Python bundles.
3. Open your deployed Vercel site, click **Settings**, and paste your Render service URL into the **Render Backend URL** field. Across AI will connect to your cloud API with full streaming support.

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone <repo-url>
cd <repo-folder>

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the template and provide your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your preferred model and credentials:

```env
LLM_BASE_URL="https://api.groq.com/openai/v1"
LLM_API_KEY="your-api-key-here"
LLM_MODEL="llama-3.3-70b-versatile"
LLM_FALLBACK_MODELS="qwen/qwen3.8-27b,llama-3.1-8b-instant"
CHROMA_PERSIST_DIR="./mgi_chroma_db"
CHROMA_COLLECTION="mgi_policy_handbook"
```

### 4. Build the Vector Index
Extract and index the handbook into ChromaDB:

```bash
python -m backend.ingest
```

### 5. Launch the Application
Start the Uvicorn web server:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000` to interact with Across AI.

---

## Running Automated Tests

Run the full end-to-end test suite:

```bash
python -m unittest backend.test_suite
```

---

## Security & Privacy Notice
Sensitive configuration files (`.env`), API keys, local databases, and temporary build files are strictly ignored via `.gitignore`. Never commit API keys or production secrets to version control.
