import json
import logging
from pathlib import Path
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_chunks")

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "mgi_chroma_db"
DATA_DIR = BASE_DIR / "backend" / "data"
OUTPUT_FILE = DATA_DIR / "chunks.json"

def export_chroma_chunks():
    """Extract all indexed chunks from ChromaDB and save as clean JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("mgi_policy_handbook")

    count = collection.count()
    logger.info(f"Total documents in collection: {count}")

    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    chunks = []
    for doc_id, text, meta in zip(ids, docs, metas):
        meta = meta or {}
        chunks.append({
            "id": doc_id,
            "text": text,
            "chapter": meta.get("chapter", "Handbook"),
            "policy": meta.get("policy", "Policy"),
            "section": meta.get("section", "Section"),
            "page": meta.get("page", 1),
            "doc_type": meta.get("doc_type", "policy"),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully exported {len(chunks)} chunks to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
    return len(chunks)

if __name__ == "__main__":
    export_chroma_chunks()

