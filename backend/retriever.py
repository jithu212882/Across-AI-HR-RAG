import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNKS_DATA_PATH,
    PDF_PATH,
    DENSE_K,
    BM25_K,
    HYBRID_ALPHA,
)

logger = logging.getLogger("retriever")

class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (ChromaDB + SentenceTransformers)
    with sparse lexical search (BM25) and exact alphanumeric code boosting.
    Supports seamless automatic ingestion on cloud/container cold start.
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        alpha: float = HYBRID_ALPHA,
    ):
        self.chroma_path = str(chroma_path or CHROMA_DB_PATH)
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        self.alpha = alpha  # Weight for dense vs BM25

        logger.info(f"Initializing HybridRetriever with collection '{self.collection_name}' at {self.chroma_path}")
        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            # Verify collection access is writable/usable
            self.client.list_collections()
        except Exception as e:
            logger.warning(f"Persistent ChromaDB at {self.chroma_path} unavailable ({e}); falling back to /tmp/mgi_chroma_db or EphemeralClient.")
            try:
                tmp_path = "/tmp/mgi_chroma_db"
                self.client = chromadb.PersistentClient(
                    path=tmp_path,
                    settings=Settings(anonymized_telemetry=False)
                )
            except Exception:
                self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

        logger.info(f"Loading embedding model '{self.embedding_model_name}'...")
        self.embedder = SentenceTransformer(self.embedding_model_name)

        # Auto-ingest if collection is missing or empty; fall back if primary path is read-only
        try:
            self._ensure_collection_and_data()
        except Exception as e:
            logger.warning(f"Primary ChromaDB collection setup failed ({e}); falling back to /tmp/mgi_chroma_db or EphemeralClient and re-ingesting.")
            try:
                self.client = chromadb.PersistentClient(
                    path="/tmp/mgi_chroma_db",
                    settings=Settings(anonymized_telemetry=False)
                )
                self._ensure_collection_and_data()
            except Exception as e2:
                logger.warning(f"/tmp/mgi_chroma_db fallback failed ({e2}); using EphemeralClient.")
                self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
                self._ensure_collection_and_data()

        # Build in-memory BM25 index
        self._init_bm25_index()

    def _ensure_collection_and_data(self):
        """Ensure Chroma collection exists and is populated with policy chunks."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

        count = self.collection.count()
        if count == 0:
            logger.info(f"ChromaDB collection '{self.collection_name}' is empty. Running auto-ingestion...")
            # 1. Prefer fast auto-ingestion from committed JSON chunks
            if CHUNKS_DATA_PATH.exists():
                logger.info(f"Auto-ingesting pre-processed chunks from {CHUNKS_DATA_PATH}...")
                with open(CHUNKS_DATA_PATH, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                ids = [c["id"] for c in chunks]
                texts = [c.get("text", "") for c in chunks]
                metas = [{
                    "chapter": c.get("chapter", ""),
                    "policy": c.get("policy", ""),
                    "section": c.get("section", ""),
                    "page": c.get("page", 1),
                    "doc_type": c.get("doc_type", "general"),
                } for c in chunks]

                batch_size = 64
                for i in range(0, len(chunks), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_texts = texts[i:i+batch_size]
                    batch_metas = metas[i:i+batch_size]
                    batch_embs = self.embedder.encode(
                        batch_texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist()
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metas,
                        embeddings=batch_embs,
                    )
                logger.info(f"Auto-ingestion complete. Indexed {self.collection.count()} chunks into ChromaDB.")
            elif PDF_PATH.exists():
                logger.info(f"Auto-ingesting PDF directly into active ChromaDB collection from {PDF_PATH}...")
                from backend.ingest import extract_and_chunk_pdf
                chunks = extract_and_chunk_pdf(PDF_PATH)
                ids = [c["id"] for c in chunks]
                texts = [c.get("text", "") for c in chunks]
                metas = [{
                    "chapter": c.get("chapter", ""),
                    "policy": c.get("policy", ""),
                    "section": c.get("section", ""),
                    "page": c.get("page", 1),
                    "doc_type": c.get("doc_type", "general"),
                } for c in chunks]

                batch_size = 64
                for i in range(0, len(chunks), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_texts = texts[i:i+batch_size]
                    batch_metas = metas[i:i+batch_size]
                    batch_embs = self.embedder.encode(
                        batch_texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).tolist()
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metas,
                        embeddings=batch_embs,
                    )
                logger.info(f"PDF auto-ingestion complete. Indexed {self.collection.count()} chunks into ChromaDB.")

    def _init_bm25_index(self):
        """Fetch all documents and initialize BM25Okapi index."""
        logger.info("Building BM25 lexical index from ChromaDB documents...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = all_data.get("ids") or []
        self.corpus_docs = all_data.get("documents") or []
        self.corpus_metas = all_data.get("metadatas") or []

        if not self.corpus_docs:
            logger.warning("BM25 index initialized with 0 documents.")
            self.tokenized_corpus = []
            self.bm25 = None
            return

        # Tokenize corpus for BM25
        self.tokenized_corpus = [
            self._tokenize(doc) for doc in self.corpus_docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index initialized with {len(self.corpus_docs)} documents.")

    def count(self) -> int:
        """Return total document count in collection."""
        return len(self.corpus_docs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple alphanumeric tokenizer preserving codes like HR-03-02, HR-F07, Band D, etc."""
        text = text.lower()
        # Keep hyphenated words and alphanumeric sequences intact
        tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
        return tokens

    def _detect_exact_codes(self, query: str) -> List[str]:
        """Extract policy codes, form codes, bands, and dollar amounts from query."""
        patterns = [
            r'\bhr-\d{2}-\d{2}\b',    # e.g., HR-03-02
            r'\bhr-f\d{2}\b',         # e.g., HR-F07
            r'\bband\s+[a-h]\b',       # e.g., Band D
            r'\busd\s*[\d,]+\b',      # e.g., USD 50
            r'\bq\d{1,2}\b',          # e.g., Q14
            r'\bg\d{1,2}\b',          # e.g., G2
            r'\bannexure\s+[a-j]\b',  # e.g., Annexure A
        ]
        codes = []
        for pat in patterns:
            matches = re.findall(pat, query, flags=re.IGNORECASE)
            for m in matches:
                codes.append(m.lower().replace(" ", ""))
        return codes

    def retrieve(
        self,
        query: str,
        top_k: int = DENSE_K,
        dense_k: int = 12,
        bm25_k: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search:
        1. Dense semantic search via ChromaDB
        2. Sparse lexical search via BM25
        3. Exact alphanumeric code boost (e.g. HR-03-02, HR-F07, Band D, USD 50)
        4. Reciprocal / Weighted fusion and top-k ranking
        """
        if not query.strip() or not self.corpus_docs or self.bm25 is None:
            return []

        exact_codes = self._detect_exact_codes(query)

        # 1. Dense Vector Search
        query_emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        dense_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(dense_k, len(self.corpus_docs)),
            include=["documents", "metadatas", "distances"]
        )

        dense_scores: Dict[str, float] = {}
        if dense_res and dense_res["ids"] and len(dense_res["ids"][0]) > 0:
            doc_ids = dense_res["ids"][0]
            distances = dense_res["distances"][0]
            for doc_id, dist in zip(doc_ids, distances):
                # Convert cosine distance to similarity in [0, 1]
                sim = max(0.0, 1.0 - (dist / 2.0))
                dense_scores[doc_id] = sim

        # 2. BM25 Lexical Search
        tokenized_q = self._tokenize(query)
        bm25_scores_raw = self.bm25.get_scores(tokenized_q)
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 and max(bm25_scores_raw) > 0 else 1.0

        bm25_top_indices = np.argsort(bm25_scores_raw)[::-1][:bm25_k]
        bm25_scores: Dict[str, float] = {}
        for idx in bm25_top_indices:
            doc_id = self.corpus_ids[idx]
            norm_score = bm25_scores_raw[idx] / max_bm25 if max_bm25 > 0 else 0.0
            bm25_scores[doc_id] = norm_score

        # 3. Combine Candidates and Apply Code Boost
        all_candidate_ids = set(dense_scores.keys()).union(set(bm25_scores.keys()))

        id_to_index = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        scored_candidates = []

        for doc_id in all_candidate_ids:
            d_score = dense_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)

            hybrid_score = (self.alpha * d_score) + ((1.0 - self.alpha) * b_score)

            # Code boost for exact alphanumeric matches
            idx = id_to_index[doc_id]
            doc_text = self.corpus_docs[idx].lower()
            doc_meta = self.corpus_metas[idx]

            boost = 0.0
            for code in exact_codes:
                clean_text = doc_text.replace(" ", "")
                if code in clean_text:
                    boost += 0.35  # Substantial boost for direct code mention
                # Also check metadata policy or section
                meta_str = (doc_meta.get("policy", "") + " " + doc_meta.get("section", "")).lower().replace(" ", "")
                if code in meta_str:
                    boost += 0.40

            total_score = hybrid_score + boost
            scored_candidates.append({
                "id": doc_id,
                "content": self.corpus_docs[idx],
                "metadata": doc_meta,
                "score": total_score,
                "dense_score": d_score,
                "bm25_score": b_score,
                "boost": boost,
            })

        # Sort descending by total score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_candidates[:top_k]

        return top_results

# Global singleton retriever instance
_retriever_instance: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    """Get or create singleton HybridRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
