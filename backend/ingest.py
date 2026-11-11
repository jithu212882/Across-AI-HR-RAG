import os
import re
import sys
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import pymupdf4llm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from backend.config import (
    PDF_PATH,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

def is_toc_page(text: str) -> bool:
    """Detect if a page is part of the Table of Contents with dot leaders."""
    if re.search(r'(\.\s*){5,}', text) or "Table of Contents" in text:
        return True
    return False

def clean_page_markdown(md_text: str) -> str:
    """Remove header/footer artifacts and page numbering."""
    # Strip confidentiality notices and page numbers
    cleaned = re.sub(r'Confidential\s*-\s*Internal\s*use\s*only\s*\n+Page\s*\d+\s*of\s*\d+\s*', '', md_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\*\*MGI\s*HR\s*Policy\s*Handbook\*\*\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    # Remove standalone bracketed metadata tags if any exist
    cleaned = re.sub(r'^\[.*? > .*?\]\s*\n', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()

def parse_handbook_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Layout-aware extraction from MGI HR Policy Handbook:
    - Filters out TOC pages (pages 3-6)
    - Normalizes structural headings:
        # Chapter X: <Title>
        ## Policy <ID>: <Title>
        ### <Subsection>
    - Preserves Annexures A-E as intact table chunks
    - Indexes Annexure F forms (HR-F01 to HR-F26) as form-level chunks
    - Indexes Annexure H FAQs (Q1 to Q30) as Q&A pair chunks
    - Hierarchically splits oversized sections (chunk_size=750, overlap=100)
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found at {pdf_path}")

    logger.info(f"Opening PDF document: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"Total pages: {total_pages}")

    # Track structural hierarchy across pages
    current_chapter_title = "Employment Framework"
    current_chapter_id = "1"
    current_policy_id = "HR-01-01"
    current_policy_title = "Equal Opportunity Employment"

    raw_chunks: List[Dict[str, Any]] = []

    # Map of known annexures
    annexure_titles = {
        109: ("Annexure A", "Band and Grade Structure", "table"),
        110: ("Annexure B", "Leave Entitlement Matrix", "table"),
        111: ("Annexure C", "HR Delegation of Authority", "table"),
        112: ("Annexure D", "HR Service Standards", "table"),
        113: ("Annexure E", "Contacts and Escalation Guide", "table"),
    }

    # Header splitter for chapter policies
    headers_to_split = [
        ("#", "chapter"),
        ("##", "policy"),
        ("###", "section"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_raw_text = doc[page_idx].get_text()

        # 1. Filter out TOC pages (pages 3-6) containing dot leaders
        if 3 <= page_num <= 6 or is_toc_page(page_raw_text):
            logger.debug(f"Skipping TOC page {page_num}")
            continue

        # Extract markdown representation of the page
        page_md = pymupdf4llm.to_markdown(doc, pages=[page_idx])
        cleaned_md = clean_page_markdown(page_md)

        if not cleaned_md.strip():
            continue

        # 2. Annexures A - E: Standalone intact Markdown table chunks
        if page_num in annexure_titles:
            ann_code, ann_name, doc_type = annexure_titles[page_num]
            logger.info(f"Processing intact table chunk for {ann_code}: {ann_name} (Page {page_num})")
            raw_chunks.append({
                "content": f"# {ann_code}: {ann_name}\n\n{cleaned_md}",
                "metadata": {
                    "chapter": "Annexures",
                    "policy": f"{ann_code}: {ann_name}",
                    "section": "Table Matrix",
                    "page": page_num,
                    "doc_type": doc_type,
                }
            })
            continue

        # 3. Annexure F: HR Forms HR-F01 through HR-F26 (pages 114 to 139)
        if 114 <= page_num <= 139:
            form_match = re.search(r'(HR-F\d{2})\s+([^\n\r]+)', cleaned_md)
            if form_match:
                form_id = form_match.group(1).strip()
                form_title = form_match.group(2).replace('*', '').strip()
            else:
                form_id = f"HR-F{page_num - 113:02d}"
                form_title = "HR Application Form"

            logger.info(f"Processing Form-level chunk {form_id}: {form_title} (Page {page_num})")
            raw_chunks.append({
                "content": f"# Annexure F: HR Forms\n## Form {form_id}: {form_title}\n\n{cleaned_md}",
                "metadata": {
                    "chapter": "Annexure F: Forms",
                    "policy": f"{form_id}: {form_title}",
                    "section": "Form Specification",
                    "page": page_num,
                    "doc_type": "form",
                }
            })
            continue

        # 4. Annexure G: Manager Checklists (pages 140 to 142)
        if 140 <= page_num <= 142:
            checklists = re.split(r'\n(?=##\s*\*\*?G\d+\.)', cleaned_md)
            for idx, item in enumerate(checklists):
                item_text = item.strip()
                if not item_text:
                    continue
                g_match = re.search(r'##\s*\*\*?(G\d+)\.\s*([^\n\r*]+)', item_text)
                if g_match:
                    g_id = g_match.group(1).strip()
                    g_title = g_match.group(2).strip()
                    chunk_title = f"{g_id}: {g_title}"
                else:
                    g_id = f"G-Checklist"
                    chunk_title = "Manager Checklist"

                raw_chunks.append({
                    "content": f"# Annexure G: Manager Checklists\n\n{item_text}",
                    "metadata": {
                        "chapter": "Annexure G: Checklists",
                        "policy": chunk_title,
                        "section": "Checklist",
                        "page": page_num,
                        "doc_type": "checklist",
                    }
                })
            continue

        # 5. Annexure H: FAQs Q1 through Q30 (pages 143 to 145)
        if 143 <= page_num <= 145:
            faq_items = re.split(r'\n(?=##\s*\*\*?Q\d+\.)', cleaned_md)
            for item in faq_items:
                item_text = item.strip()
                if not item_text or item_text.startswith("# Annexure H"):
                    continue
                q_match = re.search(r'##\s*\*\*?(Q\d+)\.\s*([^\n\r*]+)', item_text)
                if q_match:
                    q_id = q_match.group(1).strip()
                    q_question = q_match.group(2).strip()
                    chunk_title = f"{q_id}: {q_question}"
                else:
                    q_id = "FAQ"
                    chunk_title = "Frequently Asked Question"

                raw_chunks.append({
                    "content": f"# Annexure H: Frequently Asked Questions\n\n{item_text}",
                    "metadata": {
                        "chapter": "Annexure H: FAQs",
                        "policy": f"FAQ {q_id}",
                        "section": chunk_title,
                        "page": page_num,
                        "doc_type": "faq",
                    }
                })
            continue

        # 6. Annexure I: Glossary of Terms & Annexure J: Revision History
        if page_num in (146, 147):
            raw_chunks.append({
                "content": f"# Annexure I: Glossary of Terms\n\n{cleaned_md}",
                "metadata": {
                    "chapter": "Annexure I: Glossary",
                    "policy": "Glossary of Terms",
                    "section": "Definitions",
                    "page": page_num,
                    "doc_type": "glossary",
                }
            })
            continue
        if page_num == 148:
            raw_chunks.append({
                "content": f"# Annexure J: Revision History\n\n{cleaned_md}",
                "metadata": {
                    "chapter": "Annexure J: Revision History",
                    "policy": "Document Revision History",
                    "section": "Change Log",
                    "page": page_num,
                    "doc_type": "revision",
                }
            })
            continue

        # 7. Intro / Document Control / Forewords (Pages 1-2, 7-10)
        if page_num < 11:
            title_match = re.search(r'^(#+\s*)?([A-Za-z0-9 ,:\'-]+)', cleaned_md)
            title = title_match.group(2).strip() if title_match else f"Introduction Page {page_num}"
            raw_chunks.append({
                "content": cleaned_md,
                "metadata": {
                    "chapter": "Introduction & Governance",
                    "policy": title[:50],
                    "section": "Overview",
                    "page": page_num,
                    "doc_type": "general",
                }
            })
            continue

        # 8. Main Policy Chapters 1 to 12 (Pages 11 to 107)
        # Update chapter state
        chap_match = re.search(r'#\s*\*\*(\d+)\.\s*([^*]+)\*\*', cleaned_md)
        if chap_match:
            current_chapter_id = chap_match.group(1).strip()
            current_chapter_title = chap_match.group(2).strip()

        # Update policy state
        pol_match = re.search(r'##\s*\*\*(\d+\.\d+)\s+([^*]+)\*\*', cleaned_md)
        if pol_match:
            sub_id = pol_match.group(1).strip()
            pol_name = pol_match.group(2).strip()
            pid_match = re.search(r'\|\*\*Policy ID\*\*\|\s*([A-Z0-9-]+)\s*\|', cleaned_md)
            if pid_match:
                current_policy_id = pid_match.group(1).strip()
            else:
                current_policy_id = f"HR-{int(current_chapter_id):02d}-{int(sub_id.split('.')[1]):02d}"
            current_policy_title = pol_name

        # Normalize markdown headers for consistent hierarchical splitting
        # Ensure chapter header has clean # format
        normalized_md = re.sub(
            r'###\s*\*\*CHAPTER\s*(\d+)\*\*\s*\n+#\s*\*\*(\d+)\.\s*([^*]+)\*\*',
            r'# Chapter \1: \3',
            cleaned_md
        )
        # Normalize policy headings
        normalized_md = re.sub(
            r'##\s*\*\*(\d+\.\d+)\s+([^*]+)\*\*',
            f'## Policy {current_policy_id}: \\2 (\\1)',
            normalized_md
        )
        # Normalize subsection headings like ### **1. Purpose** -> ### 1. Purpose
        normalized_md = re.sub(
            r'###\s*\*\*([^*]+)\*\*',
            r'### \1',
            normalized_md
        )

        # Split text by Markdown headers
        try:
            md_docs = markdown_splitter.split_text(normalized_md)
        except Exception as e:
            logger.warning(f"Markdown splitter failed on page {page_num}: {e}. Falling back to raw text.")
            md_docs = []

        if not md_docs:
            raw_chunks.append({
                "content": normalized_md,
                "metadata": {
                    "chapter": f"Chapter {current_chapter_id}: {current_chapter_title}",
                    "policy": f"{current_policy_id}: {current_policy_title}",
                    "section": "General",
                    "page": page_num,
                    "doc_type": "policy",
                }
            })
            continue

        for doc_item in md_docs:
            text = doc_item.page_content.strip()
            if not text or len(text) < 25:
                continue

            sec_meta = doc_item.metadata.get("section", "Policy Details").strip("* ")
            chap_meta = doc_item.metadata.get("chapter", f"Chapter {current_chapter_id}: {current_chapter_title}").strip("* ")
            pol_meta = doc_item.metadata.get("policy", f"{current_policy_id}: {current_policy_title}").strip("* ")

            # Split oversized sections with RecursiveCharacterTextSplitter (chunk_size=750, overlap=100)
            if len(text) > 750:
                sub_splits = text_splitter.split_text(text)
                for s in sub_splits:
                    if len(s.strip()) > 20:
                        raw_chunks.append({
                            "content": s.strip(),
                            "metadata": {
                                "chapter": chap_meta,
                                "policy": pol_meta,
                                "section": sec_meta,
                                "page": page_num,
                                "doc_type": "policy",
                            }
                        })
            else:
                raw_chunks.append({
                    "content": text,
                    "metadata": {
                        "chapter": chap_meta,
                        "policy": pol_meta,
                        "section": sec_meta,
                        "page": page_num,
                        "doc_type": "policy",
                    }
                })

    logger.info(f"Successfully extracted {len(raw_chunks)} clean structural chunks.")
    return raw_chunks

def index_chunks(chunks: List[Dict[str, Any]], recreate: bool = True) -> int:
    """
    Index extracted chunks into ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings.
    """
    logger.info(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )

    if recreate:
        try:
            logger.info(f"Removing old collection '{COLLECTION_NAME}' if it exists...")
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "MGI HR Policy Handbook Vector Store"}
    )

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    batch_size = 64
    total = len(chunks)
    logger.info(f"Generating embeddings and indexing {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        docs = [item["content"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        # Generate normalized embeddings
        embeddings = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"Indexed chunks {min(i + batch_size, total)} / {total}")

    final_count = collection.count()
    logger.info(f"Indexing complete! Collection '{COLLECTION_NAME}' has {final_count} documents.")
    return final_count

def run_ingest(force_rebuild: bool = True):
    """Main pipeline execution."""
    logger.info("Starting MGI HR Policy Handbook ingestion pipeline...")
    chunks = parse_handbook_pages(PDF_PATH)
    count = index_chunks(chunks, recreate=force_rebuild)
    logger.info(f"Ingestion pipeline finished successfully with {count} chunks indexed.")
    return count

if __name__ == "__main__":
    force = "--no-rebuild" not in sys.argv
    run_ingest(force_rebuild=force)
