import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
import logging

from canvas_api.fetch_section_data import get_section_data
from canvas_api.fetch_course_data import get_section_content
from utils.gpt import get_embedding
from utils.text_splitter import split_text, chunk_text
from processing.retriever import save_faiss_index

logger = logging.getLogger(__name__)
VECTOR_DIR = "vectorstore/course_index"

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks into vectors."""
    if not texts:
        return []

    embeddings = []
    for chunk in tqdm(texts, desc="Generating embeddings"):
        try:
            vec = get_embedding(chunk)
            embeddings.append(vec)
        except Exception as e:
            logger.error(f"Failed to embed chunk: {str(e)}")
            continue

    return embeddings

def embed_section(course_id: str, section: str, raw_html: str) -> bool:
    """Embed a single section's content."""
    if not raw_html or len(raw_html.strip()) < 20:
        logger.warning(f"No content in section: {section}")
        return False

    chunks = chunk_text(raw_html)
    if not chunks:
        logger.warning(f"Could not split section: {section}")
        return False

    logger.info(f"[{section}] Chunked into {len(chunks)} pieces.")
    embeddings = []
    final_chunks = []  # standardized chunk metadata

    for chunk in tqdm(chunks, desc=f"[{section}] Embedding"):
        try:
            vec = get_embedding(chunk)
            embeddings.append(vec)
            final_chunks.append({"text": chunk})
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            continue

    if embeddings:
        save_faiss_index(embeddings, final_chunks, section, course_id)
        logger.info(f"[{section}] ✅ FAISS index saved.")
        return True
    else:
        logger.error(f"[{section}] ❌ No embeddings created.")
        return False

def build_course_embeddings(course_id: str) -> dict[str, bool]:
    """Build embeddings for all sections in a course."""
    results = {}
    section_data = get_section_data(course_id)
    if not section_data:
        logger.error(f"No section data found for course {course_id}")
        return results

    for section_name, meta in section_data.items():
        if meta.get("type") != "internal":
            continue

        try:
            html = get_section_content(course_id, section_name)
            results[section_name] = embed_section(course_id, section_name, html)
        except Exception as e:
            logger.error(f"Failed to embed section '{section_name}': {e}")
            results[section_name] = False

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embedder.py <course_id>")
    else:
        build_course_embeddings(sys.argv[1])
