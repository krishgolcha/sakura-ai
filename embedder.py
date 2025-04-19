import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm

from canvas_api.fetch_section_data import get_section_data
from canvas_api.fetch_course_data import get_section_content
from utils.gpt import get_embedding
from utils.text_splitter import chunk_text
from retriever import save_faiss_index

VECTOR_DIR = "vectorstore/course_index"

def embed_section(course_id: str, section: str, raw_html: str):
    if not raw_html or len(raw_html.strip()) < 20:
        print(f"[SKIP] No content in section: {section}")
        return

    chunks = chunk_text(raw_html)
    if not chunks:
        print(f"[SKIP] Could not split section: {section}")
        return

    print(f"[{section}] Chunked into {len(chunks)} pieces.")
    embeddings = []
    final_chunks = []  # standardized chunk metadata

    for chunk in tqdm(chunks, desc=f"[{section}] Embedding"):
        try:
            vec = get_embedding(chunk)
            embeddings.append(vec)
            final_chunks.append({"text": chunk})
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            continue

    if embeddings:
        save_faiss_index(embeddings, final_chunks, section, course_id)
        print(f"[{section}] ✅ FAISS index saved.\n")
    else:
        print(f"[{section}] ❌ No embeddings created.")

def build_course_embeddings(course_id: str):
    section_data = get_section_data(course_id)
    if not section_data:
        print(f"[ERROR] No section data found for course {course_id}")
        return

    for section_name, meta in section_data.items():
        if meta.get("type") != "internal":
            continue

        try:
            html = get_section_content(course_id, section_name)
            embed_section(course_id, section_name, html)
        except Exception as e:
            print(f"[ERROR] Failed to embed section '{section_name}': {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embedder.py <course_id>")
    else:
        build_course_embeddings(sys.argv[1])
