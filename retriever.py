import os
import pickle
import faiss
import numpy as np

VECTOR_DIR = "vectorstore/course_index"

def clean_section_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def load_faiss_index(section, course_id=None):
    folder = f"{VECTOR_DIR}/{course_id}" if course_id else VECTOR_DIR
    name = clean_section_name(section)
    index_path = f"{folder}/{name}.index"
    meta_path = f"{folder}/{name}_meta.pkl"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing FAISS index or metadata for section: {section}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def retrieve_chunks(question, section, course_id, top_k=5):
    from utils.gpt import get_embedding

    index, metadata = load_faiss_index(section, course_id)
    query_vec = get_embedding(question)
    query_vec_np = np.array([query_vec]).astype("float32")
    distances, indices = index.search(query_vec_np, top_k)

    chunks = [metadata[i] for i in indices[0] if i < len(metadata)]
    print(f"[RETRIEVER] Retrieved {len(chunks)} chunks from {section}")
    return chunks

def save_faiss_index(embeddings, chunks, section, course_id=None):
    if not embeddings:
        raise ValueError("No embeddings provided.")

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    course_folder = f"{VECTOR_DIR}/{course_id}" if course_id else VECTOR_DIR
    os.makedirs(course_folder, exist_ok=True)

    name = clean_section_name(section)
    index_path = f"{course_folder}/{name}.index"
    meta_path = f"{course_folder}/{name}_meta.pkl"

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[SAVED] Index written to: {index_path}")
    print(f"[SAVED] Metadata written to: {meta_path}")
