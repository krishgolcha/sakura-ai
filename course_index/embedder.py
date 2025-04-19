from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts)

def build_index(texts, metadata):
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, metadata
    if not all_chunks:
     print(f"[{section}] No content to index.")
    return
