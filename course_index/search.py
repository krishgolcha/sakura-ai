import faiss
from .embedder import embed_texts

def search_index(index, query, metadata, top_k=3):
    query_vec = embed_texts([query])
    distances, indices = index.search(query_vec, top_k)
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    return results
