import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Optimized chunk size
CHUNK_SIZE = 2048
VECTOR_DIR = "vectorstore/course_index"

def get_index_path(section: str, course_id: str) -> Tuple[str, str]:
    """Get paths for index and metadata files."""
    base_dir = Path(VECTOR_DIR) / course_id / section
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "index.faiss"), str(base_dir / "metadata.pkl")

@lru_cache(maxsize=10)
def load_faiss_index(section: str, course_id: str) -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index and metadata with caching."""
    index_path, meta_path = get_index_path(section, course_id)
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing index files for {section} in course {course_id}")
    
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise

def save_faiss_index(embeddings: List[List[float]], metadata: List[str], section: str, course_id: str):
    """Save FAISS index and metadata."""
    if not embeddings:
        raise ValueError("No embeddings provided.")
        
    try:
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype("float32")
        
        # Create and configure index
        dim = embeddings_np.shape[1]
        nlist = min(4, len(embeddings))  # number of clusters
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        
        # Train and add vectors
        if not index.is_trained:
            index.train(embeddings_np)
        index.add(embeddings_np)
        
        # Save index and metadata
        index_path, meta_path = get_index_path(section, course_id)
        faiss.write_index(index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved index for {section} with {len(embeddings)} vectors")
        
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise

@lru_cache(maxsize=100)
def retrieve_chunks(question: str, section: str, course_id: str, top_k: int = 3) -> List[str]:
    """Retrieve relevant chunks using FAISS with improved caching and efficiency."""
    from utils.gpt import get_embedding
    
    try:
        # Load index and metadata
        index, metadata = load_faiss_index(section, course_id)
        
        # Get query embedding
        query_vec = get_embedding(question)
        query_vec_np = np.array([query_vec]).astype("float32")
        
        # Configure search parameters
        if isinstance(index, faiss.IndexIVFFlat):
            index.nprobe = 2  # Number of clusters to visit during search
        
        # Perform search
        distances, indices = index.search(query_vec_np, min(top_k, len(metadata)))
        
        # Filter results
        chunks = []
        seen_content = set()  # Avoid duplicate content
        for i, dist in zip(indices[0], distances[0]):
            if i < 0 or i >= len(metadata):
                continue
                
            chunk = metadata[i]
            if isinstance(chunk, dict):
                chunk = chunk.get("text", "")
                
            # Skip if too similar to already included chunks
            chunk_normalized = ' '.join(chunk.split())
            if chunk_normalized in seen_content:
                continue
                
            chunks.append(chunk)
            seen_content.add(chunk_normalized)
        
        logger.info(f"Retrieved {len(chunks)} unique chunks from {section}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def clean_section_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

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
