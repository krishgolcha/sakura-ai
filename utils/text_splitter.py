import re
from typing import List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1000)
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing newlines."""
    # Combine multiple regex operations into one
    text = re.sub(r'[\s\n]+', ' ', text)
    return text.strip()

def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """Split text into chunks based on semantic boundaries."""
    if not text:
        return []
    
    # Clean and normalize text first
    text = clean_text(text)
    
    # Use more efficient string splitting
    chunks = []
    current_pos = 0
    text_len = len(text)
    
    while current_pos < text_len:
        # Find the end of the current chunk
        chunk_end = min(current_pos + chunk_size, text_len)
        
        # If we're not at the end, try to find a sentence boundary
        if chunk_end < text_len:
            # Look for sentence endings within the last 100 chars of the chunk
            look_back = min(100, chunk_size // 4)
            last_period = text.rfind('.', chunk_end - look_back, chunk_end)
            if last_period != -1:
                chunk_end = last_period + 1
        
        # Extract the chunk
        chunk = text[current_pos:chunk_end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move position forward, accounting for overlap
        current_pos = chunk_end - chunk_overlap
        if current_pos < 0:
            current_pos = 0
    
    return chunks

# Use the more efficient split_text as the default chunking method
chunk_text = split_text
