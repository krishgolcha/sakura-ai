from typing import Callable, Any, Dict
from cachetools import TTLCache

# Initialize cache with 1-hour TTL
_cache = TTLCache(maxsize=100, ttl=3600)

def get_cached_content(key: str, fetch_func: Callable[[], Any], ttl: int = 3600) -> Any:
    """
    Get content from cache or fetch it using the provided function.
    
    Args:
        key (str): Cache key
        fetch_func (Callable): Function to fetch content if not in cache
        ttl (int): Time-to-live in seconds (default 1 hour)
        
    Returns:
        Any: Cached or freshly fetched content
    """
    cache_key = f"{key}:{ttl}"
    
    if cache_key in _cache:
        return _cache[cache_key]
    
    content = fetch_func()
    _cache[cache_key] = content
    return content 