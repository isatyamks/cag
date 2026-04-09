from transformers.cache_utils import DynamicCache

def clean_up(cache: DynamicCache, origin_len: int):
    # Use the built-in crop method if available (added in recent versions)
    if hasattr(cache, "crop"):
        cache.crop(origin_len)
    else:
        # Fallback: Manually slice and reset seen_tokens tally
        for i in range(len(cache.key_cache)):
            # Slicing the sequence dimension (usually index -2)
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]
        
        # Crucial: Reset the internal counter so the next forward pass 
        # knows the correct position to start appending new tokens.
        cache._seen_tokens = origin_len 
