from transformers.cache_utils import DynamicCache

def clean_up(cache: DynamicCache, origin_len: int):
    if hasattr(cache, "crop"):
        cache.crop(origin_len)
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]
        cache._seen_tokens = origin_len
