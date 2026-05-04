import os
import torch
from src.kvcache import get_kv_cache
from src.clean import clean_up

def build_and_save_cache(model, tokenizer, prompt: str, cache_dir: str, cache_filename: str):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading KV cache from disk: {cache_path}")
        device = next(model.parameters()).device
        cache = torch.load(cache_path, map_location=device, weights_only=False)
        origin_len = cache.get_seq_length()
        print(f"KV cache loaded ({origin_len} tokens).")
        return cache, origin_len

    print("KV cache not found. Building...")
    cache = get_kv_cache(model, tokenizer, prompt)
    origin_len = cache.get_seq_length()
    clean_up(cache, origin_len)
    torch.save(cache, cache_path)
    print(f"KV cache built and saved to {cache_path} ({origin_len} tokens).")
    return cache, origin_len
