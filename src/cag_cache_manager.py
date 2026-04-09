import os
import torch
from src.kvcache import get_kv_cache
from src.clean import clean_up

def build_and_save_cache(model, tokenizer, prompt: str, cache_dir: str, cache_filename: str):
    print("Building CAG KV cache...")
    
    cache = get_kv_cache(model, tokenizer, prompt)
    origin_len = cache.get_seq_length()
    
    print("KV cache built.")
    
    clean_up(cache, origin_len)
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)
    torch.save(cache, cache_path)
    print(f"Cache saved to disk at {cache_path}")
    
    return cache, origin_len
