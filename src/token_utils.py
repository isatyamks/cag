import torch
from src.config import CLOSE_AND_OPEN


def count_raw_question_tokens(tokenizer, question: str) -> int:
    return tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]


def count_cag_format_tokens(tokenizer) -> int:
    return tokenizer(CLOSE_AND_OPEN, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]


def compute_cache_size_mb(model, seq_len: int) -> float:
    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim     = cfg.hidden_size // cfg.num_attention_heads
    dtype_bytes  = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(
        next(model.parameters()).dtype, 2
    )
    return (2 * num_layers * num_kv_heads * seq_len * head_dim * dtype_bytes) / (1024 * 1024)
