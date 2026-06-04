import time
from transformers.cache_utils import DynamicCache
from src.llm.generation import generate
from src.cache.cleaner import clean_up
from src.core.config import CLOSE_AND_OPEN
from src.llm.tokens import count_raw_question_tokens, compute_cache_size_mb


class CAGRunner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, question: str, cache: DynamicCache, origin_len: int) -> dict:
        trace = []

        t0 = time.time()
        clean_up(cache, origin_len)
        t_clean = (time.time() - t0) * 1000

        t0 = time.time()
        input_ids = self.tokenizer(
            question + CLOSE_AND_OPEN,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        new_tokens = input_ids.shape[-1]
        raw_q_tokens = count_raw_question_tokens(self.tokenizer, question)
        fmt_tokens = new_tokens - raw_q_tokens
        t_token = (time.time() - t0) * 1000

        trace.append({"step": "restore_kv_cache",      "time_ms": round(t_clean), "tokens": origin_len})
        trace.append({"step": "tokenize_query+format", "time_ms": round(t_token), "tokens": new_tokens})

        gen_ids, prefill_ms, decode_ms = generate(self.model, input_ids, cache)
        output_len = gen_ids.shape[-1]

        trace.append({"step": "forward_pass_new_only", "time_ms": round(prefill_ms), "tokens": new_tokens})
        trace.append({"step": "decode_output",         "time_ms": round(decode_ms),  "tokens": output_len})

        answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "trace": trace,
            "metrics": {
                "raw_question_tokens": raw_q_tokens,
                "fmt_overhead_tokens": fmt_tokens,
                "new_tokens_encoded":  new_tokens,
                "cached_tokens":       origin_len,
                "effective_context":   origin_len + new_tokens,
                "generated_tokens":    output_len,
                "restore_ms":          round(t_clean),
                "tokenization_ms":     round(t_token),
                "forward_pass_ms":     round(prefill_ms),
                "decode_ms":           round(decode_ms),
                "total_ms":            round(sum(t["time_ms"] for t in trace)),
                "cache_size_mb":       round(compute_cache_size_mb(self.model, origin_len), 2),
            },
        }
