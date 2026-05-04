import time
from transformers.cache_utils import DynamicCache
from src.gen import generate
from src.clean import clean_up
from src.prompts import build_rag_prompt
from src.config import CLOSE_AND_OPEN
from src.token_utils import (
    count_raw_question_tokens,
    count_cag_format_tokens,
    compute_cache_size_mb,
)


class BenchmarkRunner:
    def __init__(self, model, tokenizer, doc_text):
        self.model     = model
        self.tokenizer = tokenizer
        self.doc_text  = doc_text
        self._cag_fmt  = count_cag_format_tokens(tokenizer)

        # Calculate RAG system text vs format overhead
        from src.prompts import get_rag_system_messages, build_rag_prompt
        msgs = get_rag_system_messages("", "")
        sys_text = " ".join(m["content"] for m in msgs)
        self._rag_sys_tokens = count_raw_question_tokens(tokenizer, sys_text)
        
        empty_prompt = build_rag_prompt(tokenizer, "", "")
        empty_tokens = tokenizer(empty_prompt, return_tensors="pt").input_ids.shape[-1]
        self._rag_fmt_tokens = empty_tokens - self._rag_sys_tokens

    def run_cag(self, question: str, cache: DynamicCache, origin_len: int) -> dict:
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
        new_tokens   = input_ids.shape[-1]
        raw_q_tokens = count_raw_question_tokens(self.tokenizer, question)
        fmt_tokens   = new_tokens - raw_q_tokens
        t_token = (time.time() - t0) * 1000

        trace.append({"step": "restore_kv_cache",      "time_ms": round(t_clean), "tokens": origin_len})
        trace.append({"step": "tokenize_query+format", "time_ms": round(t_token), "tokens": new_tokens})

        gen_ids, prefill_ms, decode_ms = generate(self.model, input_ids, cache)
        output_len = gen_ids.shape[-1]

        trace.append({"step": "forward_pass_new_only", "time_ms": round(prefill_ms), "tokens": new_tokens})
        trace.append({"step": "decode_output",         "time_ms": round(decode_ms),  "tokens": output_len})

        answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        return {
            "answer":  answer,
            "trace":   trace,
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

    def run_rag(self, question: str) -> dict:
        trace = []

        t0 = time.time()
        prompt      = build_rag_prompt(self.tokenizer, self.doc_text, question)
        input_ids   = self.tokenizer(prompt, return_tensors="pt").input_ids
        full_tokens = input_ids.shape[-1]

        raw_q_tokens  = count_raw_question_tokens(self.tokenizer, question)
        ctx_tokens    = count_raw_question_tokens(self.tokenizer, self.doc_text)
        sys_tokens    = self._rag_sys_tokens
        fmt_tokens    = full_tokens - raw_q_tokens - ctx_tokens - sys_tokens
        t_token = (time.time() - t0) * 1000

        trace.append({"step": "tokenize_full_prompt", "time_ms": round(t_token), "tokens": full_tokens})

        gen_ids, prefill_ms, decode_ms = generate(self.model, input_ids, DynamicCache())
        output_len = gen_ids.shape[-1]

        trace.append({"step": "forward_pass_full", "time_ms": round(prefill_ms), "tokens": full_tokens})
        trace.append({"step": "decode_output",     "time_ms": round(decode_ms),  "tokens": output_len})

        answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        return {
            "answer":  answer,
            "trace":   trace,
            "metrics": {
                "raw_question_tokens":  raw_q_tokens,
                "context_tokens":       ctx_tokens,
                "sys_tokens":           sys_tokens,
                "fmt_overhead_tokens":  fmt_tokens,
                "full_tokens_encoded":  full_tokens,
                "generated_tokens":     output_len,
                "tokenization_ms":      round(t_token),
                "forward_pass_ms":      round(prefill_ms),
                "decode_ms":            round(decode_ms),
                "total_ms":             round(sum(t["time_ms"] for t in trace)),
            },
        }
