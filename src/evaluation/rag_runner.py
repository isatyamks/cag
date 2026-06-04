import time
from transformers.cache_utils import DynamicCache
from src.llm.generation import generate
from src.llm.prompts import build_rag_prompt, get_rag_system_messages
from src.llm.tokens import count_raw_question_tokens
from src.rag.retriever import SimpleRetriever


class RAGRunner:
    def __init__(self, model, tokenizer, doc_text: str):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = SimpleRetriever(doc_text)

        msgs = get_rag_system_messages("", "")
        sys_text = " ".join(m["content"] for m in msgs)
        self._rag_sys_tokens = count_raw_question_tokens(tokenizer, sys_text)

        empty_prompt = build_rag_prompt(tokenizer, "", "")
        empty_tokens = tokenizer(empty_prompt, return_tensors="pt").input_ids.shape[-1]
        self._rag_fmt_tokens = empty_tokens - self._rag_sys_tokens

    def run(self, question: str) -> dict:
        trace = []

        t_ret_start = time.time()
        retrieved_context = self.retriever.retrieve(question)
        t_ret_ms = (time.time() - t_ret_start) * 1000

        t0 = time.time()
        prompt = build_rag_prompt(self.tokenizer, retrieved_context, question)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        full_tokens = input_ids.shape[-1]

        raw_q_tokens = count_raw_question_tokens(self.tokenizer, question)
        ctx_tokens = count_raw_question_tokens(self.tokenizer, retrieved_context)
        sys_tokens = self._rag_sys_tokens
        fmt_tokens = full_tokens - raw_q_tokens - ctx_tokens - sys_tokens
        t_token = (time.time() - t0) * 1000

        trace.append({"step": "tokenize_full_prompt", "time_ms": round(t_token), "tokens": full_tokens})

        gen_ids, prefill_ms, decode_ms = generate(self.model, input_ids, DynamicCache())
        output_len = gen_ids.shape[-1]

        trace.append({"step": "forward_pass_full", "time_ms": round(prefill_ms), "tokens": full_tokens})
        trace.append({"step": "decode_output",     "time_ms": round(decode_ms),  "tokens": output_len})

        answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "trace": trace,
            "metrics": {
                "raw_question_tokens": raw_q_tokens,
                "context_tokens":      ctx_tokens,
                "sys_tokens":          sys_tokens,
                "fmt_overhead_tokens": fmt_tokens,
                "full_tokens_encoded": full_tokens,
                "generated_tokens":    output_len,
                "retrieval_ms":        round(t_ret_ms),
                "tokenization_ms":     round(t_token),
                "forward_pass_ms":     round(prefill_ms),
                "decode_ms":           round(decode_ms),
                "total_ms":            round(t_ret_ms + t_token + prefill_ms + decode_ms),
            },
        }
