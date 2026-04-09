from transformers.cache_utils import DynamicCache
from src.gen import generate
from src.clean import clean_up
from src.prompts import build_rag_prompt
from src.config import CLOSE_AND_OPEN

class BenchmarkRunner:
    def __init__(self, model, tokenizer, doc_text):
        self.model = model
        self.tokenizer = tokenizer
        self.doc_text = doc_text
        
    def run_cag(self, question: str, cache: DynamicCache, origin_len: int) -> dict:
        clean_up(cache, origin_len)

        cag_query_text = question + CLOSE_AND_OPEN
        input_ids_cag = self.tokenizer(cag_query_text, return_tensors="pt", add_special_tokens=False).input_ids
        cag_input_len = input_ids_cag.shape[-1]

        gen_ids_cag = generate(self.model, input_ids_cag, cache)

        answer_cag = self.tokenizer.decode(gen_ids_cag[0], skip_special_tokens=True).strip()
        cag_output_len = gen_ids_cag.shape[-1]
        
        return {
            "answer": answer_cag,
            "input_len": cag_input_len,
            "output_len": cag_output_len
        }

    def run_rag(self, question: str) -> dict:
        rag_prompt = build_rag_prompt(self.tokenizer, self.doc_text, question)
        input_ids_rag = self.tokenizer(rag_prompt, return_tensors="pt").input_ids
        rag_input_len = input_ids_rag.shape[-1]

        rag_cache = DynamicCache()

        gen_ids_rag = generate(self.model, input_ids_rag, rag_cache)

        answer_rag = self.tokenizer.decode(gen_ids_rag[0], skip_special_tokens=True).strip()
        rag_output_len = gen_ids_rag.shape[-1]
        
        return {
            "answer": answer_rag,
            "input_len": rag_input_len,
            "output_len": rag_output_len
        }
    
    def print_results(self, title: str, results: dict, prompt_label: str):
        print(f"\n[{title} Response]")
        print(f"Answer: {results['answer']}")
        print(f"Metrics:")
        print(f"  Prompt Tokens ({prompt_label}): {results['input_len']}")
        print(f"  Output Tokens:                {results['output_len']}")
