from transformers.cache_utils import DynamicCache
from src.evaluation.cag_runner import CAGRunner
from src.evaluation.rag_runner import RAGRunner


class BenchmarkRunner:
    def __init__(self, model, tokenizer, doc_text: str):
        self.cag = CAGRunner(model, tokenizer)
        self.rag = RAGRunner(model, tokenizer, doc_text)

    def run_cag(self, question: str, cache: DynamicCache, origin_len: int) -> dict:
        return self.cag.run(question, cache, origin_len)

    def run_rag(self, question: str) -> dict:
        return self.rag.run(question)
