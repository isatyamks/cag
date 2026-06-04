from transformers.cache_utils import DynamicCache
from src.evaluation.benchmark import BenchmarkRunner
from src.visualization.terminal import print_unified_result
from src.visualization.plotter import LivePlotter


class BenchmarkSession:
    def __init__(self, model, tokenizer, doc_text: str, cache: DynamicCache, origin_len: int):
        self.runner = BenchmarkRunner(model, tokenizer, doc_text)
        self.cache = cache
        self.origin_len = origin_len
        self.plotter = LivePlotter(cag_setup_cost=origin_len)
        self.query_count = 0
        self.total_cag_encoded = 0
        self.total_rag_encoded = 0

    def _next_question(self) -> str | None:
        try:
            q = input("\nEnter the Question:\n")
            return None if q.strip().lower() == "exit" else q
        except (EOFError, KeyboardInterrupt):
            return None

    def start(self):
        print("\nStarting Interactive Benchmark (Type 'exit' to quit)")
        print("-" * 60)

        while True:
            question = self._next_question()
            if question is None:
                break

            print("\n--- Processing ---")

            cag_result = self.runner.run_cag(question, self.cache, self.origin_len)
            rag_result = self.runner.run_rag(question)

            self.query_count += 1
            self.total_cag_encoded += cag_result["metrics"]["new_tokens_encoded"]
            self.total_rag_encoded += rag_result["metrics"]["full_tokens_encoded"]

            print_unified_result(
                self.query_count,
                cag_result,
                rag_result,
                self.total_cag_encoded,
                self.total_rag_encoded,
            )

            self.plotter.update(self.query_count, self.total_cag_encoded, self.total_rag_encoded)
