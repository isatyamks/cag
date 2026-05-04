from src.config import MODEL_NAME, HF_TOKEN, CACHE_DIR
from src.data import DOC_TEXT
from src.model_utils import load_model_and_tokenizer
from src.prompts import build_cag_prompt
from src.cag_cache_manager import build_and_save_cache
from src.runner import BenchmarkRunner
from src.display import print_unified_result


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    print(f"Loaded {MODEL_NAME}.")

    system_prompt = build_cag_prompt(tokenizer, DOC_TEXT)
    cache, origin_len = build_and_save_cache(
        model, tokenizer, system_prompt, CACHE_DIR, cache_filename="india.cache"
    )

    runner = BenchmarkRunner(model, tokenizer, DOC_TEXT)

    print("\nStarting Interactive Benchmark (Type 'exit' to quit)")
    print("-" * 60)

    total_cag_encoded = 0
    total_rag_encoded = 0
    query_count = 0

    while True:
        try:
            question = input("\nEnter the Question:\n")
            if question.strip().lower() == "exit":
                break
        except (EOFError, KeyboardInterrupt):
            break

        print("\n--- Processing ---")

        cag_result = runner.run_cag(question, cache, origin_len)
        rag_result = runner.run_rag(question)

        query_count += 1
        total_cag_encoded += cag_result["metrics"]["new_tokens_encoded"]
        total_rag_encoded += rag_result["metrics"]["full_tokens_encoded"]

        print_unified_result(
            query_count, cag_result, rag_result, total_cag_encoded, total_rag_encoded
        )


if __name__ == "__main__":
    main()
