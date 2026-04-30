from src.config import MODEL_NAME, HF_TOKEN, CACHE_DIR
from src.data import DOC_TEXT
from src.model_utils import load_model_and_tokenizer
from src.prompts import build_cag_prompt
from src.cag_cache_manager import build_and_save_cache
from src.benchmark import BenchmarkRunner

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    print(f"Loaded {MODEL_NAME}.")

    system_prompt = build_cag_prompt(tokenizer, DOC_TEXT)
    
    cache, origin_len = build_and_save_cache(
        model, tokenizer, system_prompt, 
        CACHE_DIR, cache_filename="Satyam_Kumar.cache"
    )

    runner = BenchmarkRunner(model, tokenizer, DOC_TEXT)

    print("\nStarting Interactive Benchmark (Type 'exit' to quit)")
    print("-" * 60)

    total_cag_tokens_processed = origin_len
    total_rag_tokens_processed = 0
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

        print(f"\nQuestion: {question}")
        runner.print_results("CAG", cag_result, "Actual Input")
        runner.print_results("RAG", rag_result, "Full Context")

        query_count += 1
        total_cag_tokens_processed += cag_result['input_len']
        total_rag_tokens_processed += rag_result['input_len']

        print(f"\n\033[93m\033[1m[Comparison - Query #{query_count}]\033[0m")
        
        tokens_saved = rag_result['input_len'] - cag_result['input_len']
        print(f"  Tokens Saved on this Query: \033[92m{tokens_saved}\033[0m")
        
        x_ratio = rag_result['input_len'] / cag_result['input_len']
        print(f"  RAG Workload: \033[91m{'-' * int(5*x_ratio)}\033[0m (Heavy)")
        print(f"  CAG Workload: \033[92m{'-' * 5}\033[0m (Light)")
        
        print("\n  \033[1m---  CUMULATIVE SERVER COST (Session Total) ---\033[0m")
        print(f"  Total Tokens Processed by RAG: \033[91m{total_rag_tokens_processed}\033[0m")
        print(f"  Total Tokens Processed by CAG: \033[92m{total_cag_tokens_processed}\033[0m")
        
        savings_percent = ((total_rag_tokens_processed - total_cag_tokens_processed) / total_rag_tokens_processed) * 100 if query_count > 1 else 0
        if savings_percent > 0:
            print(f"  => Total Compute Reduction: \033[1m\033[92m{savings_percent:.1f}%\033[0m")

if __name__ == "__main__":
    main()
