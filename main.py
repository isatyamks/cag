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

    while True:
        try:
            question = input("\nEnter the Question:\n")
            if question.strip().lower() == "exit":
                break
        except (EOFError, KeyboardInterrupt):
            break

        print("\n--- Processing ---")

        # Run CAG
        cag_result = runner.run_cag(question, cache, origin_len)
        
        # Run RAG
        rag_result = runner.run_rag(question)

        # Print
        print(f"\nQuestion: {question}")
        runner.print_results("CAG", cag_result, "Actual Input")
        runner.print_results("RAG", rag_result, "Full Context")

        # Compare
        print(f"\n[Comparison]")
        print(f"  Tokens Saved per Query: {rag_result['input_len'] - cag_result['input_len']}")
        x_ratio = rag_result['input_len'] / cag_result['input_len']

        print(f"for Rag: {'-' * int(5*x_ratio)}")
        print(f"for CAG: {'-' * 5}")

if __name__ == "__main__":
    main()
