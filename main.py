from src.core.config import MODEL_NAME, HF_TOKEN, CACHE_DIR
from src.llm import load_model_and_tokenizer, build_cag_prompt
from src.cache import build_and_save_cache
from src.evaluation import BenchmarkSession


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)
    print(f"Loaded {MODEL_NAME}.")

    print("\n" + "=" * 60)
    topic = "India"
    print(f"Loading document context from data/{topic}.txt...")

    with open(f"data/{topic}.txt", "r", encoding="utf-8") as f:
        doc_text = f.read()

    system_prompt = build_cag_prompt(tokenizer, doc_text)
    cache, origin_len = build_and_save_cache(
        model, tokenizer, system_prompt, CACHE_DIR, cache_filename=f"{topic}.cache"
    )

    session = BenchmarkSession(model, tokenizer, doc_text, cache, origin_len)
    session.start()


if __name__ == "__main__":
    main()
