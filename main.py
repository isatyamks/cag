import re
from src.config import MODEL_NAME, HF_TOKEN, CACHE_DIR
from src.data import DOC_TEXT
from src.model_utils import load_model_and_tokenizer
from src.prompts import build_cag_prompt
from src.cag_cache_manager import build_and_save_cache
from src.benchmark import BenchmarkRunner

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
WHITE  = "\033[97m"
PURPLE = "\033[95m"
W      = 72


def _box_row(key, value, val_color=WHITE):
    line  = f"  {BOLD}{key:<34}{RESET}{val_color}{value}{RESET}"
    plain = re.sub(r'\033\[[0-9;]*m', '', line)
    pad   = W - len(plain) + 2
    print(f"{PURPLE}│{RESET}{line}{' ' * max(pad, 0)}{PURPLE}│{RESET}")


def _box_note(text):
    line  = f"    {DIM}{text}{RESET}"
    plain = re.sub(r'\033\[[0-9;]*m', '', line)
    pad   = W - len(plain) + 2
    print(f"{PURPLE}│{RESET}{line}{' ' * max(pad, 0)}{PURPLE}│{RESET}")


def _box_top(title):
    left   = f" {title} "
    dashes = "─" * ((W - len(left)) // 2)
    extra  = "─" if (W - len(left)) % 2 else ""
    print(f"{BOLD}{PURPLE}┌{dashes}{left}{dashes}{extra}┐{RESET}")


def _box_div():
    print(f"{PURPLE}├{'─' * W}┤{RESET}")


def _box_bot():
    print(f"{PURPLE}└{'─' * W}┘{RESET}")


def print_comparison(query_count, cag_result, rag_result,
                     total_cag_encoded, total_rag_encoded):
    cm = cag_result["metrics"]
    rm = rag_result["metrics"]

    cag_new_encoded = cm["new_tokens_encoded"]
    cag_raw_q       = cm["raw_question_tokens"]
    cag_fmt         = cm["fmt_overhead_tokens"]
    cag_cached      = cm["cached_tokens"]
    cag_effective   = cm["effective_context"]

    rag_full_encoded = rm["full_tokens_encoded"]
    rag_raw_q        = rm["raw_question_tokens"]
    rag_fmt          = rm["fmt_overhead_tokens"]
    rag_ctx          = rm["context_tokens"]

    gpu_saved = rag_full_encoded - cag_new_encoded
    pct_saved = (gpu_saved / rag_full_encoded * 100) if rag_full_encoded > 0 else 0

    cum_saved = total_rag_encoded - total_cag_encoded
    cum_pct   = (cum_saved / total_rag_encoded * 100) if total_rag_encoded > 0 else 0

    print()
    _box_top(f"Query #{query_count}  ·  Fair Token Comparison")

    _box_row("Question tokens  [CAG]", f"{cag_raw_q} tokens", GREEN)
    _box_row("Question tokens  [RAG]", f"{rag_raw_q} tokens", GREEN)
    _box_note("↑ counted identically: raw text, no BOS/EOS padding")

    _box_div()

    _box_row("CAG — chat-format overhead",
             f"{cag_fmt} tokens  (<|im_end|>…<|im_start|>assistant)", DIM + WHITE)
    _box_row("CAG — total GPU encoded (new)",
             f"{cag_new_encoded} tokens  (question + format)", GREEN)
    _box_row("CAG — KV-cache reused",
             f"{cag_cached} tokens  [zero re-encoding cost]", DIM + WHITE)
    _box_row("CAG — logical context",
             f"{cag_effective} tokens  (= {cag_new_encoded} + {cag_cached})", WHITE)

    _box_div()

    _box_row("RAG — chat-format overhead", f"{rag_fmt} tokens", DIM + WHITE)
    _box_row("RAG — context tokens",       f"{rag_ctx} tokens", DIM + WHITE)
    _box_row("RAG — total GPU encoded (full)",
             f"{rag_full_encoded} tokens  (re-encoded every query)", RED)

    _box_div()

    _box_row("GPU encoding saved (this query)",
             f"{gpu_saved} tokens  ({pct_saved:.1f}% less forward-pass work)", GREEN)

    _box_div()

    _box_row("[Session] RAG total GPU tokens", f"{total_rag_encoded} tokens", RED)
    _box_row("[Session] CAG total GPU tokens",
             f"{total_cag_encoded} tokens  (new only; cache built once)", GREEN)
    _box_row("[Session] GPU work reduction",
             f"{cum_pct:.1f}%  ({cum_saved} tokens never re-encoded)",
             GREEN if cum_pct > 0 else WHITE)
    _box_bot()


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

    total_cag_new_encoded  = 0
    total_rag_full_encoded = 0
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
        total_cag_new_encoded  += cag_result["metrics"]["new_tokens_encoded"]
        total_rag_full_encoded += rag_result["metrics"]["full_tokens_encoded"]

        print_comparison(
            query_count, cag_result, rag_result,
            total_cag_new_encoded, total_rag_full_encoded
        )


if __name__ == "__main__":
    main()
