import re
from itertools import zip_longest

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
WHITE  = "\033[97m"
PURPLE = "\033[95m"

COL_W = 54
BOX_W = COL_W * 2 + 1

def _strip_ansi(text: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', text)

def _pad(text: str, width: int) -> str:
    plain_len = len(_strip_ansi(text))
    return text + " " * max(0, width - plain_len)

def _top(color, title):
    left   = f" {title} "
    dashes = "─" * ((BOX_W - len(left)) // 2)
    extra  = "─" if (BOX_W - len(left)) % 2 else ""
    print(f"\n{BOLD}{color}┌{dashes}{left}{dashes}{extra}┐{RESET}")

def _div(color):
    print(f"{color}├{'─' * COL_W}┼{'─' * COL_W}┤{RESET}")

def _bot(color):
    print(f"{color}└{'─' * COL_W}┴{'─' * COL_W}┘{RESET}")

def _full_row(color, text, text_color=WHITE, center=False):
    if center:
        text = text.center(BOX_W)
    else:
        text = f"  {text}"
    padded = _pad(f"{text_color}{text}{RESET}", BOX_W)
    print(f"{color}│{RESET}{padded}{color}│{RESET}")

def _full_div(color):
    print(f"{color}├{'─' * BOX_W}┤{RESET}")

def _full_bot(color):
    print(f"{color}└{'─' * BOX_W}┘{RESET}")

def _split_row(color, left_lines, right_lines):
    if isinstance(left_lines, str):  left_lines  = [left_lines]
    if isinstance(right_lines, str): right_lines = [right_lines]

    for l, r in zip_longest(left_lines, right_lines, fillvalue=""):
        l_padded = _pad(l, COL_W)
        r_padded = _pad(r, COL_W)
        print(f"{color}│{RESET}{l_padded}{color}│{RESET}{r_padded}{color}│{RESET}")

def _wrap_text(text: str, max_w: int, prefix: str = "", val_color: str = WHITE) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_w:
            lines.append(f"{val_color}{cur}{RESET}")
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(f"{val_color}{cur}{RESET}")

    if not lines:
        return [f"{val_color}(empty){RESET}"]

    if prefix:
        lines[0] = f"{prefix}{lines[0]}"
        indent = " " * len(_strip_ansi(prefix))
        for i in range(1, len(lines)):
            lines[i] = f"{indent}{lines[i]}"
    return lines

def _kv(key: str, value: str, key_color: str = WHITE, val_color: str = WHITE, dim_key: bool = False) -> str:
    k_style = DIM if dim_key else BOLD
    return f"  {k_style}{key_color}{key:<26}{RESET}{val_color}{value}{RESET}"

def _metric(label: str, val: str, comment: str, val_color: str = WHITE) -> str:
    base = f"  {BOLD}{label:<15}{RESET} {val_color}{val:<4}{RESET}"
    if comment:
        base += f" {DIM}{comment}{RESET}"
    return base

def print_unified_result(query_count: int, cag_result: dict, rag_result: dict,
                         total_cag_encoded: int, total_rag_encoded: int) -> None:
    cm = cag_result["metrics"]
    rm = rag_result["metrics"]

    cag_ans = cag_result["answer"]
    rag_ans = rag_result["answer"]

    color = PURPLE

    _top(color, f"Query #{query_count}  ·  CAG vs RAG")

    _split_row(color,
               f"  {BOLD}{CYAN}CAG (Cache-Augmented){RESET}",
               f"  {BOLD}{YELLOW}RAG (Retrieval-Augmented){RESET}")
    _div(color)

    l_ans = _wrap_text(cag_ans, COL_W - 14, f"  {BOLD}Answer: {RESET}", CYAN)
    r_ans = _wrap_text(rag_ans, COL_W - 14, f"  {BOLD}Answer: {RESET}", YELLOW)
    _split_row(color, l_ans, r_ans)

    _div(color)

    l_tok = [
        _metric("GPU Encoded", str(cm["new_tokens_encoded"]), "(this query)", CYAN),
        f"    {DIM}↳ Question: {cm['raw_question_tokens']}{RESET}",
        f"    {DIM}↳ Format:   {cm['fmt_overhead_tokens']}{RESET}",
        "",
        _metric("Cached", str(cm["cached_tokens"]), "[zero cost]", WHITE),
        _metric("Logical Ctx", str(cm["effective_context"]), "", WHITE),
        _metric("Output", str(cm["generated_tokens"]), "", WHITE)
    ]

    r_tok = [
        _metric("GPU Encoded", str(rm["full_tokens_encoded"]), "(every query)", YELLOW),
        f"    {DIM}↳ Question: {rm['raw_question_tokens']}{RESET}",
        f"    {DIM}↳ Context:  {rm['context_tokens']}{RESET}",
        f"    {DIM}↳ Sys Text: {rm['sys_tokens']}{RESET}",
        f"    {DIM}↳ Format:   {rm['fmt_overhead_tokens']}{RESET}",
        "",
        _metric("Output", str(rm["generated_tokens"]), "", WHITE)
    ]

    _split_row(color, l_tok, r_tok)

    _div(color)

    l_lat = [
        _kv("Restore (Trim)", f"{cm['restore_ms']} ms", CYAN, WHITE, True),
        _kv("Tokenize",       f"{cm['tokenization_ms']} ms", CYAN, WHITE, True),
        _kv("Forward Pass",   f"{cm['forward_pass_ms']} ms", CYAN, WHITE, True),
        _kv("Decode",         f"{cm['decode_ms']} ms", CYAN, WHITE, True),
        _kv("Total Latency",  f"{cm['total_ms']} ms", CYAN, CYAN)
    ]

    r_lat = [
        _kv("Retrieval (BM25)", f"{rm.get('retrieval_ms', 0)} ms", YELLOW, WHITE, True),
        _kv("Tokenize",       f"{rm['tokenization_ms']} ms", YELLOW, WHITE, True),
        _kv("Forward Pass",   f"{rm['forward_pass_ms']} ms", YELLOW, WHITE, True),
        _kv("Decode",         f"{rm['decode_ms']} ms", YELLOW, WHITE, True),
        _kv("Total Latency",  f"{rm['total_ms']} ms", YELLOW, YELLOW)
    ]

    _split_row(color, l_lat, r_lat)

    _full_div(color)

    gpu_saved = rm["full_tokens_encoded"] - cm["new_tokens_encoded"]
    pct_saved = (gpu_saved / rm["full_tokens_encoded"] * 100) if rm["full_tokens_encoded"] > 0 else 0
    cum_saved = total_rag_encoded - total_cag_encoded
    cum_pct   = (cum_saved / total_rag_encoded * 100) if total_rag_encoded > 0 else 0

    _full_row(color, "PERFORMANCE SUMMARY", PURPLE, center=True)
    _full_row(color, "")

    _full_row(color,
              f"{BOLD}Encoding Saved (This Query): {RESET}{GREEN}{gpu_saved} tokens{RESET} "
              f"{DIM}({pct_saved:.1f}% less forward-pass work){RESET}")

    _full_row(color,
              f"{BOLD}Session CAG GPU Tokens:      {RESET}{CYAN}{total_cag_encoded} tokens{RESET} "
              f"{DIM}(new only; cache built once){RESET}")
    _full_row(color,
              f"{BOLD}Session RAG GPU Tokens:      {RESET}{YELLOW}{total_rag_encoded} tokens{RESET} "
              f"{DIM}(re-encoded entirely){RESET}")
    _full_row(color,
              f"{BOLD}Session Total Reduction:     {RESET}{GREEN}{cum_pct:.1f}%{RESET} "
              f"{DIM}({cum_saved} tokens never re-encoded){RESET}")

    _full_bot(color)
