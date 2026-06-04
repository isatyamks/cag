def get_cag_system_messages(context_text: str):
    return [
        {"role": "system", "content": "You are a strict assistant. You must answer the question using ONLY the provided context. If the context does not contain the answer, you must output exactly 'I cannot answer this based on the provided context.' Do not guess or use outside knowledge."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: "},
    ]

def get_rag_system_messages(context_text: str, question: str):
    return [
        {"role": "system", "content": "You are a strict assistant. You must answer the question using ONLY the provided context. If the context does not contain the answer, you must output exactly 'I cannot answer this based on the provided context.' Do not guess or use outside knowledge."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]

def build_cag_prompt(tokenizer, context_text: str) -> str:
    messages = get_cag_system_messages(context_text)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def build_rag_prompt(tokenizer, context_text: str, question: str) -> str:
    messages = get_rag_system_messages(context_text, question)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
