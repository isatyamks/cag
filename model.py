"""
RAG vs CAG – Correct End-to-End Comparison
=========================================

This script:
1. Loads a causal LLM
2. Loads knowledge text
3. Builds a KV cache with a semantic boundary (CAG)
4. Runs RAG (retrieve + generate)
5. Runs CAG (cache-augmented manual decoding)
6. Prints clean, comparable answers

Requirements:
- transformers==4.38.2
- accelerate==0.27.2
- sentence-transformers==2.5.1
"""

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
KNOWLEDGE_PATH = "data/ronan_knowledge.txt"
CACHE_PATH = "cache/ronan_knowledge.cache"
MAX_NEW_TOKENS = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# LOAD MODEL
# -----------------------
print("🔧 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✅ Model loaded")


# -----------------------
# LOAD KNOWLEDGE
# -----------------------
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()

doc_chunks = [c.strip() for c in doc_text.split("\n\n") if c.strip()]


# -----------------------
# BUILD CAG CACHE (ONCE)
# -----------------------
def build_cag_cache():
    print("⚡ Building CAG cache...")

    os.makedirs("cache", exist_ok=True)

    knowledge_prompt = (
        doc_text
        + "\n\n### Instruction:\n"
        + "Answer questions using ONLY the information above.\n\n"
        + "### Question:\n"
    )

    input_ids = tokenizer(
        knowledge_prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    cache = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True
        )

    torch.save(cache, CACHE_PATH)
    print("✅ CAG cache saved")


if not os.path.exists(CACHE_PATH):
    build_cag_cache()


# -----------------------
# SIMPLE RAG
# -----------------------
class SimpleRAG:
    def __init__(self, documents):
        self.documents = documents
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("🔄 Encoding documents for RAG...")
        self.doc_embeddings = self.embedder.encode(documents)
        print("✅ RAG ready")

    def answer(self, question):
        q_emb = self.embedder.encode([question])
        sims = cosine_similarity(q_emb, self.doc_embeddings)[0]
        top_idx = np.argsort(sims)[-2:][::-1]

        context = " ".join(self.documents[i] for i in top_idx)

        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(
            out[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )


# -----------------------
# CAG MANUAL GENERATION
# -----------------------
def cag_answer(question, cache):
    prompt = question + "\n### Answer:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = input_ids
    past = cache

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past,
                use_cache=True
            )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        past = outputs.past_key_values

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# -----------------------
# MAIN EXPERIMENT
# -----------------------
def main():
    print("\n🚀 RAG vs CAG Experiment")
    print("=" * 40)

    questions = [
        "Who is Ronan Takizawa?",
        "What are his main projects?",
        "What technologies has he worked with?",
        "What is his background?",
        "Where does he study?"
    ]

    # ---- RAG ----
    rag = SimpleRAG(doc_chunks)

    print("\n🔍 RAG RESULTS")
    for q in questions:
        ans = rag.answer(q)
        print(f"\nQ: {q}\nA: {ans}")

    # ---- CAG ----
    cache = torch.load(CACHE_PATH, map_location=device)

    print("\n⚡ CAG RESULTS")
    for q in questions:
        ans = cag_answer(q, cache)
        print(f"\nQ: {q}\nA: {ans}")


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    main()
