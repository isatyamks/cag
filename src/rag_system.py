from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda"},
    trust_remote_code=True,
    token=HF_TOKEN
    
    )


class SimpleRAG:
    def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedder = SentenceTransformer(model_name)
        print("Encoding documents for RAG")
        self.doc_embeddings = self.embedder.encode(documents)
        print("RAG setup complete")

    def retrieve_and_generate(self, query, top_k=2):
        # Step 1: Retrieve relevant documents
        query_embedding = self.embedder.encode([query])
        query_setup = tokenizer(query, return_tensors="pt").input_ids

        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        print(top_indices)
        retrieved_docs = [self.documents[i] for i in top_indices]

        # Step 2: Generate response (simulated - in reality this would call LLM)
        # Simulate generation with the retrieved context + query
        context = " ".join(retrieved_docs)
        generation_prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        # Use our existing model for generation
        input_ids = tokenizer(generation_prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response, retrieved_docs