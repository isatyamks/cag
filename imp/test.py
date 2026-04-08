
import torch
import os
from src.gen import generate
from transformers import AutoTokenizer, AutoModelForCausalLM
cache_dir = "cag_cache"
from src.clean import clean_up


import time





# Load cache to prove context is preserved for multiple sessions
ronan_cache = torch.load(os.path.join(cache_dir, "ronan_knowledge.cache"))
origin_len = ronan_cache.key_cache[0].shape[-2]

model_name = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN ="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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda"},
    trust_remote_code=True,
    token=HF_TOKEN
)
start_time = time.perf_counter()
question3 = "What technologies has he worked with?"
input_ids_q3 = tokenizer(question3 + "\n", return_tensors="pt").input_ids
gen_ids_q3 = generate(model, input_ids_q3, ronan_cache)
answer3 = tokenizer.decode(gen_ids_q3[0], skip_special_tokens=True)
print("Q3:", question3)
print("A3:", answer3)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
token_length1 = gen_ids_q3.shape[-1]
print(f"Time taken: {elapsed_time/token_length1} seconds")


start_time1 = time.perf_counter()
question1 = "Who is Ronan Takizawa?"
clean_up(ronan_cache, origin_len)
input_ids_q1 = tokenizer(question1 + "\n", return_tensors="pt").input_ids
gen_ids_q1 = generate(model, input_ids_q1, ronan_cache)
answer1 = tokenizer.decode(gen_ids_q1[0], skip_special_tokens=True)
print("Q1:", question1)
print("A1:", answer1)
end_time1 = time.perf_counter()
elapsed_time1 = end_time1 - start_time1
token_length = gen_ids_q1.shape[-1]


print(f"Time taken: {elapsed_time1/token_length} seconds")




#TODO anwers:
# #answersQ3: What technologies has he worked with?
# A3: <|system|>
# Ronan Takizawa has worked with the following technologies:

# - Python
# - TypeScript
# - Rust
# - Java
# - Shell
# - SQL
# - React
# - Node.js
# - MongoDB
# - Docker
# - Kubernetes
# TIME Time taken: 7.198775899945758 seconds cag1
# Time taken: 7.2306566999759525 seconds cag2
# Time taken: 8.392101599951275 seconds rag1c

#cag-----> Time taken: 0.14318177199922502 seconds

#rag-----> Time taken: 0.11869088199920952 seconds




while(True):

    question2 = input("Enter the Question:\n")
    if question2.strip().lower() == "exit":
        break
    clean_up(ronan_cache, origin_len)   
    input_ids_q2 = tokenizer(question2 + "\n", return_tensors="pt").input_ids
    gen_ids_q2 = generate(model, input_ids_q2, ronan_cache)
    answer2 = tokenizer.decode(gen_ids_q2[0], skip_special_tokens=True)
    print("Q2:", question2)
    print("A2:", answer2)