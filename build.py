import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import os 

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # use the SAME model
KNOWLEDGE_PATH = "data/know.txt"
CACHE_PATH = "cache/ronan_knowledge.cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
cache = DynamicCache()

with torch.no_grad():
    _ = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True
    )

os.makedirs("cache", exist_ok=True)
torch.save(cache, CACHE_PATH)

print("✅ CAG cache built and saved:", CACHE_PATH)
