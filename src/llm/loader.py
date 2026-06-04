import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Running on: {device.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=hf_token
    )
    return tokenizer, model
