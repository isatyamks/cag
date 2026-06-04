import os
import src.utils.env  # noqa: F401 — side-effect: loads .env into os.environ

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CACHE_DIR = "cag_cache"
CLOSE_AND_OPEN = "<|im_end|>\n<|im_start|>assistant\n"
