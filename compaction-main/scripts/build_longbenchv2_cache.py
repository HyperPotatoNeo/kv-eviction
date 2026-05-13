"""
One-time script to download LongBench v2, tokenize all examples with Qwen3-4B,
and save a local cache with num_tokens included.

Output: data/longbenchv2_cache.jsonl
"""
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE_PATH = Path(__file__).parent.parent / "data" / "longbenchv2_cache.jsonl"
TOK_NAME = "Qwen/Qwen3-4B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
tokenizer.model_max_length = int(1e9)

print("Loading LongBench v2 from HuggingFace...")
dataset = load_dataset("THUDM/LongBench-v2", split="train")
print(f"  {len(dataset)} examples")

print("Tokenizing and saving...")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CACHE_PATH, "w") as f:
    for i, entry in enumerate(dataset):
        n_tokens = len(tokenizer.encode(entry["context"], add_special_tokens=False))
        row = dict(entry)
        row["num_tokens"] = n_tokens
        f.write(json.dumps(row) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dataset)}")

print(f"Saved to {CACHE_PATH}")
