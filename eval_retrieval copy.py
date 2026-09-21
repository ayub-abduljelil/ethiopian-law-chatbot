import json
import numpy as np
from pathlib import Path
from embedding import Embedding

# Change this to your actual JSONL path
dataset_path = Path("eval_dataset.jsonl")

# Load the evaluation dataset
entries = []

for line in dataset_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        entries.append(json.loads(line))

print("Loaded:", len(entries), "questions")

# Load production embedding model
emb_model = Embedding()

# Pick which question to inspect
i = 0

question = entries[i]["question"]

embedding = np.asarray(
    emb_model._embed_one(question),
    dtype=np.float32
)

print("\nQUESTION:")
print(question)

print("\nSHAPE:", embedding.shape)
print("NORM:", np.linalg.norm(embedding))

print("\nFIRST 10 VALUES:")
print(embedding[:10])