import torch

from embedding import Embedding
from db_pgvector import search_containers, search_chunks, search_chunks_exact

QUESTION = "Under what conditions is a child merely conceived considered to have been born for the protection of its interests under Ethiopian law?"

embedding = Embedding()

query = "Under what conditions is a child merely conceived considered to have been born for the protection of its interests under Ethiopian law?"

query_vector = embedding._embed_one(query)

results = search_chunks_exact(query_vector, k=10)

for i, (chunk, distance) in enumerate(results, 1):
    print(
        f"{i}. {chunk['id']} | "
        f"distance={distance:.10f}"
    )