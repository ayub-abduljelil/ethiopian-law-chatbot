import json
import psycopg2
import os

def load_embeddings(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

chunks = load_embeddings("chunk_embeddings_finetuned.jsonl")
legal_containers = load_embeddings("container_embeddings_finetuned.jsonl")

conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ.get("PGDATABASE", "FIRST_RAG"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "a4a3a2a1"),
    )
cur = conn.cursor()

for row in chunks:
    vector = "[" + ",".join(map(str, row["embedding"])) + "]"

    cur.execute(
        """
        UPDATE chunks
        SET embedding = %s
        WHERE id = %s
        """,
        (vector, row["id"])
    )

print(f"Chunks updated: {cur.rowcount}")

for row in legal_containers:
    vector = "[" + ",".join(map(str, row["embedding"])) + "]"

    cur.execute(
        """
        UPDATE legal_containers
        SET embedding = %s
        WHERE id = %s
        """,
        (vector, row["id"])
    )

print(f"Containers updated: {cur.rowcount}")

conn.commit()

cur.close()
conn.close()

print("DONE — database now contains the fine-tuned embeddings.")