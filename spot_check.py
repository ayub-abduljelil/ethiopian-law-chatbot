from db_pgvector import get_conn

conn = get_conn()
cur  = conn.cursor()

# Sample one article from each document
docs = [
    ("Civil_Code.pdf",           "Civil_Code-Art47"),
    ("Criminal_Code.pdf",        "Criminal_Code-Art2"),
    ("New_Commercial_Code.pdf",  "New_Commercial_Code-Art37"),
    ("Family_Code.pdf",          "Family_Code-Art6"),
    ("Civil_Procedure_Code.pdf", "Civil_Procedure_Code-Art148"),
    ("Ethiopia_Constitution.pdf","Ethiopia_Constitution-Art14"),
]

for source, chunk_id in docs:
    cur.execute("SELECT content FROM chunks WHERE id = %s;", (chunk_id,))
    row = cur.fetchone()
    print(f"\n{'='*60}")
    print(f"  {chunk_id}")
    print(f"{'='*60}")
    print(row[0] if row else "NOT FOUND")

conn.close()
