from db_pgvector import get_conn
conn = get_conn()
cur  = conn.cursor()
cur.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source;")
print("Chunks per source:")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")
cur.execute("SELECT document, COUNT(*) FROM legal_containers GROUP BY document ORDER BY document;")
print("\nContainers per document:")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")
conn.close()
