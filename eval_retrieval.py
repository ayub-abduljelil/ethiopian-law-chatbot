"""
Retrieval evaluation script.

Pipeline
--------
1. GENERATE  — pull articles from DB in batches, ask OpenRouter to write
               one question per article, save to eval_dataset.jsonl.

2. EVALUATE  — for each (article_id, question) pair, run two-step retrieval
               with configurable k_containers / k_chunks and record whether
               the target article was retrieved.

3. REPORT    — print hit-rate @ k broken down by document and overall.

Usage
-----
# Step 1 — generate questions (run once, takes a while)
python eval_retrieval.py generate --batch-size 100 --out eval_dataset.jsonl

# Step 2+3 — evaluate retrieval with specific k values
python eval_retrieval.py evaluate --dataset eval_dataset.jsonl \
       --k-containers 3 --k-chunks 8 --direct 5

# Sweep multiple k combinations at once
python eval_retrieval.py sweep --dataset eval_dataset.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

from db_pgvector import get_conn, search_containers, search_chunks, search_chunks_exact
from embedding import Embedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Use a large-context model for batch question generation
GEN_MODEL      = "google/gemini-2.0-flash-001"
MAX_TOKENS_GEN = 4096
TEMPERATURE    = 0.3

# ---------------------------------------------------------------------------
# OpenRouter helper
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, model: str = GEN_MODEL, timeout: int = 120) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "http://localhost",
            "X-Title":       "legal-rag-eval",
        },
        json={
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens":  MAX_TOKENS_GEN,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# STEP 1 — Question generation
# ---------------------------------------------------------------------------

_GEN_SYSTEM = """\
You are a legal question generator.
You will be given a numbered list of legal articles, each prefixed with its ID.
For every article write EXACTLY ONE question that:
  - Can be answered using ONLY that article.
  - Is specific enough that the answer would NOT be found in other articles.
  - Reads like a real question a lawyer or citizen might ask.

Return your response as a JSON array of objects, one per article, in this exact format:
[
  {"id": "<article_id>", "question": "<question text>"},
  ...
]
Return ONLY the JSON array — no extra commentary.
"""

def _build_gen_prompt(batch: list[dict]) -> str:
    lines = []
    for item in batch:
        lines.append(f"[{item['id']}] ({item['document']})\n{item['content'][:600]}")
    articles_text = "\n\n".join(lines)
    return f"{_GEN_SYSTEM}\n\nArticles:\n\n{articles_text}"


def _parse_gen_response(raw: str, batch: list[dict]) -> list[dict]:
    """
    Parse the JSON array returned by the model.
    Falls back gracefully if the model returns extra text around the JSON.
    """
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip(" \n`")

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON array anywhere in the response
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group())
            except Exception:
                print(f"  [WARN] Could not parse generation response for batch.")
                return []
        else:
            return []

    # Build a lookup so we can enrich with document info
    id_to_doc = {item["id"]: item["document"] for item in batch}
    result = []
    for entry in items:
        art_id   = entry.get("id", "").strip()
        question = entry.get("question", "").strip()
        if art_id and question:
            result.append({
                "article_id": art_id,
                "document":   id_to_doc.get(art_id, ""),
                "question":   question,
            })
    return result


def cmd_generate(args):
    out_path   = Path(args.out)
    batch_size = args.batch_size
    limit      = args.limit       # 0 = all
    source_filter = args.source   # e.g. "Civil_Code.pdf"

    # Load already-generated IDs so we can resume
    existing_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(json.loads(line)["article_id"])
            except Exception:
                pass
        print(f"Resuming: {len(existing_ids)} questions already generated.")

    # Fetch articles from DB
    conn = get_conn()
    cur  = conn.cursor()
    if source_filter:
        cur.execute(
            "SELECT id, source, content FROM chunks WHERE source = %s ORDER BY id;",
            (source_filter,),
        )
    else:
        cur.execute("SELECT id, source, content FROM chunks ORDER BY source, id;")
    rows = cur.fetchall()
    conn.close()

    articles = [
        {"id": r[0], "document": r[1], "content": r[2]}
        for r in rows
        if r[0] not in existing_ids
    ]
    if limit:
        articles = articles[:limit]

    total      = len(articles)
    generated  = 0
    skipped    = 0
    print(f"Articles to process: {total}")

    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, total, batch_size):
            batch = articles[start : start + batch_size]
            print(f"  Batch {start // batch_size + 1}: articles {start+1}–{start+len(batch)}", end=" … ")

            prompt = _build_gen_prompt(batch)
            try:
                raw     = _call_openrouter(prompt, timeout=180)
                entries = _parse_gen_response(raw, batch)
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                generated += len(entries)
                print(f"✓ {len(entries)} questions")
            except Exception as exc:
                print(f"✗ Error: {exc}")
                skipped += len(batch)
            # Brief pause to avoid rate-limiting
            time.sleep(1)

    print(f"\nDone. Generated {generated} questions, skipped {skipped} articles.")
    print(f"Dataset saved to: {out_path}")


# ---------------------------------------------------------------------------
# STEP 2 — Retrieval evaluation
# ---------------------------------------------------------------------------

def _retrieve(
    query_vector:  list,
    k_containers:  int,
    k_chunks:      int,
    k_direct:      int,
) -> list[str]:
    """
    Run two-step retrieval and return the list of retrieved article IDs.
    """
    seen:   set[str]  = set()
    ranked: list[tuple[float, str]] = []

    # Step 1 — container search
    containers = search_containers(query_vector, k=k_containers)

    # Step 2 — articles from each container
    for container, _ in containers:
        hits = search_chunks_exact(query_vector, k=k_chunks, container_id=container["id"])
        for chunk, dist in hits:
            cid = chunk["id"]
            if cid not in seen:
                seen.add(cid)
                ranked.append((dist, cid))

    # Direct fallback
    for chunk, dist in search_chunks_exact(query_vector, k=k_direct):
        cid = chunk["id"]
        if cid not in seen:
            seen.add(cid)
            ranked.append((dist, cid))

    ranked.sort(key=lambda x: x[0])
    return [cid for _, cid in ranked]


def cmd_evaluate(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    k_containers = args.k_containers
    k_chunks     = args.k_chunks
    k_direct     = args.direct

    # Load dataset
    entries = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    if not entries:
        print("Dataset is empty.")
        sys.exit(1)

    print(f"Loaded {len(entries)} question-article pairs.")
    print(f"Config: k_containers={k_containers}, k_chunks={k_chunks}, k_direct={k_direct}\n")

    emb_model = Embedding()

    # Per-document counters
    doc_hits:  dict[str, int] = {}
    doc_total: dict[str, int] = {}
    total_hits = 0

    results = []   # for saving detailed output

    for i, entry in enumerate(entries, 1):
        art_id   = entry["article_id"]
        document = entry.get("document", "")
        question = entry["question"]

        query_vector = emb_model._embed_one(question)
        retrieved    = _retrieve(query_vector, k_containers, k_chunks, k_direct)
        hit          = art_id in retrieved
        rank         = (retrieved.index(art_id) + 1) if hit else None

        doc_hits[document]  = doc_hits.get(document, 0)  + (1 if hit else 0)
        doc_total[document] = doc_total.get(document, 0) + 1
        total_hits          += (1 if hit else 0)

        results.append({**entry, "hit": hit, "rank": rank, "retrieved": retrieved[:10]})

        if i % 50 == 0 or i == len(entries):
            pct = total_hits / i * 100
            print(f"  Progress {i}/{len(entries)} — running hit-rate: {pct:.1f}%")

    # Save detailed results
    results_path = dataset_path.parent / f"eval_results_kc{k_containers}_kch{k_chunks}.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print report
    overall = total_hits / len(entries) * 100
    print(f"\n{'='*55}")
    print(f"  k_containers={k_containers}  k_chunks={k_chunks}  k_direct={k_direct}")
    print(f"  Overall hit-rate: {total_hits}/{len(entries)} = {overall:.1f}%")
    print(f"{'='*55}")
    print(f"  {'Document':<45} {'Hit-rate':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for doc in sorted(doc_total):
        h  = doc_hits[doc]
        t  = doc_total[doc]
        pct = h / t * 100
        name = doc.replace(".pdf", "")[:44]
        print(f"  {name:<45} {h}/{t} ({pct:.0f}%)")
    print(f"{'='*55}")
    print(f"Detailed results saved to: {results_path}")


# ---------------------------------------------------------------------------
# STEP 3 — Sweep multiple k combinations
# ---------------------------------------------------------------------------

def cmd_sweep(args):
    """Try a grid of k_containers × k_chunks and print a comparison table."""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    entries = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    if not entries:
        print("Dataset is empty."); sys.exit(1)

    # k grid to sweep
    kc_values  = args.kc_values  or [1, 2, 3, 5]
    kch_values = args.kch_values or [5, 8, 12, 20]
    k_direct   = args.direct

    emb_model    = Embedding()
    print(f"Sweeping {len(kc_values) * len(kch_values)} combinations on {len(entries)} questions …\n")

    table: list[tuple] = []

    for kc in kc_values:
        for kch in kch_values:
            hits = 0
            for entry in entries:
                qv        = emb_model._embed_one(entry["question"])
                retrieved = _retrieve(qv, kc, kch, k_direct)
                if entry["article_id"] in retrieved:
                    hits += 1
            pct = hits / len(entries) * 100
            table.append((kc, kch, hits, len(entries), pct))
            print(f"  k_containers={kc:2d}  k_chunks={kch:2d}  →  {hits}/{len(entries)} ({pct:.1f}%)")

    print(f"\n{'='*55}")
    print(f"  Best configuration:")
    best = max(table, key=lambda x: x[4])
    print(f"  k_containers={best[0]}  k_chunks={best[1]}  →  {best[2]}/{best[3]} ({best[4]:.1f}%)")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Retrieval evaluation for the Ethiopian Legal RAG pipeline"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # generate
    g = sub.add_parser("generate", help="Generate evaluation questions via LLM")
    g.add_argument("--out",        default="eval_dataset.jsonl",
                   help="Output JSONL file (default: eval_dataset.jsonl)")
    g.add_argument("--batch-size", type=int, default=100,
                   help="Articles per LLM call (default: 100)")
    g.add_argument("--limit",      type=int, default=0,
                   help="Max articles to process (0 = all)")
    g.add_argument("--source",     default="",
                   help="Filter by source file, e.g. Civil_Code.pdf")

    # evaluate
    e = sub.add_parser("evaluate", help="Evaluate retrieval hit-rate")
    e.add_argument("--dataset",      required=True, help="Path to eval_dataset.jsonl")
    e.add_argument("--k-containers", type=int, default=3)
    e.add_argument("--k-chunks",     type=int, default=8)
    e.add_argument("--direct",       type=int, default=5,
                   help="Number of direct (non-hierarchical) fallback chunks")

    # sweep
    s = sub.add_parser("sweep", help="Sweep multiple k combinations")
    s.add_argument("--dataset",    required=True)
    s.add_argument("--kc-values",  type=int, nargs="+", default=[1, 2, 3, 5],
                   help="k_containers values to try (default: 1 2 3 5)")
    s.add_argument("--kch-values", type=int, nargs="+", default=[5, 8, 12, 20],
                   help="k_chunks values to try (default: 5 8 12 20)")
    s.add_argument("--direct",     type=int, default=5)

    args = ap.parse_args()

    if   args.cmd == "generate":  cmd_generate(args)
    elif args.cmd == "evaluate":  cmd_evaluate(args)
    elif args.cmd == "sweep":     cmd_sweep(args)


if __name__ == "__main__":
    main()
