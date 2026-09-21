"""
Generate one evaluation question per article using the LLM.

What it does
------------
1. Pulls every article (id, source, content) from the chunks table.
2. Sends them to OpenRouter in batches of 100.
3. Parses the returned JSON and writes one line per article to
   eval_dataset.jsonl:

   {"article_id": "Civil_Code-Art47", "document": "Civil_Code.pdf",
    "question": "What are the modes of proof for civil status?"}

Resume-safe: already-generated article IDs are skipped if the output
file already exists, so you can interrupt and re-run safely.

Run
---
python generate_questions.py
python generate_questions.py --source Civil_Code.pdf   # one doc only
python generate_questions.py --limit 200               # first N articles
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

from db_pgvector import get_conn

# ── Config ────────────────────────────────────────────────────────────────

OUT_FILE   = Path("eval_dataset.jsonl")
BATCH_SIZE = 50
MODEL      = "nvidia/nemotron-3-ultra-550b-a55b:free"   # 1M context, free
TIMEOUT    = 180   # seconds per API call

# ── Prompt ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a legal question generator for Ethiopian law.

Below is a numbered list of legal articles. Each starts with its ID in square brackets.

For EVERY article write exactly ONE question that:
- Can only be answered by reading that specific article.
- Sounds like a real question a lawyer or citizen would ask.
- Is specific — do not write generic questions that could match many articles.

Return ONLY a JSON array in this exact format, nothing else:
[
  {{"id": "<article_id>", "question": "<question text>"}},
  ...
]

Articles:

{articles}
"""

# ── Helpers ───────────────────────────────────────────────────────────────

def call_openrouter(prompt: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "http://localhost",
            "X-Title":       "legal-rag-eval",
        },
        json={
            "model":       MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens":  16000,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected response: {str(data)[:300]}")
    finish = data["choices"][0].get("finish_reason", "")
    if finish == "length":
        print("    [WARN] Response was truncated (finish_reason=length) — partial results only.")
    return data["choices"][0]["message"]["content"].strip()


def parse_response(raw: str, batch: list[dict]) -> list[dict]:
    """Extract the JSON array from the model response and attach document info."""
    # Strip markdown fences if present
    raw = re.sub(r'^```[a-z]*\n?', '', raw.strip(), flags=re.MULTILINE)
    raw = raw.strip('`').strip()

    # The response is often a clean JSON array — try direct parse first
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            pass  # success
        else:
            raise ValueError("not a list")
    except Exception:
        # Fall back: find the outermost [...] that contains objects
        # Use the LAST '[' to avoid matching article-content brackets
        start = raw.rfind('[{')
        end   = raw.rfind('}]')
        if start == -1 or end == -1:
            # Try any [...] as last resort
            start = raw.find('[')
            end   = raw.rfind(']')
        if start == -1 or end == -1 or end <= start:
            print("    [WARN] No JSON array found in response.")
            return []
        try:
            items = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse error: {e}")
            return []

    id_to_doc = {row["id"]: row["document"] for row in batch}
    results = []
    for entry in items:
        art_id   = str(entry.get("id",       "")).strip()
        question = str(entry.get("question", "")).strip()
        if art_id and question and art_id in id_to_doc:
            results.append({
                "article_id": art_id,
                "document":   id_to_doc[art_id],
                "question":   question,
            })
    return results


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for row in batch:
        # Truncate content so 100 articles comfortably fit in the context window
        content = row["content"][:500].replace("\n", " ")
        lines.append(f'[{row["id"]}]\n{content}')
    return PROMPT_TEMPLATE.format(articles="\n\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate evaluation questions for legal articles")
    ap.add_argument("--out",    default=str(OUT_FILE), help="Output JSONL file")
    ap.add_argument("--source", default="",            help="Filter by source file, e.g. Civil_Code.pdf")
    ap.add_argument("--limit",  type=int, default=0,   help="Max articles to process (0 = all)")
    args = ap.parse_args()

    out_path = Path(args.out)

    # Load already-generated IDs for resume support
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["article_id"])
            except Exception:
                pass
        print(f"Resuming — {len(done_ids)} questions already generated.")

    # Fetch articles from DB
    conn = get_conn()
    cur  = conn.cursor()
    if args.source:
        cur.execute(
            "SELECT id, source, content FROM chunks WHERE source = %s ORDER BY id;",
            (args.source,),
        )
    else:
        cur.execute("SELECT id, source, content FROM chunks ORDER BY source, id;")
    rows = cur.fetchall()
    conn.close()

    articles = [
        {"id": r[0], "document": r[1], "content": r[2]}
        for r in rows
        if r[0] not in done_ids
    ]
    if args.limit:
        articles = articles[: args.limit]

    total = len(articles)
    print(f"Articles to process: {total}  (batch size: {BATCH_SIZE})")

    generated = 0
    failed    = 0

    with out_path.open("a", encoding="utf-8") as f:
        for batch_start in range(0, total, BATCH_SIZE):
            batch      = articles[batch_start : batch_start + BATCH_SIZE]
            batch_num  = batch_start // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\nBatch {batch_num}/{total_batches}  "
                  f"(articles {batch_start+1}–{batch_start+len(batch)}) … ", end="", flush=True)

            prompt = build_prompt(batch)
            try:
                raw     = call_openrouter(prompt)
                entries = parse_response(raw, batch)
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                generated += len(entries)
                print(f"✓  {len(entries)}/{len(batch)} questions generated")
            except Exception as exc:
                print(f"✗  Error: {exc}")
                failed += len(batch)

            # Pause between batches to stay within rate limits
            if batch_start + BATCH_SIZE < total:
                time.sleep(10)

    print(f"\n{'─'*50}")
    print(f"Done.  Generated: {generated}  Failed: {failed}")
    print(f"Dataset saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
