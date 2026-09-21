"""
Ingestion pipeline — PDF → hierarchy containers → article chunks → Postgres.

Flow
----
PDF
 └─ LegalPDFParser.parse_with_hierarchy()
     ├─ articles          : [(article_id, content)]
     ├─ containers        : {container_id → container_dict}
     └─ article_container_map : {article_id → container_id}

For each container  → embed embedding_text → upsert_container()
For each article    → embed content        → upsert_chunk(container_id=...)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from embedding import Embedding
from legal_pdf_parser import LegalPDFParser
from db_pgvector import init_db, upsert_container, upsert_chunk


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata_index(pdfs_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata_file = pdfs_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata.json: {e}")
    return {}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def ingest_pdfs(
    pdf_paths:  Optional[List[str]] = None,
    pdfs_dir:   Optional[str]       = None,
    bbox:       Optional[tuple]     = None,
    margins:    Optional[dict]      = None,
    skip_pages: int                 = 0,
    verbose:    bool                = False,
) -> None:
    emb_model = Embedding()
    dim = emb_model.get_embeddings(["test"], prefix="passage: ").shape[1]
    init_db(dim=dim)

    # Locate metadata.json
    if pdfs_dir:
        metadata_index = load_metadata_index(Path(pdfs_dir))
    else:
        metadata_index = {}
        if pdf_paths:
            for candidate in [
                Path(pdf_paths[0]).parent.parent / "metadata.json",
                Path(pdf_paths[0]).parent / "metadata.json",
            ]:
                if candidate.exists():
                    try:
                        with open(candidate, "r", encoding="utf-8") as f:
                            metadata_index = json.load(f)
                    except Exception:
                        pass
                    break

    if pdfs_dir:
        pdf_list = sorted(Path(pdfs_dir).rglob("*.pdf"))
        if verbose:
            print(f"Found {len(pdf_list)} PDF(s) in {pdfs_dir}")
        for p in pdf_list:
            _ingest_single(p, emb_model, metadata_index, bbox, margins, skip_pages, verbose)
    elif pdf_paths:
        for p in pdf_paths:
            _ingest_single(Path(p), emb_model, metadata_index, bbox, margins, skip_pages, verbose)


# ---------------------------------------------------------------------------
# Per-document ingestion
# ---------------------------------------------------------------------------

def _ingest_single(
    pdf_path:       Path,
    emb_model:      "Embedding",
    metadata_index: Dict[str, Dict[str, Any]],
    bbox:           Optional[tuple],
    margins:        Optional[dict],
    skip_pages:     int,
    verbose:        bool,
) -> None:
    pdf_path = Path(pdf_path)
    doc_stem = pdf_path.stem
    if verbose:
        print(f"\nProcessing: {pdf_path.name}")

    try:
        doc_meta  = metadata_index.get(doc_stem, {})
        doc_title = doc_meta.get("title", doc_stem.replace("_", " "))

        # skip_pages: CLI value > 0 wins; else metadata.json
        effective_skip = skip_pages if skip_pages > 0 else doc_meta.get("skip_pages", 0)
        toc_pages      = doc_meta.get("toc_pages", 0)

        # Category from metadata or folder name
        category = doc_meta.get("category", "")

        base_meta = {
            **{k: v for k, v in doc_meta.items() if k not in ("skip_pages", "toc_pages")},
            "source_path":   str(pdf_path.resolve()),
            "filename":      pdf_path.name,
            "document_name": doc_title,
            "category":      category,
        }

        # ── Parse ─────────────────────────────────────────────────────────
        parser = LegalPDFParser(
            str(pdf_path),
            bbox=bbox,
            margins=margins,
            skip_pages=effective_skip,
            toc_pages=toc_pages,
        )
        articles, containers, art_container_map = parser.parse_with_hierarchy(
            doc_stem=doc_stem,
            doc_title=doc_title,
        )

        if verbose:
            print(f"  Parsed  : {len(articles)} articles, {len(containers)} containers")

        # ── Step 1: embed and store containers ────────────────────────────
        if containers:
            emb_texts        = [c["embedding_text"] for c in containers.values()]
            container_embeds = emb_model.get_embeddings(emb_texts)

            for idx, (cid, container) in enumerate(containers.items()):
                upsert_container(
                    container_id   = cid,
                    document       = doc_stem,
                    hierarchy_path = container["hierarchy_path"],
                    embedding_text = container["embedding_text"],
                    embedding      = container_embeds[idx].tolist(),
                    book           = container.get("book"),
                    part           = container.get("part"),
                    title          = container.get("title"),
                    chapter        = container.get("chapter"),
                    section        = container.get("section"),
                    sub_section    = container.get("sub_section"),
                    metadata       = {**base_meta, "container_id": cid},
                )

        if verbose:
            print(f"  Stored  : {len(containers)} containers")

        # ── Step 2: embed and store article chunks ─────────────────────────
        ingested = 0
        for article_id, content in articles:
            if not content.strip():
                continue
            chunk_id     = f"{doc_stem}-{article_id}"
            container_id = art_container_map.get(article_id)
            vector = emb_model._embed_one(content, prefix="passage: ")
            upsert_chunk(
                chunk_id     = chunk_id,
                content      = content,
                source       = pdf_path.name,
                embedding    = vector,
                metadata     = {
                    **base_meta,
                    "article_id":   article_id,
                    "chunk_type":   "legal_article",
                    "container_id": container_id,
                },
                container_id = container_id,
            )
            ingested += 1

        if verbose:
            print(f"  Ingested: {ingested} articles from {pdf_path.name}")

    except Exception as e:
        import traceback
        print(f"  ✗ Error processing {pdf_path}: {e}")
        if verbose:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ingest PDFs into Postgres hierarchical legal store"
    )
    ap.add_argument("--pdf", action="append", dest="pdf",
                    help="Path to a specific PDF (repeat for multiple files)")
    ap.add_argument("--pdfs-dir",
                    help="Directory to scan recursively for PDFs")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="Absolute crop box in points")
    ap.add_argument("--margins", type=float, nargs=4,
                    metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    help="Proportional margins (0.0–1.0)")
    ap.add_argument("--skip-pages", type=int, default=0,
                    help="Leading pages to skip (overrides metadata.json when > 0)")
    ap.add_argument("--verbose", "-v", action="store_true")

    args = ap.parse_args()
    if not args.pdf and not args.pdfs_dir:
        ap.error("Provide --pdf or --pdfs-dir")

    bbox    = tuple(args.bbox) if args.bbox else None
    margins = None
    if args.margins:
        margins = {
            "left": args.margins[0], "top":    args.margins[1],
            "right": args.margins[2], "bottom": args.margins[3],
        }

    ingest_pdfs(
        pdf_paths  = args.pdf,
        pdfs_dir   = args.pdfs_dir,
        bbox       = bbox,
        margins    = margins,
        skip_pages = args.skip_pages,
        verbose    = args.verbose,
    )
