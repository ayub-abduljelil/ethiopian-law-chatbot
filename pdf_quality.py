"""
Quick PDF quality audit.
Samples 10 pages from each PDF and scores OCR noise:
  - % of characters that are non-ASCII or symbols
  - % of words that look like real English/Latin words
  - avg word length (garbled OCR tends to produce very long or very short tokens)
"""
import re
import pdfplumber
from pathlib import Path

PDFS = [
    ("pdfs/Ethiopian-Codes/Civil_Code.pdf",           18),
    ("pdfs/Ethiopian-Codes/Criminal_Code.pdf",        53),
    ("pdfs/Ethiopian-Codes/New_Commercial_Code.pdf",  59),
    ("pdfs/Ethiopian-Codes/Family_Code.pdf",           6),
    ("pdfs/Ethiopian-Codes/Maritime_Code.pdf",        18),
    ("pdfs/Ethiopian-Codes/Civil_Procedure_Code.pdf", 66),
    ("pdfs/Ethiopia_Constitution.pdf",                 0),
]

def score_text(text: str) -> dict:
    if not text.strip():
        return {"noise_pct": 100, "real_word_pct": 0, "avg_word_len": 0, "chars": 0}

    chars      = len(text)
    # Non-ASCII or known noise symbols
    noise      = sum(1 for c in text if ord(c) > 127 or c in "~<>{}[]|^\\@#$%")
    noise_pct  = round(noise / chars * 100, 1)

    words = re.findall(r'[A-Za-z]+', text)
    if not words:
        return {"noise_pct": noise_pct, "real_word_pct": 0, "avg_word_len": 0, "chars": chars}

    # "real" word: 2-20 letters, no run of 3+ consonants (heuristic for garbled text)
    consonants = re.compile(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}')
    real        = [w for w in words if 2 <= len(w) <= 20 and not consonants.search(w)]
    real_pct    = round(len(real) / len(words) * 100, 1)
    avg_len     = round(sum(len(w) for w in words) / len(words), 1)

    return {"noise_pct": noise_pct, "real_word_pct": real_pct,
            "avg_word_len": avg_len, "chars": chars}

print(f"\n{'Document':<35} {'Noise%':>7} {'RealWords%':>11} {'AvgWordLen':>11}  Verdict")
print("─" * 80)

for fpath, skip in PDFS:
    p = Path(fpath)
    if not p.exists():
        print(f"  {p.stem:<33} {'MISSING':>7}")
        continue

    all_text = []
    with pdfplumber.open(fpath) as pdf:
        content_pages = [pg for i, pg in enumerate(pdf.pages) if i >= skip]
        # Sample up to 10 evenly spread pages
        step  = max(1, len(content_pages) // 10)
        sample = content_pages[::step][:10]
        for pg in sample:
            t = pg.extract_text() or ""
            all_text.append(t)

    combined = "\n".join(all_text)
    s        = score_text(combined)

    # Verdict
    if s["noise_pct"] > 3 or s["real_word_pct"] < 70:
        verdict = "⚠  BAD OCR"
    elif s["noise_pct"] > 1 or s["real_word_pct"] < 85:
        verdict = "~  MODERATE"
    else:
        verdict = "✓  CLEAN"

    print(f"  {p.stem:<33} {s['noise_pct']:>6}% {s['real_word_pct']:>10}% "
          f"{s['avg_word_len']:>10}   {verdict}")

print()
