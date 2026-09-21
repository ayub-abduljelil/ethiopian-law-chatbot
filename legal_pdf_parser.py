"""
PDF Parser for Ethiopian Legal Documents — all four article styles + hierarchy.

Style A: Art. 6. - Title      (Civil Code)
Style B: Article 6 / Title    (Constitution)
Style C: Article 37. Title    (New Commercial Code)
Style D: Article 1.- Title    (Criminal Code)
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber

# ---------------------------------------------------------------------------
# OCR constant
# ---------------------------------------------------------------------------
_OCR_NUM = r'[0-9lISB]+'   # 1→l/I, 5→S, 8→B common OCR substitutions

# ---------------------------------------------------------------------------
# Article patterns
# ---------------------------------------------------------------------------
_ARTICLE_HEADER_RE = re.compile(
    r'^(?:Article|Art\.)\s+' + _OCR_NUM + r'(?:\.-|-\s|\.\s+[A-Z]|\.\s+-|\s+[A-Z]|$)',
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(
    r'(?=^(?:Article|Art\.)\s+' + _OCR_NUM + r')',
    re.IGNORECASE | re.MULTILINE,
)
_ARTICLE_NUM_RE = re.compile(
    r'^(?:Article|Art\.)\s+(' + _OCR_NUM + r')',
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(
    r'^(?:Article|Art\.)\s+' + _OCR_NUM + r'[^\n]*\s+\d+\s*$',
    re.IGNORECASE,
)

def _normalize_article_num(raw: str) -> str:
    s = re.sub(r'[lI]', '1', raw)
    s = re.sub(r'S', '5', s)
    s = re.sub(r'B', '8', s)
    return re.sub(r'[^0-9]', '', s)

# ---------------------------------------------------------------------------
# Hierarchy patterns
# ---------------------------------------------------------------------------
_HIER_RE = re.compile(
    r'^(PART|BOOK|TITLE|CHAPTER|SECTION|SUB[\s-]?SECTION|DIVISION)(\s.+)?$',
    re.IGNORECASE,
)
_HIER_FIELD = {
    "part": "part", "book": "book", "title": "title",
    "chapter": "chapter", "section": "section",
    "subsection": "sub_section", "sub-section": "sub_section",
    "sub section": "sub_section", "division": "section",
}
_HIER_PRIORITY = {
    "part": 1, "book": 2, "title": 3,
    "chapter": 4, "section": 5, "sub_section": 6,
}
_ORDINAL_RE = re.compile(
    r'^\s*(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|'
    r'ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|'
    r'NINETEEN|TWENTY(?:[-\s]?(?:ONE|TWO|THREE))?|THIRTY|FORTY|FIFTY)'
    r'[\s~\'.\-]*',
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# ToC / label helpers
# ---------------------------------------------------------------------------

def _text_fingerprint(s: str) -> str:
    return re.sub(r'[^a-z]', '', s.lower())

def _roman_numeral(s: str) -> str:
    m = re.search(r'\b([IVXLCDM]{1,8})\b', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r'(?:BOOK|TITLE)\s*[\'.\s~]*(\d+)\b', s, re.IGNORECASE)
    if m2:
        return re.sub(r'\d', 'I', m2.group(1)).upper()
    return ""

def _build_toc_lookup(pdf_path: str, toc_pages: int) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    hier_kw   = re.compile(r'^\s*(BOOK|TITLE)\b', re.IGNORECASE)
    toc_entry = re.compile(r'\d+\s*$')
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(toc_pages, len(pdf.pages))):
            for line in (pdf.pages[i].extract_text() or "").split("\n"):
                s = line.strip()
                if not hier_kw.match(s): continue
                if toc_entry.search(s): continue
                if len(s) < 6: continue
                fp = _text_fingerprint(s)
                if fp and fp not in lookup:
                    lookup[fp] = s
    return lookup

def _build_toc_chapter_map(pdf_path: str, toc_pages: int) -> Dict[str, Dict]:
    chapter_map: Dict[str, Dict] = {}
    book_re    = re.compile(r'^\s*BOOK\b',    re.IGNORECASE)
    title_re   = re.compile(r'^\s*TITLE\b',   re.IGNORECASE)
    chapter_re = re.compile(r'^\s*Chapter\b', re.IGNORECASE)
    toc_num    = re.compile(r'\d+\s*$')
    current_book = current_title = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(toc_pages, len(pdf.pages))):
            for line in (pdf.pages[i].extract_text() or "").split("\n"):
                s = line.strip()
                if not s or len(s) < 4: continue
                is_entry = bool(toc_num.search(s))
                if   book_re.match(s)    and not is_entry: current_book = s; current_title = ""
                elif title_re.match(s)   and not is_entry: current_title = s
                elif chapter_re.match(s):
                    label = re.sub(r'\s+\d+\s*$', '', s).strip()
                    fp    = _text_fingerprint(label)
                    if fp and fp not in chapter_map:
                        chapter_map[fp] = {"book": current_book, "title": current_title}
    return chapter_map

def _resolve_hierarchy_label(raw: str, toc_lookup: Dict[str, str]) -> str:
    if not toc_lookup: return raw
    raw_fp    = _text_fingerprint(raw)
    raw_roman = _roman_numeral(raw)
    if not raw_fp: return raw
    raw_set          = set(raw_fp)
    best_label       = raw
    best_score       = 0.0
    best_roman_match = False
    for fp, clean in toc_lookup.items():
        fp_set    = set(fp)
        union_len = len(raw_set | fp_set)
        if not union_len: continue
        jaccard      = len(raw_set & fp_set) / union_len
        common       = sum(1 for a, b in zip(raw_fp, fp) if a == b)
        shorter      = min(len(raw_fp), len(fp))
        prefix_ratio = common / shorter if shorter else 0
        clean_roman  = _roman_numeral(clean)
        roman_bonus  = 0.20 if (raw_roman and clean_roman and raw_roman == clean_roman) else 0.0
        score        = 0.50 * jaccard + 0.30 * prefix_ratio + roman_bonus
        if score > best_score:
            best_score = score; best_label = clean; best_roman_match = roman_bonus > 0
    if best_score >= 0.55: return best_label
    if best_score >= 0.45 and best_roman_match: return best_label
    for fp, clean in toc_lookup.items():
        fp_set    = set(fp)
        union_len = len(raw_set | fp_set)
        if union_len and len(raw_set & fp_set) / union_len >= 0.85:
            return clean
    return raw

def _clean_label(raw: str) -> str:
    s = re.sub(r'^(PART|BOOK|TITLE|CHAPTER|SECTION|SUB[\s-]?SECTION|DIVISION)\s*', '', raw.strip(), flags=re.IGNORECASE)
    s = _ORDINAL_RE.sub('', s)
    s = re.sub(r'^[\s~\'.\-:]+', '', s).strip()
    if s:
        if re.search(r'[~<>;@#\$\^=\|\\\{\}]', s): return ""
        if re.match(r"^[\d'\"`]+", s): return ""
        if re.match(r'^Pa.{0,2}e?\s*\d*$', s, re.IGNORECASE): return ""
        if re.match(r'^[\w]+\s+\d+[\s\.–-]', s): return ""
        clean_chars = sum(1 for c in s if c.isalpha() or c == ' ')
        if len(s) > 0 and clean_chars / len(s) < 0.75: return ""
    s = re.sub(r'\s+\d+\s*$', '', s).strip()
    s = re.sub(r'[\s.\-]+$', '', s).strip()
    return s.title() if s else ""

def _build_embedding_text(doc_title: str, levels: Dict[str, str]) -> str:
    parts = [doc_title]
    for field in ("part", "book", "title", "chapter", "section", "sub_section"):
        raw = levels.get(field)
        if not raw: continue
        cleaned = _clean_label(raw)
        if cleaned: parts.append(cleaned)
    result = ". ".join(parts) + "."
    result = re.sub(r'[^\x20-\x7E]', '', result)
    result = re.sub(r'\.{2,}', '.', result)
    return re.sub(r'\s{2,}', ' ', result).strip()

def _is_hierarchy_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    m = _HIER_RE.match(s)
    if not m: return None
    kw  = m.group(1).lower().replace(" ", "").replace("-", "")
    if kw.startswith("sub"): kw = "sub_section"
    field = _HIER_FIELD.get(kw) or _HIER_FIELD.get(m.group(1).lower())
    if not field: return None
    rest = (m.group(2) or "").strip()
    if rest and re.match(r'^(of|the|a |an |in |to |is |it |by |or |and |jointly|together)', rest, re.IGNORECASE): return None
    if re.search(r'[,;:]\s+\w', rest): return None
    if re.search(r'\(Arts?\.\s*\d+', rest, re.IGNORECASE): return None
    if len(s.split()) > 10: return None
    raw_label = (m.group(1) + (" " + rest if rest else "")).strip()
    if len(raw_label) < 3: return None
    return field, raw_label

# ---------------------------------------------------------------------------
# Container state machine
# ---------------------------------------------------------------------------

class _ContainerState:
    def __init__(self):
        self._levels: Dict[str, str] = {}

    @property
    def levels(self) -> Dict[str, str]:
        return dict(self._levels)

    def update(self, field: str, raw_label: str) -> None:
        priority = _HIER_PRIORITY.get(field, 99)
        for f, p in list(_HIER_PRIORITY.items()):
            if p >= priority:
                self._levels.pop(f, None)
        self._levels[field] = raw_label

    def is_empty(self) -> bool:
        return len(self._levels) == 0

    def container_id(self, doc_stem: str) -> str:
        key = doc_stem + "|" + "|".join(f"{f}:{v}" for f, v in sorted(self._levels.items()))
        return doc_stem + "-" + hashlib.sha1(key.encode()).hexdigest()[:12]

    def hierarchy_path(self) -> str:
        return " > ".join(
            self._levels[f]
            for f in ("part", "book", "title", "chapter", "section", "sub_section")
            if f in self._levels
        )

# ---------------------------------------------------------------------------
# Hierarchy correction
# ---------------------------------------------------------------------------

def _correct_container_hierarchy(
    containers: Dict[str, Dict],
    chapter_map: Dict[str, Dict],
    toc_lookup:  Dict[str, str],
) -> None:
    for cid, c in containers.items():
        chap = c.get("chapter")
        if not chap: continue
        chap_fp  = _text_fingerprint(chap)
        if not chap_fp: continue
        best_fp    = None
        best_score = 0.0
        chap_set   = set(chap_fp)
        for fp in chapter_map:
            fp_set    = set(fp)
            union_len = len(chap_set | fp_set)
            if not union_len: continue
            score = len(chap_set & fp_set) / union_len
            if score > best_score:
                best_score = score; best_fp = fp
        if best_fp is None or best_score < 0.75: continue
        correct       = chapter_map[best_fp]
        correct_book  = correct["book"]
        correct_title = correct["title"]
        current_book  = c.get("book",  "") or ""
        current_title = c.get("title", "") or ""
        needs_fix = (
            (correct_book  and _text_fingerprint(correct_book)  != _text_fingerprint(current_book)) or
            (correct_title and _text_fingerprint(correct_title) != _text_fingerprint(current_title))
        )
        if not needs_fix: continue
        if correct_book:  c["book"]  = correct_book
        if correct_title: c["title"] = correct_title or None
        levels     = {k: c.get(k) for k in ("part","book","title","chapter","section","sub_section") if c.get(k)}
        path_parts = [levels[k] for k in ("part","book","title","chapter","section","sub_section") if k in levels]
        c["hierarchy_path"] = " > ".join(path_parts)

# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class LegalPDFParser:
    DEFAULT_MARGINS = {"left": 0.08, "top": 0.10, "right": 0.08, "bottom": 0.10}

    def __init__(self, pdf_path: str, bbox=None, margins=None, skip_pages: int = 0, toc_pages: int = 0):
        self.pdf_path    = Path(pdf_path)
        self.custom_bbox = bbox
        self.margins     = margins or self.DEFAULT_MARGINS.copy()
        self.skip_pages  = skip_pages
        self.toc_pages   = toc_pages
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _calculate_bbox(self, page):
        pw, ph = page.width, page.height
        if self.custom_bbox:
            x0, y0, x1, y1 = self.custom_bbox
            return (max(0,min(x0,pw)), max(0,min(y0,ph)), max(0,min(x1,pw)), max(0,min(y1,ph)))
        return (pw*self.margins["left"], ph*self.margins["top"],
                pw*(1-self.margins["right"]), ph*(1-self.margins["bottom"]))

    def extract_page_text(self, page) -> str:
        try:    return page.within_bbox(self._calculate_bbox(page)).extract_text() or ""
        except: return page.extract_text() or ""

    def _normalize_newlines(self, text: str) -> str:
        text = re.sub(
            r'((?:Article|Art\.)\s+[0-9lISB]+(?:\.\s+[A-Za-z][^\n]*|(?:\.\s*-[^\n]*)|\s*))(?:\n|(?=\s+[0-9lISB]+[^\n]*\n))',
            lambda m: '\x01' + m.group(1).rstrip() + '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n', ' ', text)
        text = text.replace('\x01', '\n')
        return re.sub(r' {2,}', ' ', text).strip()

    def _chunk_by_articles(self, text: str) -> List[str]:
        valid = []
        for chunk in _SPLIT_RE.split(text):
            chunk = chunk.strip()
            if not chunk: continue
            first_line = chunk.splitlines()[0].strip()
            if not _ARTICLE_HEADER_RE.match(first_line): continue
            if re.search(r'\s+\d+\s*$', first_line) and len(first_line.split()) <= 8: continue
            body_match = re.match(r'^(?:Article|Art\.)\s+[0-9lISB]+(?:[.\-][^\n]*)?\s+(.*)', chunk, re.IGNORECASE|re.DOTALL)
            body = body_match.group(1).strip() if body_match else ''
            if not body:
                inline_match = re.match(r'^(?:Article|Art\.)\s+[0-9lISB]+\.\s+\S+.*?\s+((?:[A-Z0-9]|\d).{10,})', first_line, re.IGNORECASE)
                if inline_match: body = inline_match.group(1).strip()
            if not body: continue
            valid.append(chunk)
        return valid

    def _extract_pages(self) -> List[str]:
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if page_num <= self.skip_pages: continue
                try:
                    t = self.extract_page_text(page)
                    if t.strip(): pages.append(t)
                except Exception as e:
                    print(f"Warning: Failed page {page_num}: {e}")
        return pages

    def parse(self) -> List[Tuple[str, str]]:
        full      = "\n\n".join(self._extract_pages())
        norm      = self._normalize_newlines(full)
        result    = []
        for chunk in self._chunk_by_articles(norm):
            m = _ARTICLE_NUM_RE.match(chunk.strip())
            if m:
                num = _normalize_article_num(m.group(1))
                if num: result.append((f"Art{num}", chunk.strip()))
        return result

    def parse_with_hierarchy(self, doc_stem: str, doc_title: str):
        raw_pages   = self._extract_pages()
        toc_lookup  = _build_toc_lookup(str(self.pdf_path), self.toc_pages)  if self.toc_pages > 0 else {}
        chapter_map = _build_toc_chapter_map(str(self.pdf_path), self.toc_pages) if self.toc_pages > 0 else {}

        state      = _ContainerState()
        containers : Dict[str, Dict] = {}
        all_lines  : List[str]       = []
        for p in raw_pages:
            all_lines.extend(p.split("\n"))
        n = len(all_lines)

        def _register():
            cid = state.container_id(doc_stem)
            if cid not in containers:
                lvls = state.levels
                containers[cid] = {
                    "id": cid, "document": doc_stem,
                    "hierarchy_path": state.hierarchy_path(),
                    "embedding_text": _build_embedding_text(doc_title, lvls),
                    "book": lvls.get("book"), "part": lvls.get("part"),
                    "title": lvls.get("title"), "chapter": lvls.get("chapter"),
                    "section": lvls.get("section"), "sub_section": lvls.get("sub_section"),
                }
            return cid

        def _subtitle(idx: int) -> str:
            for j in range(idx+1, min(idx+3, n)):
                nxt = all_lines[j].strip()
                if not nxt: continue
                if _is_hierarchy_line(nxt): break
                if _ARTICLE_HEADER_RE.match(nxt): break
                if re.match(r'^\d+$', nxt): break
                if len(nxt) <= 4: break
                if re.match(r'^(Para(graph)?|Sub-?para)', nxt, re.IGNORECASE): break
                if re.match(r'^Art\.?\s+\d+', nxt, re.IGNORECASE): break
                if len(nxt.split()) > 10: break
                if re.match(r'^[\w]+\s+\d+[\s\.–-]', nxt): break
                if re.search(r'\b\d+\s*$', nxt) and len(nxt.split()) <= 4: break
                if re.match(r'^(Civil|Criminal|Commercial|Maritime|Family|CIVIL|CRIMINAL)\s+\w+\s+\d+', nxt): break
                return nxt
            return ""

        # Pass 1 — build containers
        for i, line in enumerate(all_lines):
            result = _is_hierarchy_line(line)
            if not result: continue
            field, raw_label = result
            sub = _subtitle(i)
            if sub: raw_label = raw_label + " - " + sub
            if field in ("book", "title") and toc_lookup:
                raw_label = _resolve_hierarchy_label(raw_label, toc_lookup)
            state.update(field, raw_label)
            _register()

        # Correct misassigned book/title using ToC
        if chapter_map:
            _correct_container_hierarchy(containers, chapter_map, toc_lookup)
            for cid, c in containers.items():
                lvls = {k: c.get(k) for k in ("part","book","title","chapter","section","sub_section") if c.get(k)}
                c["embedding_text"] = _build_embedding_text(doc_title, lvls)

        # Root container
        root_id = doc_stem + "-root"
        if root_id not in containers:
            containers[root_id] = {
                "id": root_id, "document": doc_stem,
                "hierarchy_path": doc_title, "embedding_text": doc_title + ".",
                "book": None, "part": None, "title": None,
                "chapter": None, "section": None, "sub_section": None,
            }

        # Pass 2 — articles
        norm           = self._normalize_newlines("\n\n".join(raw_pages))
        article_chunks = self._chunk_by_articles(norm)
        articles: List[Tuple[str, str]] = []
        for chunk in article_chunks:
            m = _ARTICLE_NUM_RE.match(chunk.strip())
            if m:
                num = _normalize_article_num(m.group(1))
                if num: articles.append((f"Art{num}", chunk.strip()))

        # Pass 3 — assign articles to containers
        article_container_map: Dict[str, str] = {}
        state2      = _ContainerState()
        current_cid = root_id
        for i, line in enumerate(all_lines):
            h = _is_hierarchy_line(line)
            if h:
                field, raw_label = h
                sub = _subtitle(i)
                if sub: raw_label = raw_label + " - " + sub
                if field in ("book", "title") and toc_lookup:
                    raw_label = _resolve_hierarchy_label(raw_label, toc_lookup)
                state2.update(field, raw_label)
                cid = state2.container_id(doc_stem)
                if cid in containers: current_cid = cid
            stripped = line.strip()
            am = _ARTICLE_NUM_RE.match(stripped)
            if am:
                num    = _normalize_article_num(am.group(1))
                art_id = f"Art{num}" if num else None
                if art_id and art_id not in article_container_map:
                    if not (re.search(r'\s+\d+\s*$', stripped) and len(stripped.split()) <= 8):
                        article_container_map[art_id] = current_cid

        for art_id, _ in articles:
            if art_id not in article_container_map:
                article_container_map[art_id] = root_id

        if not any(v == root_id for v in article_container_map.values()):
            containers.pop(root_id, None)

        return articles, containers, article_container_map


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def batch_parse_pdfs(pdf_directory, bbox=None, margins=None, skip_pages=0):
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.is_dir(): raise NotADirectoryError(f"Not a directory: {pdf_directory}")
    all_results = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        try:
            print(f"Processing: {pdf_file.name}...", end=" ")
            parser   = LegalPDFParser(str(pdf_file), bbox=bbox, margins=margins, skip_pages=skip_pages)
            articles = parser.parse()
            for art_id, content in articles:
                all_results.append((pdf_file.name, art_id, content))
            print(f"✓ ({len(articles)} articles)")
        except Exception as e:
            print(f"✗ Error: {e}")
    return all_results
