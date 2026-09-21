import os
import json
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from pgvector.psycopg2.vector import Vector

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn():
    """Return a new psycopg2 connection using env vars.

    Env vars (defaults shown):
      PGHOST=localhost  PGPORT=5432  PGDATABASE=first_rag
      PGUSER=postgres   PGPASSWORD=postgres
    """
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ.get("PGDATABASE", "first_rag"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(dim: int = 384) -> None:
    """
    Create all tables and indexes if they do not already exist.

    Schema
    ------
    legal_containers
        One row per structural container (Book, Part, Title, Chapter, Section).
        Containers are embedded from their hierarchy label text so they can be
        searched semantically ("which section covers contract formation?").

    chunks
        One row per legal article.  Each chunk carries a nullable container_id
        FK pointing to the innermost container that holds the article.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # ----------------------------------------------------------
                # legal_containers
                # ----------------------------------------------------------
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS legal_containers (
                        id             TEXT PRIMARY KEY,
                        document       TEXT NOT NULL,
                        hierarchy_path TEXT NOT NULL,
                        embedding_text TEXT NOT NULL,
                        book           TEXT,
                        part           TEXT,
                        title          TEXT,
                        chapter        TEXT,
                        section        TEXT,
                        sub_section    TEXT,
                        embedding      vector({dim}),
                        metadata       JSONB,
                        created_at     TIMESTAMPTZ DEFAULT now()
                    );
                """)

                # ANN index on containers (built after enough rows exist)
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_containers_embedding
                        ON legal_containers
                        USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 50);
                    """)
                except Exception:
                    pass  # harmless on small installs

                # Index to quickly fetch all containers for a document
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_containers_document
                    ON legal_containers (document);
                """)

                # ----------------------------------------------------------
                # chunks  (add container_id column if table already exists)
                # ----------------------------------------------------------
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id           TEXT PRIMARY KEY,
                        content      TEXT NOT NULL,
                        source       TEXT,
                        container_id TEXT REFERENCES legal_containers(id)
                                         ON DELETE SET NULL,
                        metadata     JSONB,
                        embedding    vector({dim}),
                        created_at   TIMESTAMPTZ DEFAULT now()
                    );
                """)

                # If the table already existed without container_id, add it
                cur.execute("""
                    ALTER TABLE chunks
                    ADD COLUMN IF NOT EXISTS container_id TEXT
                        REFERENCES legal_containers(id) ON DELETE SET NULL;
                """)

                # ANN index on chunks
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                        ON chunks
                        USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100);
                    """)
                except Exception:
                    pass

                # Index for fast container → chunks lookup
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_container_id
                    ON chunks (container_id);
                """)

                # ----------------------------------------------------------
                # users
                # ----------------------------------------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id            SERIAL PRIMARY KEY,
                        username      TEXT UNIQUE NOT NULL,
                        email         TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at    TIMESTAMPTZ DEFAULT now()
                    );
                """)
                # Add email to existing installs that predate this column
                cur.execute("""
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;
                """)

                # ----------------------------------------------------------
                # chats  — one row per conversation (replaces user_sessions)
                # ----------------------------------------------------------
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        id         TEXT PRIMARY KEY,
                        user_id    INT  NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        title      TEXT NOT NULL DEFAULT 'New chat',
                        messages   JSONB NOT NULL DEFAULT '[]',
                        created_at TIMESTAMPTZ DEFAULT now(),
                        updated_at TIMESTAMPTZ DEFAULT now()
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chats_user_id
                    ON chats (user_id, updated_at DESC);
                """)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# legal_containers CRUD
# ---------------------------------------------------------------------------

def upsert_container(
    container_id: str,
    document: str,
    hierarchy_path: str,
    embedding_text: str,
    embedding: List[float],
    book: Optional[str] = None,
    part: Optional[str] = None,
    title: Optional[str] = None,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    sub_section: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or update a legal container row."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO legal_containers
                        (id, document, hierarchy_path, embedding_text,
                         book, part, title, chapter, section, sub_section,
                         embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document       = EXCLUDED.document,
                        hierarchy_path = EXCLUDED.hierarchy_path,
                        embedding_text = EXCLUDED.embedding_text,
                        book           = EXCLUDED.book,
                        part           = EXCLUDED.part,
                        title          = EXCLUDED.title,
                        chapter        = EXCLUDED.chapter,
                        section        = EXCLUDED.section,
                        sub_section    = EXCLUDED.sub_section,
                        embedding      = EXCLUDED.embedding,
                        metadata       = EXCLUDED.metadata;
                    """,
                    (
                        container_id, document, hierarchy_path, embedding_text,
                        book, part, title, chapter, section, sub_section,
                        Vector(embedding),
                        Json(metadata) if metadata is not None else None,
                    ),
                )
    finally:
        conn.close()


def search_containers(
    query_embedding: List[float],
    k: int = 5,
    document: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    ANN search over legal_containers.

    Optionally filter to a single document.
    Returns list of (container_dict, distance) sorted by ascending distance.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if document:
                    cur.execute(
                        """
                        SELECT id, document, hierarchy_path, embedding_text,
                               book, part, title, chapter, section, sub_section,
                               metadata,
                               embedding <-> %s AS distance
                        FROM legal_containers
                        WHERE document = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (Vector(query_embedding), document, k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, document, hierarchy_path, embedding_text,
                               book, part, title, chapter, section, sub_section,
                               metadata,
                               embedding <-> %s AS distance
                        FROM legal_containers
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (Vector(query_embedding), k),
                    )
                rows = cur.fetchall()
                results = []
                for row in rows:
                    cid, doc, hpath, emb_text, book, part, title, chap, sec, subsec, meta, dist = row
                    results.append((
                        {
                            "id": cid, "document": doc,
                            "hierarchy_path": hpath,
                            "embedding_text": emb_text,
                            "book": book, "part": part, "title": title,
                            "chapter": chap, "section": sec,
                            "sub_section": subsec, "metadata": meta,
                        },
                        dist,
                    ))
                return results
    finally:
        conn.close()


def get_chunks_for_container(
    container_id: str,
    k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch up to k article chunks that belong to a given container.
    Used in the second retrieval step.
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source, container_id, metadata
                    FROM chunks
                    WHERE container_id = %s
                    LIMIT %s;
                    """,
                    (container_id, k),
                )
                return [
                    {
                        "id": r[0], "content": r[1], "source": r[2],
                        "container_id": r[3], "metadata": r[4],
                    }
                    for r in cur.fetchall()
                ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# chunks CRUD
# ---------------------------------------------------------------------------

def upsert_chunk(
    chunk_id: str,
    content: str,
    source: Optional[str],
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None,
    container_id: Optional[str] = None,
) -> None:
    """Insert or update an article chunk, optionally linking to a container."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks
                        (id, content, source, container_id, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content      = EXCLUDED.content,
                        source       = EXCLUDED.source,
                        container_id = EXCLUDED.container_id,
                        metadata     = EXCLUDED.metadata,
                        embedding    = EXCLUDED.embedding;
                    """,
                    (
                        chunk_id, content, source, container_id,
                        Json(metadata) if metadata is not None else None,
                        Vector(embedding),
                    ),
                )
    finally:
        conn.close()


def search_chunks(
    query_embedding: List[float],
    k: int = 5,
    container_id: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    ANN search over chunks.

    Pass container_id to restrict search to a specific container
    (second retrieval step).
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if container_id:
                    cur.execute(
                        """
                        SELECT id, content, source, container_id, metadata,
                               embedding <-> %s AS distance
                        FROM chunks
                        WHERE container_id = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (Vector(query_embedding), container_id, k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, content, source, container_id, metadata,
                               embedding <-> %s AS distance
                        FROM chunks
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (Vector(query_embedding), k),
                    )
                rows = cur.fetchall()
                results = []
                for row in rows:
                    _id, content, source, cid, metadata, distance = row
                    results.append((
                        {
                            "id": _id, "content": content,
                            "source": source, "container_id": cid,
                            "metadata": metadata,
                        },
                        distance,
                    ))
                return results
    finally:
        conn.close()

def search_chunks_exact(
    query_embedding: List[float],
    k: int = 5,
    container_id: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    EXACT nearest-neighbor search over chunks.

    Forces PostgreSQL to scan the table instead of using
    the approximate vector index.
    """
    conn = get_conn()

    try:
        with conn:
            with conn.cursor() as cur:

                # Force a sequential scan for this query.
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")

                if container_id:
                    cur.execute(
                        """
                        SELECT id, content, source, container_id, metadata,
                               embedding <-> %s AS distance
                        FROM chunks
                        WHERE container_id = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (
                            Vector(query_embedding),
                            container_id,
                            k,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, content, source, container_id, metadata,
                               embedding <-> %s AS distance
                        FROM chunks
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (
                            Vector(query_embedding),
                            k,
                        ),
                    )

                rows = cur.fetchall()

                results = []

                for row in rows:
                    (
                        _id,
                        content,
                        source,
                        cid,
                        metadata,
                        distance,
                    ) = row

                    results.append(
                        (
                            {
                                "id": _id,
                                "content": content,
                                "source": source,
                                "container_id": cid,
                                "metadata": metadata,
                            },
                            distance,
                        )
                    )

                return results

    finally:
        conn.close()
        
def count_chunks() -> int:
    """Return total number of chunks in the DB."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(1) FROM chunks;")
                row = cur.fetchone()
                return int(row[0]) if row else 0
    finally:
        conn.close()


def count_containers() -> int:
    """Return total number of containers in the DB."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(1) FROM legal_containers;")
                row = cur.fetchone()
                return int(row[0]) if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def create_user(username: str, email: str, password_hash: str) -> int:
    """Insert a new user and return their id. Raises if username or email taken."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id;",
                    (username, email.lower(), password_hash),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Return {id, username, email, password_hash} or None."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, email, password_hash FROM users WHERE username = %s;",
                    (username,),
                )
                row = cur.fetchone()
                if row:
                    return {"id": row[0], "username": row[1], "email": row[2], "password_hash": row[3]}
                return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chats CRUD
# ---------------------------------------------------------------------------

def create_chat(user_id: int, title: str = "New chat") -> str:
    """Insert a new chat and return its id."""
    import uuid
    chat_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chats (id, user_id, title, messages)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (chat_id, user_id, title, Json([])),
                )
        return chat_id
    finally:
        conn.close()


def list_chats(user_id: int) -> List[Dict[str, Any]]:
    """Return all chats for a user ordered by most recently updated."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, updated_at
                    FROM chats
                    WHERE user_id = %s
                    ORDER BY updated_at DESC;
                    """,
                    (user_id,),
                )
                return [
                    {"id": r[0], "title": r[1], "updated_at": r[2]}
                    for r in cur.fetchall()
                ]
    finally:
        conn.close()


def load_chat_messages(chat_id: str) -> List[Dict[str, str]]:
    """Return the messages array for a chat."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM chats WHERE id = %s;",
                    (chat_id,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                msgs = row[0]
                if isinstance(msgs, str):
                    import json as _j
                    msgs = _j.loads(msgs)
                return msgs or []
    finally:
        conn.close()


def save_chat_messages(
    chat_id: str,
    messages: List[Dict[str, str]],
    title: Optional[str] = None,
) -> None:
    """Persist messages and optionally update the chat title."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if title is not None:
                    cur.execute(
                        """
                        UPDATE chats
                        SET messages = %s, title = %s, updated_at = now()
                        WHERE id = %s;
                        """,
                        (Json(messages), title, chat_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE chats
                        SET messages = %s, updated_at = now()
                        WHERE id = %s;
                        """,
                        (Json(messages), chat_id),
                    )
    finally:
        conn.close()


def delete_chat(chat_id: str) -> None:
    """Delete a chat and all its messages."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chats WHERE id = %s;", (chat_id,))
    finally:
        conn.close()
