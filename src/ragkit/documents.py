"""Bounded local document ingestion and transactional incremental SQLite FTS5 search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from html.parser import HTMLParser
from pathlib import Path

FORMATS = {".txt", ".md", ".html", ".htm", ".json"}
PROFILE = {"version": 1, "chunk_characters": 1600, "overlap_characters": 200,
           "html": "text excluding script/style/noscript", "json": "one object with a text string",
           "retriever": "SQLite FTS5 unicode61 BM25; no embeddings or generated answers"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.suppressed = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.suppressed += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.suppressed = max(0, self.suppressed - 1)

    def handle_data(self, value):
        if not self.suppressed:
            self.parts.append(value)


def extract(raw: bytes, suffix: str) -> str:
    text = raw.decode("utf-8", errors="strict")
    if suffix in {".html", ".htm"}:
        parser = TextHTML()
        parser.feed(text)
        text = " ".join(parser.parts)
    elif suffix == ".json":
        obj = json.loads(text)
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
            raise ValueError("JSON documents require an object with a text string")
        text = obj["text"]
    text = " ".join(text.split())
    if not text:
        raise ValueError("document has no searchable text")
    return text


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,source_sha256 TEXT NOT NULL,
          text_sha256 TEXT NOT NULL,bytes INTEGER NOT NULL,format TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS chunks(rowid INTEGER PRIMARY KEY,chunk_id TEXT UNIQUE NOT NULL,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,ordinal INTEGER NOT NULL,text TEXT NOT NULL);
      CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text,content='chunks',content_rowid='rowid',tokenize='unicode61');
      CREATE TRIGGER IF NOT EXISTS chunks_insert AFTER INSERT ON chunks BEGIN
          INSERT INTO search(rowid,text) VALUES(new.rowid,new.text); END;
      CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
          INSERT INTO search(search,rowid,text) VALUES('delete',old.rowid,old.text); END;
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    """)
    return connection


def sync(root: Path, database: Path, *, collection="local", max_files=1000, max_file_bytes=2_000_000,
         max_total_bytes=20_000_000, max_entries=10000, allow_empty=False, before_commit=None):
    """Only a complete validated scan may replace the current collection state.

    One index represents one declared collection. The lexical postings update
    only for changed documents; global BM25 statistics follow SQLite's index.
    The optional callback is an explicit failure-injection seam for tests.
    """
    if not root.is_dir() or not collection or not isinstance(collection, str):
        raise ValueError("existing source directory and nonempty collection name required")
    for bound in (max_files, max_file_bytes, max_total_bytes, max_entries):
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise ValueError("ingestion bounds must be positive integers")
    root = root.resolve()
    source = []
    total = 0
    # Bound directory enumeration too, including unsupported files and directories.
    candidates, pending, visited = [], [root], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                visited += 1
                path = Path(entry.path)
                if visited > max_entries:
                    raise ValueError("directory entries exceed configured scan bound")
                if entry.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("symlinks and external paths are not allowed in a source collection")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in FORMATS:
                    candidates.append(path)
                    if len(candidates) > max_files:
                        raise ValueError("document count exceeds configured bound")
    for path in sorted(candidates):
        if len(source) >= max_files:
            raise ValueError("document count exceeds configured bound")
        if path.stat().st_size > max_file_bytes:
            raise ValueError("document exceeds configured byte bound")
        with path.open("rb") as stream:
            raw = stream.read(max_file_bytes + 1)
        total += len(raw)
        if len(raw) > max_file_bytes or total > max_total_bytes:
            raise ValueError("source collection exceeds configured byte bound")
        relative = path.relative_to(root).as_posix()
        text = extract(raw, path.suffix.lower())
        source.append({"id": digest((collection + "\0" + relative).encode()), "path": relative,
                       "source_sha256": digest(raw), "text_sha256": digest(text.encode()),
                       "bytes": len(raw), "format": path.suffix.lower(), "text": text})
    if not source and not allow_empty:
        raise ValueError("empty collection requires --allow-empty to confirm complete deletion")
    profile_hash = digest(json.dumps(PROFILE, sort_keys=True).encode())
    connection = connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        prior_meta = dict(connection.execute("SELECT key,value FROM metadata"))
        if prior_meta and (prior_meta.get("collection") != collection or prior_meta.get("root") != str(root)):
            raise ValueError("index belongs to a different collection/root; use a new index")
        previous = {row[0]: row[1] for row in connection.execute("SELECT id,source_sha256 FROM documents")}
        desired = {item["id"] for item in source}
        removed = sorted(set(previous) - desired)
        for doc in removed:
            connection.execute("DELETE FROM documents WHERE id=?", (doc,))
        updated = unchanged = 0
        for item in source:
            if previous.get(item["id"]) == item["source_sha256"] and prior_meta.get("profile_sha256") == profile_hash:
                unchanged += 1
                continue
            updated += 1
            connection.execute("DELETE FROM documents WHERE id=?", (item["id"],))
            connection.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)",
                               tuple(item[key] for key in ("id", "path", "source_sha256", "text_sha256", "bytes", "format")))
            step = PROFILE["chunk_characters"] - PROFILE["overlap_characters"]
            for ordinal, start in enumerate(range(0, len(item["text"]), step)):
                chunk = item["text"][start:start + PROFILE["chunk_characters"]]
                chunk_id = digest((item["id"] + f"\0{ordinal}\0" + chunk).encode())
                connection.execute("INSERT INTO chunks(chunk_id,document_id,ordinal,text) VALUES(?,?,?,?)",
                                   (chunk_id, item["id"], ordinal, chunk))
        identity = [{key: item[key] for key in ("id", "path", "source_sha256", "text_sha256")} for item in source]
        snapshot = digest(json.dumps({"profile": profile_hash, "documents": identity}, sort_keys=True).encode())
        for key, value in {"collection": collection, "root": str(root), "profile_sha256": profile_hash,
                           "snapshot_sha256": snapshot}.items():
            connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, value))
        # FTS integrity-check with rank=1 checks the external-content table too.
        connection.execute("INSERT INTO search(search,rank) VALUES('integrity-check',1)")
        if before_commit:
            before_commit(connection)
        connection.commit()
        return {"documents": len(source), "updated": updated, "unchanged": unchanged, "deleted": len(removed),
                "source_bytes": total, "snapshot_sha256": snapshot, "profile": PROFILE, "sources": identity}
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def search(database: Path, query: str, k=5):
    if not isinstance(query, str) or len(query) > 2000 or isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("query must be <=2000 characters and k between 1 and 100")
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return []
    expression = " OR ".join('"' + token + '"' for token in tokens)
    # A reader cannot accidentally create a new empty index.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("""SELECT d.id,d.path,d.source_sha256,c.chunk_id,c.ordinal,c.text,bm25(search)
            FROM search JOIN chunks c ON search.rowid=c.rowid JOIN documents d ON c.document_id=d.id
            WHERE search MATCH ? ORDER BY bm25(search),d.path,c.ordinal LIMIT ?""", (expression, k)).fetchall()
    return [dict(zip(("document_id", "path", "source_sha256", "chunk_id", "ordinal", "text", "bm25"), row)) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("sync")
    ingest.add_argument("root", type=Path)
    ingest.add_argument("database", type=Path)
    ingest.add_argument("--collection", default="local")
    ingest.add_argument("--allow-empty", action="store_true")
    query = sub.add_parser("search")
    query.add_argument("database", type=Path)
    query.add_argument("query")
    query.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.command == "sync":
        result = sync(args.root, args.database, collection=args.collection, allow_empty=args.allow_empty)
    else:
        result = {"hits": search(args.database, args.query, args.k)}
    print(json.dumps({**result, "seconds": time.perf_counter() - start}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
