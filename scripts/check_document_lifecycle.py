"""Real technical-document search plus explicitly synthetic update/delete/recovery exercises."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from ragkit import documents
from ragkit.evidence import write_report

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    ("Why should I avoid grouping experiments by rounds played?", "experiment-design.html"),
    ("Can the taxi model predict at booking without a route estimate?", "taxi-model.md"),
    ("How should I interpret coverage below the nominal interval target?", "pickup-forecast.txt"),
    ("Does ranking customer future spending prove retention uplift?", "customer-value.json"),
]


def main():
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        source = work / "source"
        shutil.copytree(ROOT / "examples" / "knowledge-base", source)
        database = work / "index.sqlite"
        initial = documents.sync(source, database, collection="portfolio-operations")
        noop = documents.sync(source, database, collection="portfolio-operations")
        results = []
        for query, expected in QUERIES:
            start = time.perf_counter()
            hits = documents.search(database, query, k=5)
            results.append({"query": query, "expected_source": expected,
                            "seconds": time.perf_counter() - start, "expected_in_top5_chunks": any(h["path"] == expected for h in hits),
                            "hits": [{key: hit[key] for key in ("document_id", "path", "source_sha256", "chunk_id", "bm25")} for hit in hits]})
        changed = source / "taxi-model.md"
        changed.write_text(changed.read_text(encoding="utf-8") + "\nSynthetic lifecycle marker: recoverydrillmarker.\n", encoding="utf-8")
        updated = documents.sync(source, database, collection="portfolio-operations")
        assert updated["updated"] == 1 and documents.search(database, "recoverydrillmarker")
        # Observe the previous committed snapshot from a separate reader before
        # injecting interruption into the writer's transaction.
        old_hits = documents.search(database, "recoverydrillmarker")
        changed.write_text("Replacement that must not publish", encoding="utf-8")
        def fail_before_commit(connection):
            assert documents.search(database, "recoverydrillmarker") == old_hits
            raise OSError("synthetic interruption before commit")
        try:
            documents.sync(source, database, collection="portfolio-operations", before_commit=fail_before_commit)
        except OSError:
            pass
        else:
            raise AssertionError("failure injection did not execute")
        assert documents.search(database, "recoverydrillmarker") == old_hits
        changed.unlink()
        deleted = documents.sync(source, database, collection="portfolio-operations")
        assert deleted["deleted"] == 1 and not documents.search(database, "recoverydrillmarker")
        report = {"schema_version": 1, "scope": "real maintained portfolio technical documents converted to four local formats",
                  "query_authorship": "AI-authored source-informed smoke queries; no independent human review or blind benchmark",
                  "initial": initial, "unchanged_scan": {k: noop[k] for k in ("updated", "unchanged", "deleted", "snapshot_sha256")},
                  "queries": results, "recovery_exercise": {"label": "synthetic local lifecycle exercise", "updated_documents": updated["updated"],
                      "concurrent_reader_preserved_old_snapshot": True, "interrupted_update_rolled_back": True,
                      "deleted_documents": deleted["deleted"], "deleted_text_absent": True},
                  "provenance": json.loads((ROOT / "examples" / "sources.json").read_text(encoding="utf-8")),
                  "runtime": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
                  "code_sha256": {"documents.py": hashlib.sha256(Path(documents.__file__).read_bytes()).hexdigest(),
                                  "check_document_lifecycle.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  "limits": ["Four documents; not a large messy production corpus or full dense/reranker evaluation.",
                             "Local UTF-8 text/Markdown/HTML/JSON only; PDF/OCR, authentication and remote crawling are not implemented.",
                             "SQLite transaction safety within one local index; no distributed durability or concurrent source-writer guarantee.",
                             "No LLM generation or external API calls; CPU/storage costs are not priced."]}
        write_report(ROOT / "results" / "document-lifecycle.json", report)
        print(json.dumps({"documents": initial["documents"], "queries_with_expected_source_in_top5_chunks": sum(r["expected_in_top5_chunks"] for r in results),
                          "query_count": len(results), "recovery_exercise": report["recovery_exercise"]}, indent=2))


if __name__ == "__main__":
    main()
