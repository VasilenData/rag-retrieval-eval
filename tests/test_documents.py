import json
import sqlite3

import pytest

from ragkit.documents import search, sync


def corpus(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.md").write_text("# Operations\nAtomic publication protects the last valid dataset.", encoding="utf-8")
    (root / "two.html").write_text("<h1>Experiments</h1><script>secretleak</script><p>Randomized assignment and retention.</p>", encoding="utf-8")
    (root / "three.json").write_text(json.dumps({"text": "Model selection uses validation; future tests report accuracy."}), encoding="utf-8")
    (root / "four.txt").write_text("Monitor request latency and drift.", encoding="utf-8")
    return root, tmp_path / "index.sqlite"


def test_incremental_update_noop_and_delete_have_stable_source_identity(tmp_path):
    root, database = corpus(tmp_path)
    first = sync(root, database)
    assert first["updated"] == 4 and first["deleted"] == 0
    hit = search(database, "atomic")[0]
    assert hit["path"] == "one.md" and not search(database, "secretleak")
    assert sync(root, database)["unchanged"] == 4
    (root / "one.md").write_text("Atomic publication changed, with recoverysentinel.", encoding="utf-8")
    second = sync(root, database)
    updated = search(database, "recoverysentinel")[0]
    assert second["updated"] == 1 and second["unchanged"] == 3
    assert updated["document_id"] == hit["document_id"]
    assert updated["source_sha256"] != hit["source_sha256"] and updated["chunk_id"] != hit["chunk_id"]
    (root / "one.md").unlink()
    assert sync(root, database)["deleted"] == 1
    assert not search(database, "recoverysentinel")


def test_mid_transaction_failure_preserves_last_published_search_and_identity(tmp_path):
    root, database = corpus(tmp_path)
    before = sync(root, database)
    old_hits = search(database, "atomic")
    (root / "one.md").write_text("replacement sentinel", encoding="utf-8")
    (root / "four.txt").unlink()
    def interrupt(writer):
        # Independent reader observes the previous committed snapshot during a write.
        assert search(database, "atomic") == old_hits
        raise OSError("interruption after postings changed, before commit")
    with pytest.raises(OSError):
        sync(root, database, before_commit=interrupt)
    assert search(database, "atomic") == old_hits and not search(database, "sentinel")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM metadata WHERE key='snapshot_sha256'").fetchone()[0] == before["snapshot_sha256"]
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 4


@pytest.mark.parametrize("failure", ["invalid_json", "invalid_utf8", "oversize", "too_many", "total_bytes"])
def test_incomplete_or_invalid_scan_cannot_delete_previous_documents(tmp_path, failure):
    root, database = corpus(tmp_path)
    sync(root, database)
    expected = search(database, "atomic")
    kwargs = {}
    if failure == "invalid_json":
        (root / "bad.json").write_text('{"wrong":"shape"}')
    elif failure == "invalid_utf8":
        (root / "bad.txt").write_bytes(b"\xff\xfe")
    elif failure == "oversize":
        kwargs["max_file_bytes"] = 4
    elif failure == "too_many":
        kwargs["max_files"] = 2
    else:
        kwargs["max_total_bytes"] = 20
    with pytest.raises((ValueError, UnicodeDecodeError)):
        sync(root, database, **kwargs)
    assert search(database, "atomic") == expected


def test_empty_deletion_requires_explicit_intent_and_clears_postings(tmp_path):
    root, database = corpus(tmp_path)
    sync(root, database)
    for path in root.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="allow-empty"):
        sync(root, database)
    assert search(database, "atomic")
    assert sync(root, database, allow_empty=True)["deleted"] == 4
    assert search(database, "atomic") == []


def test_collection_mismatch_and_fts_syntax_are_not_silent(tmp_path):
    root, database = corpus(tmp_path)
    sync(root, database, collection="first")
    with pytest.raises(ValueError, match="different collection"):
        sync(root, database, collection="second")
    assert search(database, '" OR * : -- atomic')
    with pytest.raises(ValueError):
        search(database, "atomic", k=1000)
    with pytest.raises(sqlite3.OperationalError):
        search(tmp_path / "missing.sqlite", "atomic")
    assert not (tmp_path / "missing.sqlite").exists()


def test_scan_budget_counts_unsupported_files_too(tmp_path):
    root, database = corpus(tmp_path)
    sync(root, database)
    expected = search(database, "atomic")
    for i in range(4):
        (root / f"ignored-{i}.bin").write_bytes(b"unused")
    with pytest.raises(ValueError, match="scan bound"):
        sync(root, database, max_entries=5)
    assert search(database, "atomic") == expected


def test_search_closes_database_handles_on_success_and_sql_failure(tmp_path):
    root, database = corpus(tmp_path)
    sync(root, database)
    assert search(database, "atomic")
    # On Windows an open SQLite connection prevents deletion; this also ensures
    # the function does not leave handle cleanup to garbage collection.
    database.unlink()
    broken = tmp_path / "uninitialized.sqlite"
    connection = sqlite3.connect(broken)
    connection.close()
    with pytest.raises(sqlite3.OperationalError):
        search(broken, "atomic")
    broken.unlink()
