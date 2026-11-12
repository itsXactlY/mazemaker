#!/usr/bin/env python3
"""Supersession respects scope and trust, in recall and at write time.

recall(scope="curated") could answer with a transcript at rank 1: the supersedes traversal
that runs after the scope filter reads the NEWER row of every superseded result from the
store (a row no scope filter had seen), appends it, and elevates it to the old row's
relevance + 0.01. For a caller who asked for everything the same traversal demoted a curated
row by 0.5 and dropped it in favour of a transcript that merely sat close to it in meaning and
differed in a number. A transcript is evidence about the work; a decision:, a bug: or a skill:
is the conclusion, and must not be retired by one.

Run: python3 -m pytest tests/test_recall_scope_supersedes.py
"""
import sys
import tempfile
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))

from memory_client import Mazemaker  # noqa: E402

CURATED = "The podman quadlet units are symlinks into the repo; never replace one with a copy."
TURN = "User asked about the podman quadlet symlinks; the units are linked from the repo, never copied."
OTHER = "The podman quadlet directory holds symlinks on purpose, so a copy breaks the next update."
QUERY = "podman quadlet symlinks repo copy"
POOL_8 = "The dream worker connection pool is sized to 8 connections for the postgres store."
POOL_16 = "The dream worker connection pool is sized to 16 connections for the postgres store."


def _fresh() -> Mazemaker:
    d = tempfile.mkdtemp(prefix="mm-free-scope-")
    return Mazemaker(db_path=str(Path(d) / "m.db"), embedding_backend="hash",
                     use_cpp=False, retrieval_mode="semantic")


def _put(nm, text, label):
    return int(nm.remember(text, label=label, detect_conflicts=False, auto_connect=False,
                           detect_supersedes=False))


def _pair(old_label, new_label):
    nm = _fresh()
    old = _put(nm, CURATED, old_label)
    _put(nm, OTHER, "fact:podman-quadlet-dir")
    new = _put(nm, TURN, new_label)
    nm.store.add_connection(old, new, 0.8, edge_type="supersedes")     # source = the older row
    return nm, old, new


def _supersedes_edges(nm, ids):
    out = set()
    for i in ids:
        for c in nm.store.get_connections(i):
            if (c.get("edge_type") or c.get("type", "")) == "supersedes":
                out.add((int(c["source"]), int(c["target"])))
    return out


def test_who_may_retire_whom():
    allowed = Mazemaker._supersession_allowed
    assert allowed("decision:a", "decision:b")
    assert allowed("auto:turn:s:1", "auto:turn:s:2")
    assert allowed("auto:turn:s:1", "decision:b")
    assert not allowed("decision:a", "auto:turn:s:2")
    assert not allowed("skill:local:x", "auto:reasoning:s:2")


def test_scope_curated_never_returns_the_superseding_transcript():
    nm, old, new = _pair("invariant:podman-symlinks", "auto:turn:dd-test:6af00001")
    rows = nm.recall(QUERY, k=5, scope="curated")
    assert new not in [r["id"] for r in rows], rows
    assert all(not (r.get("label") or "").startswith("auto:") for r in rows)
    assert old in [r["id"] for r in rows]


def test_scope_glob_gets_the_same_treatment():
    nm, old, new = _pair("invariant:podman-symlinks", "auto:turn:dd-test:6af00001")
    assert [r["id"] for r in nm.recall(QUERY, k=5, scope="invariant:*")] == [old]


def test_unscoped_recall_keeps_a_curated_row_a_transcript_would_have_retired():
    nm, old, new = _pair("invariant:podman-symlinks", "auto:turn:dd-test:6af00001")
    by_id = {r["id"]: r for r in nm.recall(QUERY, k=5)}
    assert old in by_id, "the curated row was dropped for a transcript"
    assert not by_id[old].get("superseded_by")


def _swapped(nm, old, new) -> bool:
    """The community engine keeps a superseded row in the list, marked and demoted by 0.5,
    and puts its replacement above it."""
    rows = nm.recall(QUERY, k=5)
    ids = [r["id"] for r in rows]
    marked = next((r for r in rows if r["id"] == old), None)
    return (new in ids and marked is not None and marked.get("superseded_by") == new
            and ids.index(new) < ids.index(old))


def test_a_curated_replacement_still_retires_a_curated_row():
    nm, old, new = _pair("decision:podman-symlinks-v1", "decision:podman-symlinks-v2")
    assert _swapped(nm, old, new)


def test_a_transcript_replacement_still_retires_a_transcript():
    nm, old, new = _pair("auto:turn:dd-test:6af00001", "auto:turn:dd-test:6af00002")
    assert _swapped(nm, old, new)


def test_a_curated_replacement_retires_a_transcript():
    nm, old, new = _pair("auto:turn:dd-test:6af00001", "decision:podman-symlinks")
    assert _swapped(nm, old, new)


def test_write_time_a_transcript_does_not_link_itself_over_a_curated_row():
    nm = _fresh()
    curated = int(nm.remember(POOL_8, label="status:dream-pool", detect_conflicts=False, auto_connect=False))
    turn = int(nm.remember(POOL_16, label="auto:turn:dd-test:6af00003", detect_conflicts=False, auto_connect=False))
    assert (curated, turn) not in _supersedes_edges(nm, [curated, turn])


def test_write_time_a_curated_row_still_links_over_an_older_transcript():
    nm = _fresh()
    turn = int(nm.remember(POOL_8, label="auto:turn:dd-test:6af00003", detect_conflicts=False, auto_connect=False))
    curated = int(nm.remember(POOL_16, label="status:dream-pool", detect_conflicts=False, auto_connect=False))
    assert (turn, curated) in _supersedes_edges(nm, [turn, curated])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
