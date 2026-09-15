#!/usr/bin/env python3
"""The community engine serves the recall and label-namespace contract the pod
calls on every tier.

The Mazemaker pod's tool layer is shared by the Pro and community images and
calls `recall(query, k=..., scope=..., rerank=...)`,
`recall_advanced(mode=...)`, and the three label-namespace methods. When the
community engine lacked `scope`, every recall on every Community and Builder
pod raised TypeError. These tests pin that contract on the community engine.

Run: python3 -m pytest tests/test_recall_scope_and_labels.py
  or python3 tests/test_recall_scope_and_labels.py
"""
import inspect
import sys
import tempfile
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))

from memory_client import Mazemaker  # noqa: E402
import mcp_schemas  # noqa: E402


def _engine(tmp: Path) -> Mazemaker:
    return Mazemaker(db_path=str(tmp / "memory.db"), embedding_backend="hash",
                     use_cpp=False, retrieval_mode="semantic")


def _seed(nm: Mazemaker) -> dict:
    ids = {}
    for label, text in [
        ("decision:embed-model", "We chose bge-m3 for embeddings because of multilingual recall."),
        ("bug:recall-scope", "Recall failed on community pods because scope was not accepted."),
        ("auto:turn:s1:t1:a", "Assistant said the embedding model is bge-m3 and recall works."),
        ("auto:turn:s1:t2:a", "Assistant repeated that bge-m3 embeddings power recall."),
        ("skill:alpha", "Skill alpha does embedding maintenance."),
        ("session::skill:beta", "Skill beta lives under a session infix label."),
    ]:
        ids[label] = nm.remember(text, label=label, detect_conflicts=False,
                                 auto_connect=False, detect_supersedes=False)
    return ids


def test_recall_accepts_the_pod_contract():
    params = inspect.signature(Mazemaker.recall).parameters
    for name in ("scope", "mode", "rerank", "k"):
        assert name in params, f"recall() must accept {name}="


def test_scope_none_curated_auto_and_glob():
    with tempfile.TemporaryDirectory() as d:
        nm = _engine(Path(d))
        _seed(nm)
        q = "bge-m3 embeddings recall"

        # The exact call the pod makes, with no scope given.
        everything = nm.recall(q, k=10, scope=None, rerank=None)
        assert everything, "unscoped recall must return hits"

        curated = nm.recall(q, k=10, scope="curated")
        assert curated
        assert all(r["label"].split(":")[0] in ("decision", "bug") for r in curated), curated

        auto = nm.recall(q, k=10, scope="auto")
        assert auto
        assert all(r["label"].startswith("auto:") for r in auto), auto

        glob = nm.recall(q, k=10, scope="decision:*")
        assert [r["label"] for r in glob] == ["decision:embed-model"], glob

        assert nm.recall(q, k=10, scope="nothing-matches:*") == []


def test_mode_is_honoured_per_call():
    with tempfile.TemporaryDirectory() as d:
        nm = _engine(Path(d))
        _seed(nm)
        assert nm.recall("bge-m3", k=5, mode="semantic")
        assert nm.recall("bge-m3", k=5, mode="hybrid")


def test_label_namespace_methods():
    with tempfile.TemporaryDirectory() as d:
        nm = _engine(Path(d))
        ids = _seed(nm)

        assert nm.count_by_label_prefix("skill:") == 2          # prefix + ::infix
        page = nm.list_by_label_prefix("skill:", offset=0, limit=1)
        assert len(page) == 1 and set(page[0]) == {"id", "label"}
        assert len(nm.list_by_label_prefix("skill:", offset=1, limit=5)) == 1
        assert nm.count_by_label_prefix("") == 0

        try:
            nm.delete_memories_by_labels(["skill:alpha"])
            raise AssertionError("delete without confirm must refuse")
        except ValueError:
            pass

        a, b = ids["skill:alpha"], ids["decision:embed-model"]
        nm.store.conn.execute(
            "INSERT INTO connections (source_id, target_id, weight) VALUES (?, ?, 0.9)", (a, b))
        nm.store.conn.execute(
            "INSERT INTO memory_revisions (memory_id, old_content, new_content) VALUES (?, 'x', 'y')", (a,))
        nm.store.conn.commit()

        assert nm.delete_memories_by_labels(["skill:alpha"], confirm=True) == 1
        assert nm.count_by_label_prefix("skill:") == 1
        conn = nm.store.conn
        assert conn.execute(
            "SELECT count(*) FROM connections WHERE source_id = ? OR target_id = ?", (a, a)
        ).fetchone()[0] == 0, "edges to a deleted memory must go with it"
        assert conn.execute(
            "SELECT count(*) FROM memory_revisions WHERE memory_id = ?", (a,)
        ).fetchone()[0] == 0


def test_schemas_advertise_label_tools():
    names = {s["name"] for s in mcp_schemas.ALL_TOOL_SCHEMAS}
    assert {"mazemaker_count_by_label_prefix", "mazemaker_list_by_label_prefix",
            "mazemaker_delete_by_labels"} <= names


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ✗ {name} — {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
