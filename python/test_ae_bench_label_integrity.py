"""Ground-truth integrity for benchmarks/ae_domain_memory_bench/queries.py.

Locks two contracts:

  1. Every ground_truth_ids list contains only memory IDs that ACTUALLY
     exist in the substrate with non-empty content + present embedding.
     If a GT ID is silently deleted or never re-embedded, this test
     surfaces the drift before the next bench run inflates as a false MISS.

  2. The bench scored-query count never silently shrinks. Today's count is
     the floor; future expansions push it up. A drop means a label was
     accidentally cleared or a query removed.

The test SKIPS gracefully when the substrate file is absent (clean CI env).
This is the only way to keep the integrity check in the test suite without
forcing every dev env to materialize a 4GB substrate.
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "benchmarks" / "ae_domain_memory_bench"))

from queries import ALL_QUERIES  # noqa: E402

_SUBSTRATE_PATH = Path.home() / ".neural_memory" / "memory.db"

# Floor: queries with non-empty ground_truth_ids as of HEAD when this test
# was added. Any future label expansion increments this; any decrement
# means a label was lost.
#
# 2026-05-03 (S6a + S1e): floor lowered 57 -> 54 after temporal quarantine.
# TMP-011 (was GT=[5531]), TMP-026 (was GT=[264, 280]), and TMP-033 (was
# GT=[268, 282]) were moved to category="quarantined_temporal" with empty
# ground_truth_ids. Validation against canonical DB showed all five GT ids
# resolved to WRONG_CONTENT (5531 = Amperage Q1 invoice table, not a contact;
# 264/280 = "Sarah from Lennar called." — current-tense, no predecessor;
# 268/282 = "Sarah is the Lennar contact this week." — current assertion,
# not a change event). Quarantine rationale lives in queries.py. The floor
# decrement is a deliberate, audited removal of bad labels.
#
# 2026-05-03 (T13): floor lowered 54 -> 49 after S6-DIAG label_error sweep.
# Five non-temporal queries had GT semantically unrelated to query intent
# (independently re-verified via direct sqlite3 read of canonical DB).
# Each entry now carries category="quarantined_<original_cat>" with empty
# ground_truth_ids. WRONG_CONTENT in every case:
#   ELC-040 (was GT=[274, 286]):
#       GT = "Lennar lot 27 needs panel labels." (33 chars, toy seed) — zero
#       overlap with "permit / inspection / rework" tokens.
#   MAT-004 (was GT=[5961]):
#       GT = "[Service install] 2-inch grounding bushings" doctrine fragment
#       (231 chars, single-item) — not a panel-upgrade BOM / materials list.
#   FIN-002 (was GT=[2628, 2659]):
#       GT = "V.5 vs actual = OVER-BUY pattern" doctrine, explicitly stating
#       over-buy is a DELIBERATE operational choice ("NOT a bug") — opposite
#       of "OVERRUN" (cost surprise) intent the query asks about.
#   LOT-008 (was GT=[5531]):
#       GT = "[Recent Amperage invoices by lot (Q1 2026)]" pure invoice
#       table (lot/invoice#/date/amount) — no delivery-delay event.
#   SPA-010 (was GT=[6700, 7377, 10202, 12695, 15092]):
#       GT = 5x duplicates of bridge_mailbox WA escalation
#       "falta el breaker de 20 amperes" (intent=materials_missing) —
#       semantically OPPOSITE of query "comprar breakers" (BUYING intent).
# Each quarantine rationale lives inline in queries.py next to its query
# definition. Floor decrement is a deliberate, audited removal of bad
# labels — not silent label drift.
_SCORED_QUERY_FLOOR = 49

# Cap: number of pairs of scored queries that may share an identical GT-set.
# These collisions exist by design — the same memory can legitimately be the
# answer to two queries asked from different lenses (e.g. a Lennar lot query
# and a customer-temporal query about that lot's contact). The cap is a
# CEILING: any new collision beyond this count means lazy labeling (someone
# reused an existing GT-set instead of designing a query with disjoint
# evidence). Raise this only with documented justification per new pair.
_DUPLICATE_GT_SET_PAIR_CAP = 18


class BenchLabelIntegrityTests(unittest.TestCase):
    def test_scored_query_count_meets_floor(self) -> None:
        """Total queries with ground_truth_ids never silently shrinks."""
        scored = [q for q in ALL_QUERIES if q["ground_truth_ids"]]
        self.assertGreaterEqual(
            len(scored), _SCORED_QUERY_FLOOR,
            f"Scored query count dropped below floor "
            f"({_SCORED_QUERY_FLOOR}). Found {len(scored)}. "
            f"A label was likely cleared or a query removed; verify before "
            f"lowering the floor.",
        )

    def test_no_query_id_is_duplicated(self) -> None:
        """Every query id is unique — no accidental copy/paste."""
        ids = [q["id"] for q in ALL_QUERIES]
        dupes = {qid for qid in ids if ids.count(qid) > 1}
        self.assertFalse(
            dupes, f"Duplicate query ids detected: {sorted(dupes)}",
        )

    def test_duplicate_ground_truth_set_pair_count_under_cap(self) -> None:
        """No new GT-set collisions beyond the documented design cap.

        Two queries sharing an identical sorted ground_truth_ids tuple is a
        smell: it usually means the second query was lazily labeled with the
        first query's GT instead of being designed against disjoint evidence.

        Some collisions are legitimate (cross-category lenses on the same
        memory). The cap captures the count at the time of the S6 expansion;
        new collisions must either retire an old one or push the cap up with
        a justifying comment.
        """
        scored = [q for q in ALL_QUERIES if q["ground_truth_ids"]]
        seen: dict[tuple[int, ...], str] = {}
        collisions: list[tuple[str, str, tuple[int, ...]]] = []
        for q in scored:
            key = tuple(sorted(q["ground_truth_ids"]))
            if key in seen:
                collisions.append((seen[key], q["id"], key))
            else:
                seen[key] = q["id"]
        self.assertLessEqual(
            len(collisions),
            _DUPLICATE_GT_SET_PAIR_CAP,
            f"Duplicate GT-set pair count {len(collisions)} exceeds cap "
            f"{_DUPLICATE_GT_SET_PAIR_CAP}. New collisions: "
            f"{collisions[_DUPLICATE_GT_SET_PAIR_CAP:]}. Either design "
            f"queries against disjoint memory evidence or raise the cap "
            f"with a documented justification.",
        )

    def test_quarantined_queries_excluded_from_scoring(self) -> None:
        """Quarantined queries must have empty GT and a quarantined_* category.

        Quarantine rationale (per S6a + T13 — 2026-05-03):
          ELC-040: was GT=[274,286] — "Lennar lot 27 needs panel labels."
                   no overlap with "permit / inspection / rework".
          MAT-004: was GT=[5961] — 2-inch grounding bushings doctrine,
                   not a panel-upgrade BOM.
          FIN-002: was GT=[2628,2659] — OVER-BUY doctrine ("deliberate, NOT
                   a bug"), opposite of OVERRUN (cost surprise).
          LOT-008: was GT=[5531] — Q1 Amperage invoice table, no
                   delivery-delay event.
          SPA-010: was GT=[6700,7377,10202,12695,15092] — 5x dupes of
                   "falta el breaker" (MISSING), opposite of BUYING.
          TMP-011: was GT=[5531] — Q1 Amperage invoice table, not contact.
          TMP-026: was GT=[264,280] — "Sarah from Lennar called.", no
                   predecessor info.
          TMP-033: was GT=[268,282] — "Sarah is the Lennar contact this
                   week.", current assertion, not change-event.

        Two invariants:
          1. Each quarantined query MUST have empty ground_truth_ids
             (otherwise its mis-labeled GT contributes to scoring again).
          2. Each quarantined query MUST carry a quarantined_* category
             (otherwise per-category R@5 aggregates pull it back in).
        """
        expected_quarantined = {
            "ELC-040": "quarantined_electrical",
            "MAT-004": "quarantined_materials",
            "FIN-002": "quarantined_financial",
            "LOT-008": "quarantined_lots",
            "SPA-010": "quarantined_spanish",
            "TMP-011": "quarantined_temporal",
            "TMP-026": "quarantined_temporal",
            "TMP-033": "quarantined_temporal",
        }
        by_id = {q["id"]: q for q in ALL_QUERIES}
        for qid, expected_cat in expected_quarantined.items():
            with self.subTest(qid=qid):
                q = by_id.get(qid)
                self.assertIsNotNone(q, f"{qid} missing from ALL_QUERIES")
                self.assertEqual(
                    q["ground_truth_ids"], [],
                    f"{qid} must have empty ground_truth_ids "
                    f"(was quarantined as label_error / WRONG_CONTENT). "
                    f"Found {q['ground_truth_ids']}.",
                )
                self.assertEqual(
                    q["category"], expected_cat,
                    f"{qid} category must be {expected_cat!r} "
                    f"(quarantined_<original>). Found {q['category']!r}.",
                )

        # Ensure quarantined queries are excluded from the scored set.
        scored_ids = {q["id"] for q in ALL_QUERIES if q["ground_truth_ids"]}
        leaked = set(expected_quarantined) & scored_ids
        self.assertFalse(
            leaked,
            f"Quarantined queries leaked back into the scored set: "
            f"{sorted(leaked)}. ground_truth_ids must remain empty.",
        )

    def test_every_ground_truth_id_exists_in_substrate(self) -> None:
        """Every GT memory id must exist in substrate with non-empty content
        and a present embedding. SKIPS if substrate file is absent."""
        if not _SUBSTRATE_PATH.exists():
            self.skipTest(
                f"Substrate not present at {_SUBSTRATE_PATH}; "
                f"GT integrity check requires live DB."
            )

        all_gt_ids: set[int] = set()
        for q in ALL_QUERIES:
            all_gt_ids.update(q["ground_truth_ids"])
        if not all_gt_ids:
            self.skipTest("No ground_truth_ids in any query — nothing to check.")

        conn = sqlite3.connect(
            f"file:{_SUBSTRATE_PATH}?mode=ro", uri=True, timeout=5,
        )
        try:
            id_list = ",".join(str(i) for i in sorted(all_gt_ids))
            rows = conn.execute(
                f"SELECT id, LENGTH(content), LENGTH(embedding) "
                f"FROM memories WHERE id IN ({id_list})"
            ).fetchall()
        finally:
            conn.close()

        present_ids = {row[0] for row in rows}
        missing = all_gt_ids - present_ids
        self.assertFalse(
            missing,
            f"Ground-truth memory IDs missing from substrate: {sorted(missing)}. "
            f"Every GT ID in queries.py must point to an existing memory.",
        )

        empty_content = [row[0] for row in rows if not row[1]]
        self.assertFalse(
            empty_content,
            f"Ground-truth memories with empty content: {sorted(empty_content)}. "
            f"Cross-encoder rerank cannot score empty content.",
        )

        no_embedding = [row[0] for row in rows if not row[2]]
        self.assertFalse(
            no_embedding,
            f"Ground-truth memories without embedding: {sorted(no_embedding)}. "
            f"Dense retrieval cannot surface these without an embedding.",
        )


if __name__ == "__main__":
    unittest.main()
