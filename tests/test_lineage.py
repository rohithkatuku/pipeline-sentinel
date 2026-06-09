"""Tests for lineage tracker and impact analysis."""

import pytest
import os

from sentinel.lineage.tracker import LineageTracker
from sentinel.models import database as db


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Use temp DB for each test."""
    db_path = str(tmp_path / "test.db")
    os.environ["SENTINEL_DB_PATH"] = db_path
    db.DB_PATH = db_path
    db.init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


class TestLineageTracker:
    def _build_graph(self):
        """Create test pipeline graph: A -> B -> D, A -> C -> D -> E"""
        tracker = LineageTracker()
        for pid in ["A", "B", "C", "D", "E"]:
            db.create_pipeline(pid, f"Pipeline {pid}", f"src_{pid}", f"dst_{pid}")

        tracker.add_edge("A", "B")
        tracker.add_edge("A", "C")
        tracker.add_edge("B", "D")
        tracker.add_edge("C", "D")
        tracker.add_edge("D", "E")
        return tracker

    def test_downstream(self):
        tracker = self._build_graph()
        downstream = tracker.get_downstream("A")
        assert set(downstream) == {"B", "C", "D", "E"}

    def test_upstream(self):
        tracker = self._build_graph()
        upstream = tracker.get_upstream("E")
        assert set(upstream) == {"A", "B", "C", "D"}

    def test_impact_report(self):
        tracker = self._build_graph()
        report = tracker.get_impact_report("A")
        assert report.affected_count == 4
        assert report.depth >= 2

    def test_impact_leaf_node(self):
        tracker = self._build_graph()
        report = tracker.get_impact_report("E")
        assert report.affected_count == 0

    def test_root_causes(self):
        tracker = self._build_graph()
        roots = tracker.get_root_causes("E")
        assert roots == ["A"]

    def test_unknown_pipeline(self):
        tracker = LineageTracker()
        report = tracker.get_impact_report("nonexistent")
        assert report.affected_count == 0

    def test_to_dict(self):
        tracker = self._build_graph()
        data = tracker.to_dict()
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 5

    def test_load_from_db(self):
        self._build_graph()
        tracker2 = LineageTracker()
        tracker2.load_from_db()
        downstream = tracker2.get_downstream("A")
        assert "E" in downstream
