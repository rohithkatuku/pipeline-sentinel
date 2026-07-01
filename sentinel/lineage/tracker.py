"""Pipeline lineage tracker — DAG traversal for impact analysis."""

import networkx as nx
from dataclasses import dataclass, field

from sentinel.models import database as db


@dataclass
class ImpactReport:
    source_pipeline: str
    affected_pipelines: list[str]
    affected_count: int
    depth: int
    paths: list[list[str]]
    severity: str
    summary: str


class LineageTracker:
    """Build and query pipeline dependency graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def load_from_db(self):
        """Load lineage edges from database."""
        edges = db.get_all_edges()
        for edge in edges:
            self.graph.add_edge(
                edge["source_pipeline"],
                edge["target_pipeline"],
                relationship=edge.get("relationship", "feeds")
            )
        # Add any pipelines without edges
        for p in db.list_pipelines():
            if p["id"] not in self.graph:
                self.graph.add_node(p["id"])

    def add_edge(self, source: str, target: str, relationship: str = "feeds"):
        self.graph.add_edge(source, target, relationship=relationship)
        db.add_lineage_edge(source, target, relationship)

    def get_downstream(self, pipeline_id: str) -> list[str]:
        """All pipelines downstream of given pipeline (recursive)."""
        if pipeline_id not in self.graph:
            return []
        return list(nx.descendants(self.graph, pipeline_id))

    def get_upstream(self, pipeline_id: str) -> list[str]:
        """All pipelines upstream of given pipeline (recursive)."""
        if pipeline_id not in self.graph:
            return []
        return list(nx.ancestors(self.graph, pipeline_id))

    def get_impact_report(self, pipeline_id: str) -> ImpactReport:
        """Full blast radius analysis for a failing pipeline."""
        if pipeline_id not in self.graph:
            return ImpactReport(
                source_pipeline=pipeline_id,
                affected_pipelines=[], affected_count=0,
                depth=0, paths=[], severity="low",
                summary=f"Pipeline {pipeline_id} not found in lineage graph."
            )

        downstream = self.get_downstream(pipeline_id)

        # Find paths to affected nodes (bounded — all_simple_paths is exponential
        # on densely cross-linked DAGs, so cap depth and total paths collected)
        paths = []
        max_depth = 0
        for target in downstream:
            for path in nx.all_simple_paths(self.graph, pipeline_id, target, cutoff=10):
                paths.append(path)
                max_depth = max(max_depth, len(path) - 1)
                if len(paths) >= 100:
                    break
            if len(paths) >= 100:
                break

        # Severity based on blast radius
        count = len(downstream)
        if count >= 10:
            severity = "critical"
        elif count >= 5:
            severity = "high"
        elif count >= 2:
            severity = "medium"
        elif count >= 1:
            severity = "low"
        else:
            severity = "low"

        summary = (
            f"Pipeline '{pipeline_id}' failure affects {count} downstream pipeline(s) "
            f"across {max_depth} level(s)."
        ) if count > 0 else f"Pipeline '{pipeline_id}' has no downstream dependencies."

        return ImpactReport(
            source_pipeline=pipeline_id,
            affected_pipelines=sorted(downstream),
            affected_count=count,
            depth=max_depth,
            paths=paths,
            severity=severity,
            summary=summary
        )

    def get_root_causes(self, pipeline_id: str) -> list[str]:
        """Find root-level (source) pipelines upstream."""
        upstream = self.get_upstream(pipeline_id)
        roots = [u for u in upstream if self.graph.in_degree(u) == 0]
        return roots if roots else [pipeline_id]

    def to_dict(self) -> dict:
        """Export graph for visualization."""
        return {
            "nodes": [
                {"id": n, "downstream_count": len(self.get_downstream(n))}
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ]
        }
