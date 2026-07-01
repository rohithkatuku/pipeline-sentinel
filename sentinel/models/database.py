"""SQLite database layer for Pipeline Sentinel metadata store."""

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pipelines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                destination TEXT NOT NULL,
                schedule TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'failure', 'warning', 'running')),
                started_at TEXT NOT NULL,
                completed_at TEXT,
                row_count INTEGER,
                null_percentage REAL,
                schema_hash TEXT,
                latency_ms INTEGER,
                error_message TEXT,
                metrics TEXT DEFAULT '{}',
                FOREIGN KEY (pipeline_id) REFERENCES pipelines(id)
            );

            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                run_id INTEGER,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                metric_name TEXT NOT NULL,
                expected_value REAL,
                actual_value REAL,
                deviation_score REAL,
                detected_at TEXT DEFAULT (datetime('now')),
                resolved BOOLEAN DEFAULT 0,
                root_cause TEXT,
                remediation TEXT,
                FOREIGN KEY (pipeline_id) REFERENCES pipelines(id),
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
            );

            CREATE TABLE IF NOT EXISTS lineage_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_pipeline TEXT NOT NULL,
                target_pipeline TEXT NOT NULL,
                relationship TEXT DEFAULT 'feeds',
                FOREIGN KEY (source_pipeline) REFERENCES pipelines(id),
                FOREIGN KEY (target_pipeline) REFERENCES pipelines(id),
                UNIQUE(source_pipeline, target_pipeline)
            );

            CREATE TABLE IF NOT EXISTS schema_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id TEXT NOT NULL,
                columns TEXT NOT NULL,
                captured_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (pipeline_id) REFERENCES pipelines(id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_pipeline ON pipeline_runs(pipeline_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON pipeline_runs(status);
            CREATE INDEX IF NOT EXISTS idx_anomalies_pipeline ON anomalies(pipeline_id);
            CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
        """)


# --- Pipeline CRUD ---

def create_pipeline(pipeline_id: str, name: str, source: str, destination: str,
                    schedule: str = None, metadata: dict = None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO pipelines (id, name, source, destination, schedule, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (pipeline_id, name, source, destination, schedule, json.dumps(metadata or {}))
        )


def get_pipeline(pipeline_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM pipelines WHERE id = ?", (pipeline_id,)).fetchone()
        return dict(row) if row else None


def list_pipelines() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM pipelines ORDER BY name").fetchall()
        return [dict(r) for r in rows]


# --- Run CRUD ---

def record_run(pipeline_id: str, status: str, started_at: str,
               completed_at: str = None, row_count: int = None,
               null_percentage: float = None, schema_hash: str = None,
               latency_ms: int = None, error_message: str = None,
               metrics: dict = None) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO pipeline_runs
               (pipeline_id, status, started_at, completed_at, row_count,
                null_percentage, schema_hash, latency_ms, error_message, metrics)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pipeline_id, status, started_at, completed_at, row_count,
             null_percentage, schema_hash, latency_ms, error_message,
             json.dumps(metrics or {}))
        )
        return cursor.lastrowid


def get_recent_runs(pipeline_id: str, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM pipeline_runs WHERE pipeline_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            (pipeline_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


# --- Anomaly CRUD ---

def record_anomaly(pipeline_id: str, anomaly_type: str, severity: str,
                   metric_name: str, expected_value: float = None,
                   actual_value: float = None, deviation_score: float = None,
                   run_id: int = None, root_cause: str = None,
                   remediation: str = None) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO anomalies
               (pipeline_id, run_id, anomaly_type, severity, metric_name,
                expected_value, actual_value, deviation_score, root_cause, remediation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pipeline_id, run_id, anomaly_type, severity, metric_name,
             expected_value, actual_value, deviation_score, root_cause, remediation)
        )
        return cursor.lastrowid


def get_anomalies(pipeline_id: str = None, resolved: bool = None,
                  limit: int = 100) -> list[dict]:
    with get_connection() as conn:
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        if pipeline_id:
            query += " AND pipeline_id = ?"
            params.append(pipeline_id)
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(int(resolved))
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_anomaly(anomaly_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM anomalies WHERE id = ?", (anomaly_id,)).fetchone()
        return dict(row) if row else None


def resolve_anomaly(anomaly_id: int, root_cause: str = None, remediation: str = None):
    with get_connection() as conn:
        conn.execute(
            "UPDATE anomalies SET resolved = 1, root_cause = ?, remediation = ? WHERE id = ?",
            (root_cause, remediation, anomaly_id)
        )


# --- Lineage ---

def add_lineage_edge(source: str, target: str, relationship: str = "feeds"):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO lineage_edges (source_pipeline, target_pipeline, relationship) VALUES (?, ?, ?)",
            (source, target, relationship)
        )


def get_downstream(pipeline_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM lineage_edges WHERE source_pipeline = ?", (pipeline_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_upstream(pipeline_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM lineage_edges WHERE target_pipeline = ?", (pipeline_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_edges() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM lineage_edges").fetchall()
        return [dict(r) for r in rows]


# --- Schema Snapshots ---

def save_schema(pipeline_id: str, columns: list[dict]):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO schema_snapshots (pipeline_id, columns) VALUES (?, ?)",
            (pipeline_id, json.dumps(columns))
        )


def get_latest_schema(pipeline_id: str) -> list[dict] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT columns FROM schema_snapshots WHERE pipeline_id = ? ORDER BY captured_at DESC LIMIT 1",
            (pipeline_id,)
        ).fetchone()
        return json.loads(row["columns"]) if row else None


# --- Stats ---

def get_pipeline_health_summary() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                p.id, p.name,
                COUNT(CASE WHEN pr.status = 'success' THEN 1 END) as success_count,
                COUNT(CASE WHEN pr.status = 'failure' THEN 1 END) as failure_count,
                COUNT(CASE WHEN pr.status = 'warning' THEN 1 END) as warning_count,
                COUNT(a.id) as anomaly_count,
                AVG(pr.latency_ms) as avg_latency_ms,
                MAX(pr.started_at) as last_run
            FROM pipelines p
            LEFT JOIN pipeline_runs pr ON p.id = pr.pipeline_id
            LEFT JOIN anomalies a ON p.id = a.pipeline_id AND a.resolved = 0
            GROUP BY p.id, p.name
        """).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
