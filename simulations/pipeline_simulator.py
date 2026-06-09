"""Pipeline Simulator — generates synthetic pipeline data with injected failures.

Creates a realistic multi-pipeline environment with:
- Normal operation patterns
- Randomly injected failures (row count drops, null spikes, latency spikes, schema drift, stale data)
- Lineage relationships between pipelines
"""

import random
import hashlib
import json
from datetime import datetime, timedelta

from sentinel.models import database as db
from sentinel.detectors.statistical import ZScoreDetector, IQRDetector, NullSpikeDetector
from sentinel.detectors.ml_detector import IsolationForestDetector
from sentinel.detectors.schema_drift import detect_drift
from sentinel.analyzers import root_cause
from sentinel.lineage.tracker import LineageTracker

# --- Pipeline Definitions ---

PIPELINES = [
    {
        "id": "payments-ingestion",
        "name": "Payments Ingestion",
        "source": "Stripe API",
        "destination": "raw.payments",
        "schedule": "*/15 * * * *",
        "normal_rows": 5000,
        "normal_latency": 800,
        "normal_null_pct": 0.5,
    },
    {
        "id": "users-sync",
        "name": "User Profile Sync",
        "source": "Auth0 API",
        "destination": "raw.users",
        "schedule": "0 * * * *",
        "normal_rows": 200,
        "normal_latency": 400,
        "normal_null_pct": 1.0,
    },
    {
        "id": "orders-etl",
        "name": "Orders ETL",
        "source": "raw.payments + raw.users",
        "destination": "transformed.orders",
        "schedule": "*/30 * * * *",
        "normal_rows": 4500,
        "normal_latency": 1200,
        "normal_null_pct": 0.3,
    },
    {
        "id": "revenue-agg",
        "name": "Revenue Aggregation",
        "source": "transformed.orders",
        "destination": "analytics.revenue",
        "schedule": "0 * * * *",
        "normal_rows": 100,
        "normal_latency": 300,
        "normal_null_pct": 0.0,
    },
    {
        "id": "churn-model-features",
        "name": "Churn Model Features",
        "source": "transformed.orders + raw.users",
        "destination": "ml.churn_features",
        "schedule": "0 6 * * *",
        "normal_rows": 10000,
        "normal_latency": 3000,
        "normal_null_pct": 2.0,
    },
    {
        "id": "dashboard-refresh",
        "name": "Executive Dashboard Refresh",
        "source": "analytics.revenue",
        "destination": "dashboards.executive",
        "schedule": "0 8 * * *",
        "normal_rows": 50,
        "normal_latency": 200,
        "normal_null_pct": 0.0,
    },
    {
        "id": "email-reports",
        "name": "Daily Email Reports",
        "source": "analytics.revenue + ml.churn_features",
        "destination": "notifications.email",
        "schedule": "0 9 * * *",
        "normal_rows": 10,
        "normal_latency": 500,
        "normal_null_pct": 0.0,
    },
]

LINEAGE_EDGES = [
    ("payments-ingestion", "orders-etl"),
    ("users-sync", "orders-etl"),
    ("orders-etl", "revenue-agg"),
    ("orders-etl", "churn-model-features"),
    ("users-sync", "churn-model-features"),
    ("revenue-agg", "dashboard-refresh"),
    ("revenue-agg", "email-reports"),
    ("churn-model-features", "email-reports"),
]

FAILURE_TYPES = [
    "row_count_drop",
    "row_count_spike",
    "null_spike",
    "latency_spike",
    "complete_failure",
    "schema_drift",
]


def _jitter(base: float, pct: float = 0.1) -> float:
    """Add random noise to a value."""
    return base * (1 + random.uniform(-pct, pct))


def _generate_schema(pipeline_id: str, drifted: bool = False) -> list[dict]:
    """Generate schema for a pipeline, optionally with drift."""
    base_schemas = {
        "payments-ingestion": [
            {"name": "payment_id", "type": "string"},
            {"name": "amount", "type": "float"},
            {"name": "currency", "type": "string"},
            {"name": "customer_id", "type": "string"},
            {"name": "created_at", "type": "timestamp"},
            {"name": "status", "type": "string"},
        ],
        "users-sync": [
            {"name": "user_id", "type": "string"},
            {"name": "email", "type": "string"},
            {"name": "name", "type": "string"},
            {"name": "signup_date", "type": "timestamp"},
            {"name": "plan", "type": "string"},
        ],
    }

    schema = base_schemas.get(pipeline_id, [
        {"name": "id", "type": "integer"},
        {"name": "data", "type": "string"},
        {"name": "created_at", "type": "timestamp"},
    ])

    if drifted:
        drift_type = random.choice(["add", "remove", "type_change"])
        schema = [dict(c) for c in schema]  # deep copy
        if drift_type == "add":
            schema.append({"name": f"new_col_{random.randint(1,99)}", "type": "string"})
        elif drift_type == "remove" and len(schema) > 2:
            schema.pop(random.randint(1, len(schema) - 1))
        elif drift_type == "type_change":
            col = random.choice(schema)
            col["type"] = "string" if col["type"] != "string" else "integer"

    return schema


def simulate(num_runs: int = 100, failure_rate: float = 0.08, seed: int = 42):
    """Run full simulation.

    Args:
        num_runs: Number of historical runs per pipeline.
        failure_rate: Probability of failure injection per run.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    db.init_db()

    print("Setting up pipelines...")
    for p in PIPELINES:
        try:
            db.create_pipeline(
                p["id"], p["name"], p["source"], p["destination"],
                p["schedule"], {"normal_rows": p["normal_rows"]}
            )
        except Exception:
            pass  # already exists

    print("Setting up lineage...")
    for src, tgt in LINEAGE_EDGES:
        db.add_lineage_edge(src, tgt)

    print("Recording initial schemas...")
    for p in PIPELINES:
        db.save_schema(p["id"], _generate_schema(p["id"]))

    print(f"Generating {num_runs} runs per pipeline...")
    now = datetime.now()
    anomalies_injected = 0

    for p in PIPELINES:
        for i in range(num_runs):
            run_time = now - timedelta(hours=num_runs - i)
            inject_failure = random.random() < failure_rate

            if inject_failure:
                failure = random.choice(FAILURE_TYPES)
                anomalies_injected += 1
                run = _generate_failure_run(p, failure, run_time)
            else:
                run = _generate_normal_run(p, run_time)

            db.record_run(**run)

    print(f"Simulation complete. {anomalies_injected} failures injected.")
    print(f"Total runs: {len(PIPELINES) * num_runs}")

    # Run anomaly detection on latest runs
    print("\nRunning anomaly detection...")
    _run_detection()


def _generate_normal_run(p: dict, start_time: datetime) -> dict:
    latency = int(_jitter(p["normal_latency"], 0.15))
    return {
        "pipeline_id": p["id"],
        "status": "success",
        "started_at": start_time.isoformat(),
        "completed_at": (start_time + timedelta(milliseconds=latency)).isoformat(),
        "row_count": int(_jitter(p["normal_rows"], 0.1)),
        "null_percentage": round(_jitter(p["normal_null_pct"], 0.3), 2),
        "latency_ms": latency,
        "metrics": {"source_api_status": 200},
    }


def _generate_failure_run(p: dict, failure_type: str, start_time: datetime) -> dict:
    run = _generate_normal_run(p, start_time)

    if failure_type == "row_count_drop":
        run["row_count"] = int(p["normal_rows"] * random.uniform(0.05, 0.3))
        run["status"] = "warning"

    elif failure_type == "row_count_spike":
        run["row_count"] = int(p["normal_rows"] * random.uniform(3.0, 10.0))
        run["status"] = "warning"

    elif failure_type == "null_spike":
        run["null_percentage"] = round(random.uniform(15.0, 60.0), 2)
        run["status"] = "warning"

    elif failure_type == "latency_spike":
        run["latency_ms"] = int(p["normal_latency"] * random.uniform(5.0, 20.0))
        run["status"] = "warning"

    elif failure_type == "complete_failure":
        run["status"] = "failure"
        run["row_count"] = 0
        run["error_message"] = random.choice([
            "Connection refused: upstream API returned 503",
            "Timeout after 30s: database connection pool exhausted",
            "Authentication failed: expired API token",
            "OutOfMemoryError: GC overhead limit exceeded",
            "Table not found: source table was dropped",
        ])

    elif failure_type == "schema_drift":
        run["status"] = "warning"
        new_schema = _generate_schema(p["id"], drifted=True)
        run["schema_hash"] = hashlib.sha256(json.dumps(new_schema).encode()).hexdigest()[:16]

    return run


def _run_detection():
    """Run anomaly detection across all pipelines."""
    zscore = ZScoreDetector(threshold=2.5)

    detected = 0
    for p in PIPELINES:
        runs = db.get_recent_runs(p["id"], 50)
        if len(runs) < 10:
            continue

        row_counts = [r["row_count"] for r in runs[1:] if r["row_count"] is not None]
        current_rows = runs[0]["row_count"] if runs[0]["row_count"] is not None else 0

        result = zscore.detect(row_counts, current_rows, "row_count")
        if result.is_anomaly:
            db.record_anomaly(
                pipeline_id=p["id"],
                anomaly_type=result.method,
                severity=result.severity,
                metric_name="row_count",
                expected_value=result.expected_value,
                actual_value=result.actual_value,
                deviation_score=result.deviation_score,
                run_id=runs[0]["id"]
            )
            detected += 1
            print(f"  ANOMALY [{result.severity.upper()}] {p['name']}: {result.details}")

    print(f"\n{detected} anomalies detected across {len(PIPELINES)} pipelines.")


if __name__ == "__main__":
    simulate()
