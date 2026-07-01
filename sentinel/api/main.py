"""FastAPI application — Pipeline Sentinel API."""

import os

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from sentinel.models import database as db
from sentinel.detectors.statistical import ZScoreDetector, IQRDetector, NullSpikeDetector
from sentinel.detectors.ml_detector import IsolationForestDetector
from sentinel.detectors.schema_drift import detect_drift, schema_hash
from sentinel.analyzers import root_cause
from sentinel.lineage.tracker import LineageTracker
from sentinel.validators.quality import DataQualityValidator

app = FastAPI(
    title="Pipeline Sentinel",
    description="AI-powered data pipeline health monitor & root cause analyzer",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str = Header(default=None)):
    """Guard for state-changing / paid-LLM routes. Requires SENTINEL_API_KEY to be set."""
    expected = os.getenv("SENTINEL_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(401, "Missing or invalid X-API-Key")


# Initialize DB on startup
@app.on_event("startup")
async def startup():
    db.init_db()


# --- Models ---

class PipelineCreate(BaseModel):
    id: str
    name: str
    source: str
    destination: str
    schedule: str | None = None
    metadata: dict | None = None


class RunRecord(BaseModel):
    pipeline_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    row_count: int | None = None
    null_percentage: float | None = None
    schema_hash: str | None = None
    latency_ms: int | None = None
    error_message: str | None = None
    metrics: dict | None = None


class LineageEdge(BaseModel):
    source: str
    target: str
    relationship: str = "feeds"


class SchemaSnapshot(BaseModel):
    pipeline_id: str
    columns: list[dict]


class AnomalyCheckRequest(BaseModel):
    pipeline_id: str
    metric_name: str = "row_count"
    current_value: float | None = None


# --- Pipeline Endpoints ---

@app.post("/pipelines", tags=["Pipelines"], dependencies=[Depends(require_api_key)])
async def create_pipeline(p: PipelineCreate):
    try:
        db.create_pipeline(p.id, p.name, p.source, p.destination, p.schedule, p.metadata)
        return {"status": "created", "pipeline_id": p.id}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/pipelines", tags=["Pipelines"])
async def list_pipelines():
    return db.list_pipelines()


@app.get("/pipelines/{pipeline_id}", tags=["Pipelines"])
async def get_pipeline(pipeline_id: str):
    p = db.get_pipeline(pipeline_id)
    if not p:
        raise HTTPException(404, f"Pipeline {pipeline_id} not found")
    return p


# --- Run Endpoints ---

@app.post("/runs", tags=["Runs"], dependencies=[Depends(require_api_key)])
async def record_run(run: RunRecord):
    run_id = db.record_run(**run.model_dump())
    return {"status": "recorded", "run_id": run_id}


@app.get("/runs/{pipeline_id}", tags=["Runs"])
async def get_runs(pipeline_id: str, limit: int = 50):
    return db.get_recent_runs(pipeline_id, limit)


# --- Anomaly Detection ---

@app.post("/detect/anomaly", tags=["Detection"], dependencies=[Depends(require_api_key)])
async def check_anomaly(req: AnomalyCheckRequest):
    """Run anomaly detection on a pipeline's latest metrics."""
    runs = db.get_recent_runs(req.pipeline_id, 50)
    if not runs:
        raise HTTPException(404, f"No runs found for {req.pipeline_id}")

    metric_map = {
        "row_count": lambda r: r.get("row_count"),
        "latency_ms": lambda r: r.get("latency_ms"),
        "null_percentage": lambda r: r.get("null_percentage"),
    }

    extractor = metric_map.get(req.metric_name)
    if not extractor:
        raise HTTPException(400, f"Unknown metric: {req.metric_name}")

    historical = [extractor(r) for r in runs[1:] if extractor(r) is not None]
    current = req.current_value if req.current_value is not None else extractor(runs[0])

    if current is None:
        raise HTTPException(400, "No current value available")

    # Run both detectors
    zscore = ZScoreDetector(threshold=3.0)
    iqr = IQRDetector(multiplier=1.5)

    z_result = zscore.detect(historical, current, req.metric_name)
    iqr_result = iqr.detect(historical, current, req.metric_name)

    # Record if anomaly found
    results = []
    for result in [z_result, iqr_result]:
        if result.is_anomaly:
            anomaly_id = db.record_anomaly(
                pipeline_id=req.pipeline_id,
                anomaly_type=result.method,
                severity=result.severity,
                metric_name=result.metric_name,
                expected_value=result.expected_value,
                actual_value=result.actual_value,
                deviation_score=result.deviation_score,
                run_id=runs[0]["id"] if runs else None
            )
            result_dict = result.__dict__
            result_dict["anomaly_id"] = anomaly_id
            results.append(result_dict)
        else:
            results.append(result.__dict__)

    return {"pipeline_id": req.pipeline_id, "results": results}


# --- Root Cause Analysis ---

@app.post("/analyze/{anomaly_id}", tags=["Analysis"], dependencies=[Depends(require_api_key)])
async def analyze_anomaly(anomaly_id: int):
    """Run LLM root cause analysis on a detected anomaly."""
    anomaly = db.get_anomaly(anomaly_id)

    if not anomaly:
        raise HTTPException(404, f"Anomaly {anomaly_id} not found")

    from sentinel.detectors.statistical import AnomalyResult
    anomaly_result = AnomalyResult(
        is_anomaly=True,
        metric_name=anomaly["metric_name"],
        actual_value=anomaly["actual_value"] or 0,
        expected_value=anomaly["expected_value"] or 0,
        deviation_score=anomaly["deviation_score"] or 0,
        severity=anomaly["severity"],
        method=anomaly["anomaly_type"],
        details=f"Recorded anomaly #{anomaly_id}"
    )

    pipeline_info = db.get_pipeline(anomaly["pipeline_id"])
    recent_runs = db.get_recent_runs(anomaly["pipeline_id"], 5)

    tracker = LineageTracker()
    tracker.load_from_db()
    downstream = tracker.get_downstream(anomaly["pipeline_id"])

    analysis = root_cause.analyze(
        anomaly_result, pipeline_info, recent_runs, downstream
    )

    # Store results back
    db.resolve_anomaly(anomaly_id, analysis.diagnosis, str(analysis.remediation_steps))

    return {
        "anomaly_id": anomaly_id,
        "diagnosis": analysis.diagnosis,
        "probable_cause": analysis.probable_cause,
        "confidence": analysis.confidence,
        "remediation_steps": analysis.remediation_steps,
        "affected_systems": analysis.affected_systems,
    }


# --- Schema Drift ---

@app.post("/schema/check", tags=["Schema"], dependencies=[Depends(require_api_key)])
async def check_schema(snapshot: SchemaSnapshot):
    """Compare current schema against last known schema."""
    previous = db.get_latest_schema(snapshot.pipeline_id)

    if not previous:
        db.save_schema(snapshot.pipeline_id, snapshot.columns)
        return {"drift": False, "message": "First schema snapshot recorded."}

    diff = detect_drift(previous, snapshot.columns)
    db.save_schema(snapshot.pipeline_id, snapshot.columns)

    if diff.has_drift:
        db.record_anomaly(
            pipeline_id=snapshot.pipeline_id,
            anomaly_type="schema_drift",
            severity=diff.severity,
            metric_name="schema",
            expected_value=0,
            actual_value=1,
            deviation_score=1.0
        )

    return {
        "drift": diff.has_drift,
        "severity": diff.severity,
        "summary": diff.summary,
        "added": diff.added_columns,
        "removed": diff.removed_columns,
        "type_changes": diff.type_changes,
    }


# --- Lineage ---

@app.post("/lineage/edge", tags=["Lineage"], dependencies=[Depends(require_api_key)])
async def add_edge(edge: LineageEdge):
    db.add_lineage_edge(edge.source, edge.target, edge.relationship)
    return {"status": "added"}


@app.get("/lineage/impact/{pipeline_id}", tags=["Lineage"])
async def get_impact(pipeline_id: str):
    tracker = LineageTracker()
    tracker.load_from_db()
    report = tracker.get_impact_report(pipeline_id)
    return report.__dict__


@app.get("/lineage/graph", tags=["Lineage"])
async def get_graph():
    tracker = LineageTracker()
    tracker.load_from_db()
    return tracker.to_dict()


# --- Anomaly History ---

@app.get("/anomalies", tags=["Anomalies"])
async def list_anomalies(pipeline_id: str = None, resolved: bool = None, limit: int = 100):
    return db.get_anomalies(pipeline_id, resolved, limit)


# --- Health Dashboard ---

@app.get("/health", tags=["Health"])
async def health_summary():
    return db.get_pipeline_health_summary()


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Pipeline Sentinel",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }
