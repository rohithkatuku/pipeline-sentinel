"""LLM-powered root cause analyzer using Anthropic Claude API."""

import os
import json
from dataclasses import dataclass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from sentinel.detectors.statistical import AnomalyResult


@dataclass
class RootCauseAnalysis:
    diagnosis: str
    probable_cause: str
    confidence: str  # high, medium, low
    remediation_steps: list[str]
    affected_systems: list[str]
    raw_response: str = ""


SYSTEM_PROMPT = """You are an expert data engineering diagnostician. Given anomaly data from a
pipeline monitoring system, provide:

1. **Diagnosis**: Plain-English summary of what went wrong.
2. **Probable Cause**: Most likely root cause based on the evidence.
3. **Confidence**: high/medium/low based on evidence strength.
4. **Remediation Steps**: Ordered list of fix actions.
5. **Affected Systems**: Which downstream systems/dashboards may be impacted.

Respond in JSON format:
{
    "diagnosis": "...",
    "probable_cause": "...",
    "confidence": "high|medium|low",
    "remediation_steps": ["step1", "step2"],
    "affected_systems": ["system1", "system2"]
}

Be specific and actionable. Reference actual metric values from the data."""


def build_context(anomaly: AnomalyResult, pipeline_info: dict = None,
                  recent_runs: list[dict] = None,
                  downstream: list[str] = None) -> str:
    """Build context string for LLM analysis."""
    parts = [
        "## Anomaly Alert",
        f"- Metric: {anomaly.metric_name}",
        f"- Method: {anomaly.method}",
        f"- Severity: {anomaly.severity}",
        f"- Expected: {anomaly.expected_value}",
        f"- Actual: {anomaly.actual_value}",
        f"- Deviation Score: {anomaly.deviation_score}",
        f"- Details: {anomaly.details}",
    ]

    if pipeline_info:
        parts.extend([
            "\n## Pipeline Info",
            f"- Name: {pipeline_info.get('name', 'unknown')}",
            f"- Source: {pipeline_info.get('source', 'unknown')}",
            f"- Destination: {pipeline_info.get('destination', 'unknown')}",
            f"- Schedule: {pipeline_info.get('schedule', 'unknown')}",
        ])

    if recent_runs:
        parts.append("\n## Recent Run History (last 5)")
        for run in recent_runs[:5]:
            parts.append(
                f"  - Status: {run['status']}, Rows: {run.get('row_count', 'N/A')}, "
                f"Latency: {run.get('latency_ms', 'N/A')}ms, "
                f"Nulls: {run.get('null_percentage', 'N/A')}%"
            )

    if downstream:
        parts.append(f"\n## Downstream Dependencies: {', '.join(downstream)}")

    return "\n".join(parts)


def analyze(anomaly: AnomalyResult, pipeline_info: dict = None,
            recent_runs: list[dict] = None,
            downstream: list[str] = None) -> RootCauseAnalysis:
    """Send anomaly context to Claude for root cause analysis.

    Falls back to rule-based analysis if API key not configured.
    """
    context = build_context(anomaly, pipeline_info, recent_runs, downstream)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not HAS_ANTHROPIC:
        return _fallback_analysis(anomaly, downstream)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}]
        )

        raw = message.content[0].text
        data = json.loads(raw)

        return RootCauseAnalysis(
            diagnosis=data.get("diagnosis", "Unable to determine"),
            probable_cause=data.get("probable_cause", "Unknown"),
            confidence=data.get("confidence", "low"),
            remediation_steps=data.get("remediation_steps", []),
            affected_systems=data.get("affected_systems", downstream or []),
            raw_response=raw
        )
    except Exception as e:
        fallback = _fallback_analysis(anomaly, downstream)
        fallback.diagnosis += f" (LLM call failed: {e})"
        return fallback


def _fallback_analysis(anomaly: AnomalyResult,
                       downstream: list[str] = None) -> RootCauseAnalysis:
    """Rule-based fallback when LLM unavailable."""

    rules = {
        "row_count": {
            "diagnosis": f"Row count anomaly: expected ~{anomaly.expected_value}, got {anomaly.actual_value}",
            "cause": "Upstream source may have incomplete extraction, API rate limiting, or filter change",
            "steps": [
                "Check upstream data source availability",
                "Verify API credentials and rate limits",
                "Check for recent query/filter changes",
                "Review source system logs for errors"
            ]
        },
        "data_freshness": {
            "diagnosis": f"Data is stale: {anomaly.actual_value}min since last update (threshold: {anomaly.expected_value}min)",
            "cause": "Pipeline execution may have failed, scheduler down, or upstream dependency blocked",
            "steps": [
                "Check pipeline scheduler (Airflow/cron) status",
                "Verify upstream dependencies completed",
                "Check for resource exhaustion (memory/disk)",
                "Review pipeline execution logs"
            ]
        },
        "null_pct": {
            "diagnosis": f"Null spike detected: {anomaly.actual_value}% vs normal ~{anomaly.expected_value}%",
            "cause": "Source schema change, ETL mapping error, or upstream data quality issue",
            "steps": [
                "Check source schema for recent changes",
                "Verify ETL column mappings",
                "Check for upstream processing errors",
                "Validate source data completeness"
            ]
        },
        "multi_metric": {
            "diagnosis": f"Multi-metric anomaly detected (score: {anomaly.actual_value})",
            "cause": "Correlated metric shift suggests systemic issue — infrastructure, deployment, or source change",
            "steps": [
                "Check for recent deployments or config changes",
                "Review infrastructure health (CPU, memory, disk)",
                "Check all upstream data sources",
                "Compare against previous anomaly patterns"
            ]
        }
    }

    # Match rule
    matched = None
    for key, rule in rules.items():
        if key in anomaly.metric_name or key == anomaly.method:
            matched = rule
            break

    if not matched:
        matched = {
            "diagnosis": f"Anomaly in {anomaly.metric_name}: {anomaly.details}",
            "cause": "Requires investigation — check source systems and recent changes",
            "steps": ["Review pipeline logs", "Check upstream sources", "Inspect recent changes"]
        }

    return RootCauseAnalysis(
        diagnosis=matched["diagnosis"],
        probable_cause=matched["cause"],
        confidence="medium",
        remediation_steps=matched["steps"],
        affected_systems=downstream or []
    )
