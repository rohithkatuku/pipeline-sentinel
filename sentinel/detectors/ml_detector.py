"""ML-based anomaly detection using Isolation Forest."""

import numpy as np
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass

from sentinel.detectors.statistical import AnomalyResult


class IsolationForestDetector:
    """Multi-dimensional anomaly detection using Isolation Forest.

    Detects anomalies across correlated metrics (row_count + latency +
    null_pct together), catching patterns statistical detectors miss.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        self.is_fitted = False

    def fit(self, data: list[dict], feature_names: list[str] = None):
        """Train on historical run metrics.

        Args:
            data: List of dicts with metric values.
                  e.g. [{"row_count": 1000, "latency_ms": 200, "null_pct": 0.1}, ...]
            feature_names: Which keys to use as features.
        """
        if len(data) < 10:
            return

        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = list(data[0].keys())

        X = np.array([
            [record.get(f, 0) for f in self.feature_names]
            for record in data
        ])

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.model.fit(X)
        self.is_fitted = True

    def detect(self, current: dict, pipeline_id: str = "unknown") -> AnomalyResult:
        """Score a single data point against trained model.

        Args:
            current: Dict of current metric values.
            pipeline_id: Pipeline identifier for context.
        """
        if not self.is_fitted:
            return AnomalyResult(
                is_anomaly=False, metric_name="multi_metric",
                actual_value=0, expected_value=0,
                deviation_score=0.0, severity="low",
                method="isolation_forest",
                details="Model not fitted yet — need >= 10 historical data points."
            )

        X = np.array([[current.get(f, 0) for f in self.feature_names]])

        prediction = self.model.predict(X)[0]   # 1 = normal, -1 = anomaly
        score = self.model.decision_function(X)[0]  # lower = more anomalous

        is_anomaly = prediction == -1

        if is_anomaly:
            if score < -0.3:
                severity = "critical"
            elif score < -0.2:
                severity = "high"
            elif score < -0.1:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "low"

        feature_summary = ", ".join(
            f"{f}={current.get(f, 'N/A')}" for f in self.feature_names
        )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            metric_name="multi_metric",
            actual_value=round(score, 4),
            expected_value=0.0,
            deviation_score=round(abs(score), 4),
            severity=severity,
            method="isolation_forest",
            details=(
                f"Isolation Forest score: {score:.4f} "
                f"({'ANOMALY' if is_anomaly else 'Normal'}). "
                f"Features: [{feature_summary}]."
            )
        )

    def batch_detect(self, records: list[dict],
                     pipeline_id: str = "unknown") -> list[AnomalyResult]:
        return [self.detect(r, pipeline_id) for r in records]
