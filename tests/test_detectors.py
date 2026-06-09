"""Tests for anomaly detection modules."""

import pytest
import numpy as np

from sentinel.detectors.statistical import (
    ZScoreDetector, IQRDetector, FreshnessDetector, NullSpikeDetector
)
from sentinel.detectors.ml_detector import IsolationForestDetector
from sentinel.detectors.schema_drift import detect_drift, schema_hash


# --- Z-Score Detector ---

class TestZScoreDetector:
    def setup_method(self):
        self.detector = ZScoreDetector(threshold=3.0)
        self.normal_data = [100, 102, 98, 101, 99, 103, 97, 100, 101, 98]

    def test_normal_value_no_anomaly(self):
        result = self.detector.detect(self.normal_data, 101, "row_count")
        assert not result.is_anomaly
        assert result.severity == "low"

    def test_extreme_low_triggers_anomaly(self):
        result = self.detector.detect(self.normal_data, 20, "row_count")
        assert result.is_anomaly
        assert result.severity in ("high", "critical")

    def test_extreme_high_triggers_anomaly(self):
        result = self.detector.detect(self.normal_data, 200, "row_count")
        assert result.is_anomaly

    def test_insufficient_data(self):
        result = self.detector.detect([1, 2], 100, "row_count")
        assert not result.is_anomaly
        assert "Insufficient" in result.details

    def test_zero_variance(self):
        result = self.detector.detect([50, 50, 50, 50, 50], 50, "row_count")
        assert not result.is_anomaly

        result = self.detector.detect([50, 50, 50, 50, 50], 51, "row_count")
        assert result.is_anomaly


# --- IQR Detector ---

class TestIQRDetector:
    def setup_method(self):
        self.detector = IQRDetector(multiplier=1.5)
        self.normal_data = [100, 102, 98, 101, 99, 103, 97, 100, 101, 98]

    def test_normal_value(self):
        result = self.detector.detect(self.normal_data, 100, "row_count")
        assert not result.is_anomaly

    def test_outlier_detected(self):
        result = self.detector.detect(self.normal_data, 20, "row_count")
        assert result.is_anomaly

    def test_method_label(self):
        result = self.detector.detect(self.normal_data, 100, "test")
        assert result.method == "iqr"


# --- Freshness Detector ---

class TestFreshnessDetector:
    def test_fresh_data(self):
        detector = FreshnessDetector(max_delay_minutes=60)
        result = detector.detect(30)
        assert not result.is_anomaly
        assert result.severity == "low"

    def test_stale_data(self):
        detector = FreshnessDetector(max_delay_minutes=60)
        result = detector.detect(120)
        assert result.is_anomaly
        assert result.severity in ("medium", "high")

    def test_very_stale(self):
        detector = FreshnessDetector(max_delay_minutes=60)
        result = detector.detect(300)
        assert result.is_anomaly
        assert result.severity == "critical"


# --- Null Spike Detector ---

class TestNullSpikeDetector:
    def test_normal_nulls(self):
        detector = NullSpikeDetector(threshold_pct=5.0)
        history = [0.5, 0.3, 0.8, 0.4, 0.6, 0.5, 0.7]
        result = detector.detect(history, 0.6, "email")
        assert not result.is_anomaly

    def test_spike_detected(self):
        detector = NullSpikeDetector(threshold_pct=5.0)
        history = [0.5, 0.3, 0.8, 0.4, 0.6, 0.5, 0.7]
        result = detector.detect(history, 25.0, "email")
        assert result.is_anomaly


# --- Isolation Forest ---

class TestIsolationForest:
    def test_not_fitted(self):
        detector = IsolationForestDetector()
        result = detector.detect({"row_count": 100})
        assert not result.is_anomaly
        assert "not fitted" in result.details.lower()

    def test_fit_and_detect_normal(self):
        detector = IsolationForestDetector(contamination=0.1)
        data = [{"row_count": 100 + np.random.randint(-10, 10),
                 "latency": 200 + np.random.randint(-20, 20)} for _ in range(50)]
        detector.fit(data, ["row_count", "latency"])

        result = detector.detect({"row_count": 102, "latency": 195})
        assert result.method == "isolation_forest"

    def test_fit_and_detect_anomaly(self):
        np.random.seed(42)
        detector = IsolationForestDetector(contamination=0.05)
        data = [{"row_count": 100 + np.random.randint(-5, 5),
                 "latency": 200 + np.random.randint(-10, 10)} for _ in range(200)]
        detector.fit(data, ["row_count", "latency"])

        result = detector.detect({"row_count": -500, "latency": 50000})
        assert result.is_anomaly


# --- Schema Drift ---

class TestSchemaDrift:
    def test_no_drift(self):
        schema = [{"name": "id", "type": "int"}, {"name": "name", "type": "string"}]
        diff = detect_drift(schema, schema)
        assert not diff.has_drift

    def test_column_added(self):
        old = [{"name": "id", "type": "int"}]
        new = [{"name": "id", "type": "int"}, {"name": "email", "type": "string"}]
        diff = detect_drift(old, new)
        assert diff.has_drift
        assert "email" in diff.added_columns
        assert diff.severity == "medium"

    def test_column_removed(self):
        old = [{"name": "id", "type": "int"}, {"name": "email", "type": "string"}]
        new = [{"name": "id", "type": "int"}]
        diff = detect_drift(old, new)
        assert diff.has_drift
        assert "email" in diff.removed_columns
        assert diff.severity == "high"

    def test_type_change(self):
        old = [{"name": "id", "type": "int"}]
        new = [{"name": "id", "type": "string"}]
        diff = detect_drift(old, new)
        assert diff.has_drift
        assert len(diff.type_changes) == 1
        assert diff.severity == "high"

    def test_schema_hash_deterministic(self):
        schema = [{"name": "b", "type": "int"}, {"name": "a", "type": "string"}]
        h1 = schema_hash(schema)
        h2 = schema_hash(list(reversed(schema)))
        assert h1 == h2  # order independent
