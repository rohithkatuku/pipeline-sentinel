"""Statistical anomaly detectors: Z-score and IQR-based."""

import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    is_anomaly: bool
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_score: float
    severity: str  # low, medium, high, critical
    method: str
    details: str


def _classify_severity(deviation: float) -> str:
    abs_dev = abs(deviation)
    if abs_dev >= 4.0:
        return "critical"
    elif abs_dev >= 3.0:
        return "high"
    elif abs_dev >= 2.5:
        return "medium"
    return "low"


class ZScoreDetector:
    """Detect anomalies using Z-score (standard deviations from mean).

    Good for normally distributed metrics like row counts, latency.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect(self, values: list[float], current: float,
               metric_name: str = "metric") -> AnomalyResult:
        if len(values) < 5:
            return AnomalyResult(
                is_anomaly=False, metric_name=metric_name,
                actual_value=current, expected_value=current,
                deviation_score=0.0, severity="low",
                method="zscore", details="Insufficient history (need >= 5 data points)"
            )

        arr = np.array(values, dtype=float)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            is_anomaly = current != mean
            return AnomalyResult(
                is_anomaly=is_anomaly, metric_name=metric_name,
                actual_value=current, expected_value=mean,
                deviation_score=float('inf') if is_anomaly else 0.0,
                severity="high" if is_anomaly else "low",
                method="zscore",
                details=f"Zero variance in history. Value {'deviates' if is_anomaly else 'matches'}."
            )

        z_score = (current - mean) / std
        is_anomaly = abs(z_score) > self.threshold

        return AnomalyResult(
            is_anomaly=is_anomaly, metric_name=metric_name,
            actual_value=current, expected_value=round(mean, 2),
            deviation_score=round(z_score, 3),
            severity=_classify_severity(z_score) if is_anomaly else "low",
            method="zscore",
            details=(
                f"Z-score: {z_score:.3f} (threshold: ±{self.threshold}). "
                f"Mean: {mean:.2f}, Std: {std:.2f}. "
                f"{'ANOMALY' if is_anomaly else 'Normal'}."
            )
        )


class IQRDetector:
    """Detect anomalies using Interquartile Range.

    Robust to outliers. Good for skewed distributions.
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def detect(self, values: list[float], current: float,
               metric_name: str = "metric") -> AnomalyResult:
        if len(values) < 5:
            return AnomalyResult(
                is_anomaly=False, metric_name=metric_name,
                actual_value=current, expected_value=current,
                deviation_score=0.0, severity="low",
                method="iqr", details="Insufficient history (need >= 5 data points)"
            )

        arr = np.array(values, dtype=float)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        median = np.median(arr)

        lower = q1 - self.multiplier * iqr
        upper = q3 + self.multiplier * iqr

        is_anomaly = current < lower or current > upper

        if iqr > 0:
            deviation = (current - median) / iqr
        else:
            deviation = float('inf') if current != median else 0.0

        return AnomalyResult(
            is_anomaly=is_anomaly, metric_name=metric_name,
            actual_value=current, expected_value=round(median, 2),
            deviation_score=round(deviation, 3),
            severity=_classify_severity(deviation) if is_anomaly else "low",
            method="iqr",
            details=(
                f"IQR bounds: [{lower:.2f}, {upper:.2f}]. "
                f"Median: {median:.2f}, IQR: {iqr:.2f}. "
                f"{'ANOMALY' if is_anomaly else 'Normal'}."
            )
        )


class FreshnessDetector:
    """Detect stale data based on time since last successful run."""

    def __init__(self, max_delay_minutes: float = 60):
        self.max_delay = max_delay_minutes

    def detect(self, minutes_since_last: float,
               expected_interval: float = None) -> AnomalyResult:
        threshold = expected_interval or self.max_delay
        ratio = minutes_since_last / threshold if threshold > 0 else float('inf')
        is_anomaly = minutes_since_last > threshold

        return AnomalyResult(
            is_anomaly=is_anomaly, metric_name="data_freshness",
            actual_value=round(minutes_since_last, 1),
            expected_value=round(threshold, 1),
            deviation_score=round(ratio, 2),
            severity="critical" if ratio > 3 else "high" if ratio > 2 else "medium" if is_anomaly else "low",
            method="freshness",
            details=(
                f"Last run {minutes_since_last:.1f}min ago "
                f"(threshold: {threshold:.1f}min). "
                f"{'STALE' if is_anomaly else 'Fresh'}."
            )
        )


class NullSpikeDetector:
    """Detect unusual spikes in null/missing value percentages."""

    def __init__(self, threshold_pct: float = 5.0):
        self.threshold = threshold_pct

    def detect(self, historical_pcts: list[float], current_pct: float,
               column_name: str = "column") -> AnomalyResult:
        if len(historical_pcts) < 3:
            is_anomaly = current_pct > self.threshold
            return AnomalyResult(
                is_anomaly=is_anomaly,
                metric_name=f"null_pct_{column_name}",
                actual_value=current_pct, expected_value=self.threshold,
                deviation_score=current_pct / self.threshold if self.threshold > 0 else 0,
                severity="high" if is_anomaly else "low",
                method="null_spike",
                details=f"Null %: {current_pct:.1f}% (static threshold: {self.threshold}%)"
            )

        arr = np.array(historical_pcts)
        mean = np.mean(arr)
        std = np.std(arr)
        dynamic_threshold = mean + 2 * std if std > 0 else mean + self.threshold

        is_anomaly = current_pct > dynamic_threshold
        deviation = (current_pct - mean) / std if std > 0 else 0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            metric_name=f"null_pct_{column_name}",
            actual_value=round(current_pct, 2),
            expected_value=round(mean, 2),
            deviation_score=round(deviation, 3),
            severity=_classify_severity(deviation) if is_anomaly else "low",
            method="null_spike",
            details=(
                f"Null %: {current_pct:.1f}% vs avg {mean:.1f}% "
                f"(dynamic threshold: {dynamic_threshold:.1f}%). "
                f"{'SPIKE' if is_anomaly else 'Normal'}."
            )
        )
