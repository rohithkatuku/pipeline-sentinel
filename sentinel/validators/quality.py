"""Data quality validation checks — Great Expectations-style."""

from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    details: str
    severity: str = "low"


@dataclass
class ValidationReport:
    pipeline_id: str
    timestamp: str
    results: list[ValidationResult] = field(default_factory=list)
    passed: bool = True

    def add(self, result: ValidationResult):
        self.results.append(result)
        if not result.passed:
            self.passed = False

    @property
    def summary(self) -> str:
        total = len(self.results)
        failed = sum(1 for r in self.results if not r.passed)
        return f"{total - failed}/{total} checks passed"


class DataQualityValidator:
    """Configurable data quality checks for pipeline outputs."""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.checks = []

    def expect_row_count_between(self, min_rows: int, max_rows: int):
        self.checks.append(("row_count_range", {"min": min_rows, "max": max_rows}))
        return self

    def expect_no_nulls(self, columns: list[str]):
        self.checks.append(("no_nulls", {"columns": columns}))
        return self

    def expect_null_pct_below(self, column: str, max_pct: float):
        self.checks.append(("null_pct", {"column": column, "max_pct": max_pct}))
        return self

    def expect_unique(self, columns: list[str]):
        self.checks.append(("unique", {"columns": columns}))
        return self

    def expect_values_in_set(self, column: str, valid_values: set):
        self.checks.append(("values_in_set", {"column": column, "valid": valid_values}))
        return self

    def expect_column_match_regex(self, column: str, pattern: str):
        self.checks.append(("regex_match", {"column": column, "pattern": pattern}))
        return self

    def expect_freshness(self, max_age_minutes: float):
        self.checks.append(("freshness", {"max_age": max_age_minutes}))
        return self

    def validate(self, data: dict) -> ValidationReport:
        """Run all configured checks against data.

        Args:
            data: Dict with keys matching check requirements:
                - row_count: int
                - columns: dict of {name: {"values": list, "null_count": int, "total": int}}
                - last_updated: datetime string
        """
        report = ValidationReport(
            pipeline_id=self.pipeline_id,
            timestamp=datetime.now().isoformat()
        )

        for check_name, params in self.checks:
            result = self._run_check(check_name, params, data)
            report.add(result)

        return report

    def _run_check(self, check_name: str, params: dict, data: dict) -> ValidationResult:
        try:
            if check_name == "row_count_range":
                return self._check_row_count(data, params["min"], params["max"])
            elif check_name == "no_nulls":
                return self._check_no_nulls(data, params["columns"])
            elif check_name == "null_pct":
                return self._check_null_pct(data, params["column"], params["max_pct"])
            elif check_name == "unique":
                return self._check_unique(data, params["columns"])
            elif check_name == "values_in_set":
                return self._check_values_in_set(data, params["column"], params["valid"])
            elif check_name == "regex_match":
                return self._check_regex(data, params["column"], params["pattern"])
            elif check_name == "freshness":
                return self._check_freshness(data, params["max_age"])
            else:
                return ValidationResult(check_name, False, f"Unknown check: {check_name}")
        except Exception as e:
            return ValidationResult(check_name, False, f"Check error: {e}", severity="high")

    def _check_row_count(self, data: dict, min_r: int, max_r: int) -> ValidationResult:
        count = data.get("row_count", 0)
        passed = min_r <= count <= max_r
        return ValidationResult(
            "row_count_range", passed,
            f"Row count {count} {'within' if passed else 'outside'} [{min_r}, {max_r}]",
            severity="high" if not passed else "low"
        )

    def _check_no_nulls(self, data: dict, columns: list[str]) -> ValidationResult:
        cols = data.get("columns", {})
        failed = []
        for col in columns:
            info = cols.get(col, {})
            if info.get("null_count", 0) > 0:
                failed.append(f"{col}({info['null_count']} nulls)")
        passed = len(failed) == 0
        return ValidationResult(
            "no_nulls", passed,
            f"{'No nulls found' if passed else 'Nulls in: ' + ', '.join(failed)}",
            severity="high" if not passed else "low"
        )

    def _check_null_pct(self, data: dict, column: str, max_pct: float) -> ValidationResult:
        info = data.get("columns", {}).get(column, {})
        total = info.get("total", 1)
        null_count = info.get("null_count", 0)
        pct = (null_count / total * 100) if total > 0 else 0
        passed = pct <= max_pct
        return ValidationResult(
            f"null_pct_{column}", passed,
            f"{column} null%: {pct:.1f}% (max: {max_pct}%)",
            severity="medium" if not passed else "low"
        )

    def _check_unique(self, data: dict, columns: list[str]) -> ValidationResult:
        cols = data.get("columns", {})
        failed = []
        for col in columns:
            values = cols.get(col, {}).get("values", [])
            if len(values) != len(set(values)):
                dupes = len(values) - len(set(values))
                failed.append(f"{col}({dupes} duplicates)")
        passed = len(failed) == 0
        return ValidationResult(
            "unique", passed,
            f"{'All unique' if passed else 'Duplicates in: ' + ', '.join(failed)}",
            severity="medium" if not passed else "low"
        )

    def _check_values_in_set(self, data: dict, column: str,
                              valid: set) -> ValidationResult:
        values = data.get("columns", {}).get(column, {}).get("values", [])
        invalid = set(values) - valid
        passed = len(invalid) == 0
        return ValidationResult(
            f"values_in_set_{column}", passed,
            f"{'All valid' if passed else f'Invalid values: {invalid}'}",
            severity="medium" if not passed else "low"
        )

    def _check_regex(self, data: dict, column: str, pattern: str) -> ValidationResult:
        values = data.get("columns", {}).get(column, {}).get("values", [])
        regex = re.compile(pattern)
        failed = [v for v in values if not regex.match(str(v))]
        passed = len(failed) == 0
        return ValidationResult(
            f"regex_{column}", passed,
            f"{'All match' if passed else f'{len(failed)} values failed pattern {pattern}'}",
            severity="medium" if not passed else "low"
        )

    def _check_freshness(self, data: dict, max_age: float) -> ValidationResult:
        last = data.get("last_updated")
        if not last:
            return ValidationResult("freshness", False, "No timestamp available", severity="high")
        dt = datetime.fromisoformat(last)
        age = (datetime.now() - dt).total_seconds() / 60
        passed = age <= max_age
        return ValidationResult(
            "freshness", passed,
            f"Data age: {age:.1f}min (max: {max_age}min)",
            severity="high" if not passed else "low"
        )
