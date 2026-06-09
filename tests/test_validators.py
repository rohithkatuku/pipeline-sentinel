"""Tests for data quality validators."""

import pytest
from datetime import datetime, timedelta

from sentinel.validators.quality import DataQualityValidator


class TestDataQualityValidator:
    def setup_method(self):
        self.validator = DataQualityValidator("test-pipeline")

    def test_row_count_pass(self):
        self.validator.expect_row_count_between(100, 1000)
        report = self.validator.validate({"row_count": 500})
        assert report.passed

    def test_row_count_fail_low(self):
        self.validator.expect_row_count_between(100, 1000)
        report = self.validator.validate({"row_count": 10})
        assert not report.passed

    def test_row_count_fail_high(self):
        self.validator.expect_row_count_between(100, 1000)
        report = self.validator.validate({"row_count": 5000})
        assert not report.passed

    def test_no_nulls_pass(self):
        self.validator.expect_no_nulls(["id", "name"])
        data = {
            "columns": {
                "id": {"null_count": 0, "total": 100},
                "name": {"null_count": 0, "total": 100},
            }
        }
        report = self.validator.validate(data)
        assert report.passed

    def test_no_nulls_fail(self):
        self.validator.expect_no_nulls(["id", "email"])
        data = {
            "columns": {
                "id": {"null_count": 0, "total": 100},
                "email": {"null_count": 15, "total": 100},
            }
        }
        report = self.validator.validate(data)
        assert not report.passed

    def test_null_pct_pass(self):
        self.validator.expect_null_pct_below("email", 5.0)
        data = {"columns": {"email": {"null_count": 2, "total": 100}}}
        report = self.validator.validate(data)
        assert report.passed

    def test_null_pct_fail(self):
        self.validator.expect_null_pct_below("email", 5.0)
        data = {"columns": {"email": {"null_count": 20, "total": 100}}}
        report = self.validator.validate(data)
        assert not report.passed

    def test_unique_pass(self):
        self.validator.expect_unique(["id"])
        data = {"columns": {"id": {"values": [1, 2, 3, 4, 5]}}}
        report = self.validator.validate(data)
        assert report.passed

    def test_unique_fail(self):
        self.validator.expect_unique(["id"])
        data = {"columns": {"id": {"values": [1, 2, 2, 3, 4]}}}
        report = self.validator.validate(data)
        assert not report.passed

    def test_values_in_set(self):
        self.validator.expect_values_in_set("status", {"active", "inactive"})
        data = {"columns": {"status": {"values": ["active", "inactive", "active"]}}}
        report = self.validator.validate(data)
        assert report.passed

    def test_values_not_in_set(self):
        self.validator.expect_values_in_set("status", {"active", "inactive"})
        data = {"columns": {"status": {"values": ["active", "deleted"]}}}
        report = self.validator.validate(data)
        assert not report.passed

    def test_freshness_pass(self):
        self.validator.expect_freshness(60)
        data = {"last_updated": datetime.now().isoformat()}
        report = self.validator.validate(data)
        assert report.passed

    def test_freshness_fail(self):
        self.validator.expect_freshness(60)
        old = datetime.now() - timedelta(hours=2)
        data = {"last_updated": old.isoformat()}
        report = self.validator.validate(data)
        assert not report.passed

    def test_summary(self):
        self.validator.expect_row_count_between(100, 1000)
        self.validator.expect_no_nulls(["id"])
        data = {"row_count": 500, "columns": {"id": {"null_count": 0, "total": 500}}}
        report = self.validator.validate(data)
        assert "2/2" in report.summary

    def test_chained_config(self):
        v = (DataQualityValidator("test")
             .expect_row_count_between(10, 1000)
             .expect_no_nulls(["id"])
             .expect_freshness(120))
        assert len(v.checks) == 3
