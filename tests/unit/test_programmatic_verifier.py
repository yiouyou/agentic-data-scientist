"""Unit tests for programmatic_verifier module."""

import json
import os
import struct
import tempfile
from pathlib import Path

import pytest

from agentic_data_scientist.core.programmatic_verifier import (
    HardChecks,
    SoftSignals,
    VerificationResult,
    check_csv_schema,
    check_files_exist,
    check_json_schema,
    check_no_traceback,
    extract_metrics_from_text,
    run_programmatic_checks,
    validate_image_file,
)


@pytest.fixture
def tmp_workdir(tmp_path):
    return tmp_path


class TestCheckFilesExist:
    def test_all_exist(self, tmp_workdir):
        (tmp_workdir / "a.csv").write_text("col\n1")
        (tmp_workdir / "b.json").write_text("{}")
        ok, missing = check_files_exist(tmp_workdir, ["a.csv", "b.json"])
        assert ok is True
        assert missing == []

    def test_some_missing(self, tmp_workdir):
        (tmp_workdir / "a.csv").write_text("col\n1")
        ok, missing = check_files_exist(tmp_workdir, ["a.csv", "gone.txt"])
        assert ok is False
        assert "gone.txt" in missing

    def test_empty_list(self, tmp_workdir):
        ok, missing = check_files_exist(tmp_workdir, [])
        assert ok is True
        assert missing == []


class TestCheckCsvSchema:
    def test_valid_csv(self, tmp_workdir):
        path = tmp_workdir / "data.csv"
        path.write_text("name,age\nAlice,30\nBob,25\n")
        valid, err = check_csv_schema(path)
        assert valid is True
        assert err == ""

    def test_empty_csv(self, tmp_workdir):
        path = tmp_workdir / "empty.csv"
        path.write_text("")
        valid, err = check_csv_schema(path)
        assert valid is False
        assert "Empty CSV" in err

    def test_missing_columns(self, tmp_workdir):
        path = tmp_workdir / "data.csv"
        path.write_text("name,age\nAlice,30\n")
        valid, err = check_csv_schema(path, expected_columns=["name", "salary"])
        assert valid is False
        assert "salary" in err

    def test_min_rows(self, tmp_workdir):
        path = tmp_workdir / "data.csv"
        path.write_text("col\n1\n")
        valid, err = check_csv_schema(path, min_rows=5)
        assert valid is False
        assert "only 1 data rows" in err

    def test_file_not_found(self, tmp_workdir):
        valid, err = check_csv_schema(tmp_workdir / "nope.csv")
        assert valid is False
        assert "not found" in err.lower()


class TestCheckJsonSchema:
    def test_valid_json(self, tmp_workdir):
        path = tmp_workdir / "data.json"
        path.write_text(json.dumps({"key": "value"}))
        valid, err = check_json_schema(path)
        assert valid is True

    def test_invalid_json(self, tmp_workdir):
        path = tmp_workdir / "bad.json"
        path.write_text("{invalid json")
        valid, err = check_json_schema(path)
        assert valid is False
        assert "parse error" in err.lower()

    def test_file_not_found(self, tmp_workdir):
        valid, err = check_json_schema(tmp_workdir / "nope.json")
        assert valid is False


class TestCheckNoTraceback:
    def test_clean_file(self, tmp_workdir):
        path = tmp_workdir / "output.log"
        path.write_text("All good\nProcessing complete\n")
        clean, snippet = check_no_traceback(path)
        assert clean is True
        assert snippet == ""

    def test_file_with_traceback(self, tmp_workdir):
        path = tmp_workdir / "error.log"
        path.write_text(
            "Starting...\nTraceback (most recent call last):\n  File \"test.py\", line 10\nValueError: bad input\n"
        )
        clean, snippet = check_no_traceback(path)
        assert clean is False
        assert "Traceback" in snippet

    def test_file_not_found_is_clean(self, tmp_workdir):
        clean, snippet = check_no_traceback(tmp_workdir / "nope.txt")
        assert clean is True


class TestExtractMetrics:
    def test_accuracy_and_auc(self):
        text = "Model results:\naccuracy: 0.92\nAUC = 0.87\nloss = 0.15"
        metrics = extract_metrics_from_text(text)
        assert abs(metrics["accuracy"] - 0.92) < 1e-6
        assert abs(metrics["auc"] - 0.87) < 1e-6
        assert abs(metrics["loss"] - 0.15) < 1e-6

    def test_p_value_scientific(self):
        text = "p_value: 2.3e-05"
        metrics = extract_metrics_from_text(text)
        assert abs(metrics["p_value"] - 2.3e-05) < 1e-10

    def test_no_metrics(self):
        text = "Hello world, nothing here"
        metrics = extract_metrics_from_text(text)
        assert metrics == {}


class TestValidateImage:
    def test_png(self, tmp_workdir):
        path = tmp_workdir / "test.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert validate_image_file(path) is True

    def test_jpeg(self, tmp_workdir):
        path = tmp_workdir / "test.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert validate_image_file(path) is True

    def test_svg(self, tmp_workdir):
        path = tmp_workdir / "test.svg"
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        assert validate_image_file(path) is True

    def test_empty_file(self, tmp_workdir):
        path = tmp_workdir / "empty.png"
        path.write_bytes(b"")
        assert validate_image_file(path) is False

    def test_not_image(self, tmp_workdir):
        path = tmp_workdir / "text.png"
        path.write_text("this is not an image")
        assert validate_image_file(path) is False

    def test_missing_file(self, tmp_workdir):
        assert validate_image_file(tmp_workdir / "nope.png") is False


class TestVerificationResult:
    def test_pass_verdict(self):
        result = VerificationResult()
        assert result.verdict == "pass"
        assert "PASS" in result.to_summary()

    def test_fail_verdict_with_details(self):
        hc = HardChecks(file_exists=False, missing_files=["data.csv"])
        result = VerificationResult(hard_checks=hc, verdict="fail")
        summary = result.to_summary()
        assert "FAIL" in summary
        assert "data.csv" in summary

    def test_to_dict_roundtrip(self):
        result = VerificationResult()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["verdict"] == "pass"
        assert d["hard_checks"]["file_exists"] is True

    def test_warn_on_low_coverage(self):
        ss = SoftSignals(artifact_coverage=0.5)
        assert ss.has_warnings is True

    def test_warn_on_invalid_image(self):
        ss = SoftSignals(image_valid={"plot.png": False})
        assert ss.has_warnings is True

    def test_no_warn_on_full_coverage(self):
        ss = SoftSignals(artifact_coverage=1.0, image_valid={"plot.png": True})
        assert ss.has_warnings is False


class TestRunProgrammaticChecks:
    def test_all_pass(self, tmp_workdir):
        (tmp_workdir / "output.csv").write_text("col\n1\n2\n")
        (tmp_workdir / "results.json").write_text('{"score": 0.9}')

        stage = {
            "title": "Analysis",
            "outputs_produced": ["output.csv", "results.json"],
            "implementation_result": "accuracy: 0.95",
        }
        criteria = [{"index": 0, "criteria": "Produce results", "met": False}]

        result = run_programmatic_checks(tmp_workdir, stage, criteria)
        assert result.verdict == "pass"
        assert result.hard_checks.all_pass is True
        assert result.soft_signals.artifact_coverage == 1.0
        assert "accuracy" in result.soft_signals.metrics

    def test_missing_file_fails(self, tmp_workdir):
        stage = {
            "title": "Analysis",
            "outputs_produced": ["missing.csv"],
            "implementation_result": "",
        }
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.verdict == "fail"
        assert result.hard_checks.file_exists is False
        assert "missing.csv" in result.hard_checks.missing_files

    def test_bad_csv_schema_fails(self, tmp_workdir):
        path = tmp_workdir / "bad.csv"
        path.write_text("")
        stage = {
            "title": "Analysis",
            "outputs_produced": ["bad.csv"],
            "implementation_result": "",
        }
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.verdict == "fail"
        assert result.hard_checks.schema_valid is False

    def test_traceback_in_implementation_result(self, tmp_workdir):
        stage = {
            "title": "Analysis",
            "outputs_produced": [],
            "implementation_result": "Traceback (most recent call last):\nValueError: oops",
        }
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.verdict == "fail"
        assert result.hard_checks.no_traceback is False

    def test_partial_coverage_warns(self, tmp_workdir):
        (tmp_workdir / "a.csv").write_text("col\n1\n")
        stage = {
            "title": "Analysis",
            "outputs_produced": ["a.csv", "b.csv", "c.csv"],
            "implementation_result": "",
        }
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.verdict == "fail"
        assert result.soft_signals.artifact_coverage < 1.0

    def test_nonexistent_workdir_fails(self, tmp_workdir):
        stage = {"title": "X", "outputs_produced": [], "implementation_result": ""}
        result = run_programmatic_checks(tmp_workdir / "nonexistent", stage, [])
        assert result.verdict == "fail"

    def test_image_validation_in_outputs(self, tmp_workdir):
        png_path = tmp_workdir / "plot.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        bad_png = tmp_workdir / "bad.png"
        bad_png.write_text("not an image")

        stage = {
            "title": "Viz",
            "outputs_produced": ["plot.png", "bad.png"],
            "implementation_result": "",
        }
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.soft_signals.image_valid["plot.png"] is True
        assert result.soft_signals.image_valid["bad.png"] is False

    def test_no_outputs_passes(self, tmp_workdir):
        stage = {"title": "Thinking", "outputs_produced": [], "implementation_result": "All good"}
        result = run_programmatic_checks(tmp_workdir, stage, [])
        assert result.verdict == "pass"
        assert result.soft_signals.artifact_coverage == 1.0
