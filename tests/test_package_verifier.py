"""Unit tests for package verification, PyPI integration, and typosquatting defense."""

import unittest.mock as mock
import pytest
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import IssueCategory, Severity, VerificationStatus
from softmaxshield.verifier.package_verifier import (
    PackageVerifier,
    levenshtein_distance,
    normalized_similarity,
)


def test_levenshtein_distance_and_similarity():
    """Verify Levenshtein distance and normalized similarity computations."""
    assert levenshtein_distance("requests", "requests") == 0
    assert levenshtein_distance("requests", "reqeusts") == 2
    assert levenshtein_distance("requests", "request") == 1
    assert levenshtein_distance("", "abc") == 3

    sim_exact = normalized_similarity("requests", "requests")
    assert sim_exact == 1.0

    sim_typo = normalized_similarity("requests", "reqeusts")
    assert sim_typo >= 0.75


def test_import_to_distribution_mapping():
    """Verify standard import names map correctly to PyPI distribution package names."""
    verifier = PackageVerifier()
    assert verifier.map_import_to_distribution("yaml") == "PyYAML"
    assert verifier.map_import_to_distribution("PIL") == "Pillow"
    assert verifier.map_import_to_distribution("sklearn") == "scikit-learn"
    assert verifier.map_import_to_distribution("cv2") == "opencv-python"
    assert verifier.map_import_to_distribution("requests") == "requests"


@mock.patch("requests.get")
def test_verify_real_package_success(mock_get, mock_pypi_requests_success):
    """Verify that an existing, mature package produces no issue."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_pypi_requests_success
    mock_get.return_value = mock_resp

    verifier = PackageVerifier()
    issue = verifier.verify_package("requests", "test.py", line=1, col=0)
    assert issue is None


@mock.patch("requests.get")
def test_verify_nonexistent_package(mock_get):
    """Verify that a nonexistent package is flagged as INVALID_PACKAGE."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    verifier = PackageVerifier()
    issue = verifier.verify_package("totally_nonexistent_hallucinated_xyz", "test.py", line=2, col=0)

    assert issue is not None
    assert issue.issue_type == IssueCategory.INVALID_PACKAGE
    assert issue.severity == Severity.HIGH
    assert issue.verification_status == VerificationStatus.VERIFIED_INVALID
    assert "was not found on PyPI" in issue.explanation


@mock.patch("requests.get")
def test_verify_typosquatting_nonexistent_package(mock_get):
    """Verify that a fake package with close edit distance to a top package is flagged as CRITICAL typosquat."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    verifier = PackageVerifier()
    # 'reqeusts' is an edit distance of 2 from 'requests'
    issue = verifier.verify_package("reqeusts", "test.py", line=3, col=0)

    assert issue is not None
    assert issue.issue_type == IssueCategory.INVALID_PACKAGE
    assert issue.severity == Severity.CRITICAL
    assert issue.suggested_correction == "requests"
    assert "typosquatting risk" in issue.explanation


@mock.patch("requests.get")
def test_verify_newly_published_package(mock_get, mock_pypi_new_package):
    """Verify that a package published less than 30 days ago is flagged as SUSPICIOUS_PACKAGE with WARNING."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_pypi_new_package
    mock_get.return_value = mock_resp

    verifier = PackageVerifier()
    issue = verifier.verify_package("super-fast-llm-tool", "test.py", line=4, col=0)

    assert issue is not None
    assert issue.issue_type == IssueCategory.SUSPICIOUS_PACKAGE
    assert issue.severity == Severity.WARNING
    assert issue.verification_status == VerificationStatus.SUSPICIOUS
    assert "newly published" in issue.explanation


@mock.patch("requests.get")
def test_verify_package_with_low_metadata(mock_get):
    """Verify that an existing package with empty description/summary is flagged as suspicious."""
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "info": {"name": "empty-meta-pkg", "version": "1.0.0", "summary": "", "description": ""},
        "releases": {
            "1.0.0": [{"upload_time_iso_8601": "2020-01-01T00:00:00Z"}]
        },
        "urls": [],
    }
    mock_get.return_value = mock_resp

    verifier = PackageVerifier()
    issue = verifier.verify_package("empty-meta-pkg", "test.py", line=5, col=0)

    assert issue is not None
    assert issue.issue_type == IssueCategory.SUSPICIOUS_PACKAGE
    assert issue.severity == Severity.WARNING
    assert "low metadata completeness" in issue.explanation

