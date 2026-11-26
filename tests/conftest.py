"""Pytest fixtures and mock data for SoftmaxShield test suite."""

import datetime
from typing import Any, Dict
import pytest


@pytest.fixture
def mock_pypi_requests_success() -> Dict[str, Any]:
    """Mock PyPI API response for requests (mature, healthy package)."""
    return {
        "info": {
            "name": "requests",
            "version": "2.31.0",
            "summary": "Python HTTP for Humans.",
            "description": "Requests is an elegant and simple HTTP library for Python, built for human beings.",
        },
        "releases": {
            "2.31.0": [
                {
                    "upload_time_iso_8601": "2023-05-22T17:48:47.123456Z",
                    "filename": "requests-2.31.0-py3-none-any.whl",
                }
            ],
            "0.2.0": [
                {
                    "upload_time_iso_8601": "2011-02-14T12:00:00.000000Z",
                    "filename": "requests-0.2.0.tar.gz",
                }
            ],
        },
        "urls": [],
    }


@pytest.fixture
def mock_pypi_new_package() -> Dict[str, Any]:
    """Mock PyPI response for a package published 5 days ago."""
    five_days_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()
    return {
        "info": {
            "name": "super-fast-llm-tool",
            "version": "0.1.0",
            "summary": "A brand new package.",
            "description": "Detailed description of a brand new package.",
        },
        "releases": {
            "0.1.0": [
                {
                    "upload_time_iso_8601": five_days_ago,
                    "filename": "super_fast_llm_tool-0.1.0.tar.gz",
                }
            ]
        },
        "urls": [],
    }


@pytest.fixture
def sample_token_logprobs():
    """Sample token-level log probabilities from LLM generation."""
    return [
        {"token": "import", "logprob": -0.01},
        {"token": " ", "logprob": -0.01},
        {"token": "requests", "logprob": -0.02},
        {"token": "\n", "logprob": -0.01},
        {"token": "import", "logprob": -0.01},
        {"token": " ", "logprob": -0.01},
        {"token": "req", "logprob": -1.85},     # Low confidence token (p ~ 0.157)
        {"token": "eusts", "logprob": -2.10},   # Low confidence token (p ~ 0.122)
    ]

