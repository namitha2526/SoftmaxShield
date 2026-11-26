"""Configuration and constants for SoftmaxShield."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


# Standard well-known PyPI packages used as ground-truth reference for typosquatting detection.
TOP_PYPI_PACKAGES: List[str] = [
    "requests",
    "urllib3",
    "numpy",
    "pandas",
    "flask",
    "django",
    "cryptography",
    "torch",
    "tensorflow",
    "scipy",
    "pydantic",
    "pytest",
    "httpx",
    "fastapi",
    "boto3",
    "click",
    "typer",
    "scikit-learn",
    "beautifulsoup4",
    "jinja2",
    "sqlalchemy",
    "celery",
    "redis",
    "pillow",
    "matplotlib",
    "transformers",
    "pyyaml",
    "aiohttp",
    "black",
    "flake8",
    "mypy",
    "coverage",
    "seaborn",
    "networkx",
    "openpyxl",
    "paramiko",
    "psycopg2",
    "alembic",
    "lxml",
    "six",
    "certifi",
    "idna",
    "packaging",
    "typing-extensions",
    "colorama",
    "rich",
    "tqdm",
    "wheel",
    "setuptools",
]

# Mapping from common Python import names to PyPI distribution package names.
KNOWN_IMPORT_TO_DIST: Dict[str, str] = {
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "git": "GitPython",
    "jose": "python-jose",
    "crontab": "python-crontab",
    "pptx": "python-pptx",
    "serial": "pyserial",
    "OpenGL": "PyOpenGL",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "google.protobuf": "protobuf",
    "pkg_resources": "setuptools",
    "attr": "attrs",
    "websocket": "websocket-client",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "dns": "dnspython",
}


@dataclass
class ShieldConfig:
    """Configuration settings for the SoftmaxShield verification engine."""

    new_package_age_days: int = 30
    min_downloads_threshold: int = 500
    typosquat_distance_threshold: int = 2
    typosquat_similarity_threshold: float = 0.80
    confidence_threshold: float = 0.85
    strict_mode: bool = False
    pypi_api_timeout: float = 5.0
    pypi_base_url: str = "https://pypi.org/pypi"
    max_correction_rounds: int = 3
    offline_mode: bool = False
    correction_confidence_threshold: float = 0.75
    enable_semantic_correction: bool = True
    trusted_packages: List[str] = field(default_factory=lambda: list(TOP_PYPI_PACKAGES))
    import_to_dist_map: Dict[str, str] = field(default_factory=lambda: dict(KNOWN_IMPORT_TO_DIST))

