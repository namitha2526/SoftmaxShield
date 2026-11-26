"""Package verification against PyPI with typosquatting detection and metadata analysis."""

from __future__ import annotations

import datetime
import importlib.metadata
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    IssueCategory,
    Severity,
    ShieldIssue,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein distance between two strings using O(min(m, n)) space."""
    s1 = s1.lower()
    s2 = s2.lower()
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    previous_row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        current_row = [i + 1] + [0] * len(s1)
        for j, c1 in enumerate(s1):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity ratio in [0.0, 1.0]."""
    s1_norm = s1.lower().replace("-", "_")
    s2_norm = s2.lower().replace("-", "_")
    max_len = max(len(s1_norm), len(s2_norm))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1_norm, s2_norm)
    return max(0.0, 1.0 - (dist / max_len))


class PackageVerifier:
    """Verifies third-party Python packages using PyPI metadata and detects typosquatting."""

    def __init__(self, config: Optional[ShieldConfig] = None) -> None:
        self.config = config or ShieldConfig()
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        # Pre-populate installed packages distribution map if available
        self._installed_distributions: Dict[str, List[str]] = {}
        try:
            self._installed_distributions = importlib.metadata.packages_distributions()
        except Exception as e:
            logger.debug("Could not inspect local packages distributions: %s", e)

    def map_import_to_distribution(self, import_name: str) -> str:
        """Resolve a Python top-level import name to its PyPI distribution name.

        Handles:
            1. Known static dictionary mapping (e.g., yaml -> PyYAML, PIL -> Pillow)
            2. Direct matches in local installed packages
            3. Standard PEP 503 normalization (replacing underscores with hyphens)
        """
        top_name = import_name.split(".")[0]

        # 1. Check static known mapping first for canonical PyPI casing
        if top_name in self.config.import_to_dist_map:
            return self.config.import_to_dist_map[top_name]

        for k, v in self.config.import_to_dist_map.items():
            if k.lower() == top_name.lower():
                return v

        # 2. Check local installed distributions mapping
        if top_name in self._installed_distributions:
            dists = self._installed_distributions[top_name]
            if dists:
                return dists[0]

        # 3. Default normalized name
        return top_name

    def query_pypi(self, distribution_name: str) -> Optional[Dict[str, Any]]:
        """Fetch package metadata from PyPI JSON API with in-memory caching."""
        normalized = distribution_name.strip()
        if normalized in self._cache:
            return self._cache[normalized]

        if self.config.offline_mode:
            return None

        url = f"{self.config.pypi_base_url}/{normalized}/json"
        try:
            response = requests.get(url, timeout=self.config.pypi_api_timeout)
            if response.status_code == 200:
                data = response.json()
                self._cache[normalized] = data
                return data
            elif response.status_code == 404:
                # Also try hyphens vs underscores if not found
                alt_name = normalized.replace("_", "-") if "_" in normalized else normalized.replace("-", "_")
                if alt_name != normalized:
                    alt_url = f"{self.config.pypi_base_url}/{alt_name}/json"
                    alt_resp = requests.get(alt_url, timeout=self.config.pypi_api_timeout)
                    if alt_resp.status_code == 200:
                        data = alt_resp.json()
                        self._cache[normalized] = data
                        return data

                # Definite 404: Package does not exist on PyPI
                self._cache[normalized] = {"_status": 404}
                return {"_status": 404}
            else:
                logger.debug("PyPI returned status %s for %s", response.status_code, normalized)
                return None
        except requests.RequestException as exc:
            logger.debug("Network error querying PyPI for '%s': %s", normalized, exc)
            return None

    def find_typosquat_target(self, package_name: str) -> Optional[Tuple[str, int, float]]:
        """Check if package_name is suspiciously close to a trusted popular package.

        Returns:
            Tuple of (matched_package, distance, similarity) or None if no close match.
        """
        pkg_clean = package_name.lower().replace("-", "_")
        best_match: Optional[Tuple[str, int, float]] = None

        for trusted in self.config.trusted_packages:
            trusted_clean = trusted.lower().replace("-", "_")
            if pkg_clean == trusted_clean:
                continue

            dist = levenshtein_distance(pkg_clean, trusted_clean)
            sim = normalized_similarity(pkg_clean, trusted_clean)

            if dist <= self.config.typosquat_distance_threshold or sim >= self.config.typosquat_similarity_threshold:
                if best_match is None or dist < best_match[1]:
                    best_match = (trusted, dist, sim)

        return best_match

    def verify_package(
        self,
        import_name: str,
        file_name: str,
        line: int,
        col: int,
    ) -> Optional[ShieldIssue]:
        """Verify existence, age, download health, and typosquatting safety of a package.

        Note: NEVER installs unverified packages.
        """
        dist_name = self.map_import_to_distribution(import_name)
        pypi_data = self.query_pypi(dist_name)

        # 1. Package does not exist on PyPI (confirmed 404)
        if isinstance(pypi_data, dict) and pypi_data.get("_status") == 404:
            typo_match = self.find_typosquat_target(dist_name)
            if typo_match:
                trusted_pkg, dist, sim = typo_match
                return ShieldIssue(
                    file_name=file_name,
                    line=line,
                    col=col,
                    detected_identifier=import_name,
                    issue_type=IssueCategory.INVALID_PACKAGE,
                    severity=Severity.CRITICAL,
                    verification_status=VerificationStatus.VERIFIED_INVALID,
                    explanation=(
                        f"Package '{dist_name}' does not exist on PyPI. It is dangerously close to "
                        f"legitimate package '{trusted_pkg}' (Levenshtein distance: {dist}, similarity: {sim:.2f}). "
                        f"Confirmed hallucinated package with high typosquatting risk."
                    ),
                    suggested_correction=trusted_pkg,
                )
            else:
                return ShieldIssue(
                    file_name=file_name,
                    line=line,
                    col=col,
                    detected_identifier=import_name,
                    issue_type=IssueCategory.INVALID_PACKAGE,
                    severity=Severity.HIGH,
                    verification_status=VerificationStatus.VERIFIED_INVALID,
                    explanation=f"Package '{dist_name}' (imported as '{import_name}') was not found on PyPI. Hallucinated package reference.",
                    suggested_correction=None,
                )

        # 2. PyPI is unreachable / offline
        if pypi_data is None:
            trusted_clean = [t.lower().replace("-", "_") for t in self.config.trusted_packages]
            if dist_name.lower().replace("-", "_") in trusted_clean:
                # Known trusted package verified offline
                return None

            typo_match = self.find_typosquat_target(dist_name)
            if typo_match:
                trusted_pkg, dist, sim = typo_match
                return ShieldIssue(
                    file_name=file_name,
                    line=line,
                    col=col,
                    detected_identifier=import_name,
                    issue_type=IssueCategory.INVALID_PACKAGE,
                    severity=Severity.CRITICAL,
                    verification_status=VerificationStatus.VERIFIED_INVALID,
                    explanation=(
                        f"Package '{dist_name}' does not exist on PyPI. It is dangerously close to "
                        f"legitimate package '{trusted_pkg}' (Levenshtein distance: {dist}, similarity: {sim:.2f}). "
                        f"Confirmed hallucinated package with high typosquatting risk."
                    ),
                    suggested_correction=trusted_pkg,
                )

            # Inconclusive when network is down and package is unknown
            return ShieldIssue(
                file_name=file_name,
                line=line,
                col=col,
                detected_identifier=import_name,
                issue_type=IssueCategory.UNVERIFIED_REFERENCE,
                severity=Severity.INFO if not self.config.strict_mode else Severity.WARNING,
                verification_status=VerificationStatus.INCONCLUSIVE,
                explanation=f"PyPI is unreachable; verification for '{dist_name}' is inconclusive.",
                suggested_correction=None,
            )

        # 2. Package exists on PyPI: inspect metadata for security risks / anomalies
        info = pypi_data.get("info", {})
        releases = pypi_data.get("releases", {})
        urls = pypi_data.get("urls", [])

        # Check for typosquatting attack package that actually exists on PyPI
        typo_match = self.find_typosquat_target(dist_name)
        if typo_match:
            trusted_pkg, dist, sim = typo_match
            # If it's a known popular package itself, skip
            if dist_name.lower().replace("-", "_") != trusted_pkg.lower().replace("-", "_"):
                return ShieldIssue(
                    file_name=file_name,
                    line=line,
                    col=col,
                    detected_identifier=import_name,
                    issue_type=IssueCategory.SUSPICIOUS_PACKAGE,
                    severity=Severity.HIGH,
                    verification_status=VerificationStatus.SUSPICIOUS,
                    explanation=(
                        f"Package '{dist_name}' exists on PyPI but is suspiciously similar to "
                        f"the high-reputation package '{trusted_pkg}' (similarity: {sim:.2f}). "
                        f"Potential typosquatting attack package."
                    ),
                    suggested_correction=trusted_pkg,
                )

        # Check package age (< 30 days old)
        first_release_date = self._extract_first_release_date(releases, urls)
        if first_release_date:
            age_days = (datetime.datetime.now(datetime.timezone.utc) - first_release_date).days
            if age_days < self.config.new_package_age_days:
                return ShieldIssue(
                    file_name=file_name,
                    line=line,
                    col=col,
                    detected_identifier=import_name,
                    issue_type=IssueCategory.SUSPICIOUS_PACKAGE,
                    severity=Severity.WARNING,
                    verification_status=VerificationStatus.SUSPICIOUS,
                    explanation=(
                        f"Package '{dist_name}' is newly published ({age_days} days old, "
                        f"threshold < {self.config.new_package_age_days} days). "
                        f"Newly registered packages carry higher supply-chain risks."
                    ),
                    suggested_correction=None,
                )

        # Check package completeness / suspicious lack of description or project health
        description = info.get("description", "") or ""
        summary = info.get("summary", "") or ""
        if len(description.strip()) < 20 and len(summary.strip()) < 10:
            return ShieldIssue(
                file_name=file_name,
                line=line,
                col=col,
                detected_identifier=import_name,
                issue_type=IssueCategory.SUSPICIOUS_PACKAGE,
                severity=Severity.WARNING,
                verification_status=VerificationStatus.SUSPICIOUS,
                explanation=(
                    f"Package '{dist_name}' has unusually low metadata completeness (empty summary/description). "
                    f"Verify authenticity before using."
                ),
                suggested_correction=None,
            )

        return None

    def _extract_first_release_date(
        self, releases: Dict[str, List[Dict[str, Any]]], urls: List[Dict[str, Any]]
    ) -> Optional[datetime.datetime]:
        """Extract earliest release upload timestamp from PyPI releases data."""
        earliest: Optional[datetime.datetime] = None

        all_upload_entries = []
        for file_list in releases.values():
            all_upload_entries.extend(file_list)
        all_upload_entries.extend(urls)

        for entry in all_upload_entries:
            upload_time_str = entry.get("upload_time_iso_8601") or entry.get("upload_time")
            if upload_time_str:
                try:
                    dt = datetime.datetime.fromisoformat(upload_time_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    if earliest is None or dt < earliest:
                        earliest = dt
                except ValueError:
                    continue

        return earliest
