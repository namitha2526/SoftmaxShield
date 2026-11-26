"""Comprehensive regression and validation tests for Semantic-Aware Auto-Correction.

Verifies:
1. np.fast_average(data) -> np.average(data)
2. pd.create_dataframe(data) -> pd.DataFrame(data)
3. np.normalize_array(data) -> (data - np.min(data)) / (np.max(data) - np.min(data))
4. np.mean(data) -> No correction needed (0 issues)
5. Direct imports:
   - from numpy import array (valid, 0 issues)
   - from numpy import fast_average (corrected to from numpy import average)
6. Aliased imports (np, pd, numpy)
7. Valid APIs with incorrect arguments (json.dumps with invalid_kw)
8. Unresolvable APIs (numpy.quantum_teleport) -> requests manual intervention, does not guess
9. Execution & Functional Equivalence tests on corrected code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import IssueCategory
from softmaxshield.corrector.client import MockLLMClient
from softmaxshield.corrector.engine import SelfCorrectionEngine
from softmaxshield.corrector.semantic_analyzer import SemanticAnalyzer
from softmaxshield.pipeline import ShieldPipeline


@pytest.fixture
def corrector_engine() -> SelfCorrectionEngine:
    config = ShieldConfig(offline_mode=True, correction_confidence_threshold=0.75)
    pipeline = ShieldPipeline(config)
    return SelfCorrectionEngine(pipeline=pipeline, config=config)


# =========================================================================
# TEST 1: np.fast_average(data) -> np.average(data)
# =========================================================================
def test_fast_average_semantic_correction(corrector_engine: SelfCorrectionEngine):
    code = (
        "import numpy as np\n"
        "data = np.array([10, 20, 30])\n"
        "avg = np.fast_average(data)\n"
    )
    result = corrector_engine.run(code)

    assert result.success is True
    assert "np.average(data)" in result.final_code
    assert "np.fast_average" not in result.final_code

    # Execute functionally to verify mathematical equivalence
    namespace = {}
    exec(result.final_code, namespace)
    assert namespace["avg"] == 20.0


# =========================================================================
# TEST 2: pd.create_dataframe(data) -> pd.DataFrame(data)
# =========================================================================
def test_create_dataframe_semantic_correction(corrector_engine: SelfCorrectionEngine):
    code = (
        "import pandas as pd\n"
        "data = [10, 20, 30]\n"
        "df = pd.create_dataframe(data, columns=['Values'])\n"
    )
    result = corrector_engine.run(code)

    assert result.success is True
    assert "pd.DataFrame(data" in result.final_code
    assert "pd.create_dataframe" not in result.final_code

    # Execute functionally to verify object creation
    namespace = {}
    exec(result.final_code, namespace)
    assert isinstance(namespace["df"], pd.DataFrame)
    assert list(namespace["df"]["Values"]) == [10, 20, 30]


# =========================================================================
# TEST 3: np.normalize_array(data) -> Canonical min-max normalization
# =========================================================================
def test_normalize_array_semantic_correction(corrector_engine: SelfCorrectionEngine):
    code = (
        "import numpy as np\n"
        "data = np.array([10, 20, 30])\n"
        "normalized = np.normalize_array(data)\n"
    )
    result = corrector_engine.run(code)

    assert result.success is True
    # Verify that the system DID NOT fall into the naive trap of replacing with np.array(data)
    assert "np.array(data)" not in result.final_code.splitlines()[-1]
    # Verify canonical min-max normalization formula was synthesized
    assert "np.min" in result.final_code
    assert "np.max" in result.final_code
    assert "(data - np.min(data)) / (np.max(data) - np.min(data))" in result.final_code

    # Execute functionally to verify mathematical correctness: [10, 20, 30] -> [0.0, 0.5, 1.0]
    namespace = {}
    exec(result.final_code, namespace)
    np.testing.assert_allclose(namespace["normalized"], np.array([0.0, 0.5, 1.0]))


# =========================================================================
# TEST 4: np.mean(data) -> Valid code requires no correction
# =========================================================================
def test_valid_code_requires_no_correction(corrector_engine: SelfCorrectionEngine):
    code = (
        "import numpy as np\n"
        "data = np.array([10, 20, 30])\n"
        "mean_val = np.mean(data)\n"
    )
    result = corrector_engine.run(code)

    assert result.success is True
    assert result.rounds_attempted == 0
    assert result.final_code == code
    assert len(result.attempts) == 0


# =========================================================================
# TEST 5: Direct Imports (from numpy import ...)
# =========================================================================
def test_direct_import_valid(corrector_engine: SelfCorrectionEngine):
    code = (
        "from numpy import array\n"
        "arr = array([1, 2, 3])\n"
    )
    result = corrector_engine.run(code)
    assert result.success is True
    assert result.rounds_attempted == 0
    assert result.final_code == code


def test_direct_import_hallucinated_corrected(corrector_engine: SelfCorrectionEngine):
    code = (
        "from numpy import fast_average\n"
        "val = fast_average([10, 20, 30])\n"
    )
    result = corrector_engine.run(code)
    assert result.success is True
    assert "from numpy import average" in result.final_code
    assert "fast_average" not in result.final_code


# =========================================================================
# TEST 6: Unaliased Direct Import (import numpy)
# =========================================================================
def test_unaliased_numpy_normalization(corrector_engine: SelfCorrectionEngine):
    code = (
        "import numpy\n"
        "data = numpy.array([10, 20, 30])\n"
        "normalized = numpy.normalize_array(data)\n"
    )
    result = corrector_engine.run(code)
    assert result.success is True
    assert "numpy.min(data)" in result.final_code
    assert "numpy.max(data)" in result.final_code


# =========================================================================
# TEST 7: Valid API with Invalid Keyword Arguments
# =========================================================================
def test_invalid_kwarg_correction(corrector_engine: SelfCorrectionEngine):
    code = (
        "import json\n"
        "data = {'key': 'value'}\n"
        "dumped = json.dumps(data, indent=2, invalid_flag=True)\n"
    )
    result = corrector_engine.run(code)
    assert result.success is True
    assert "invalid_flag" not in result.final_code
    assert "json.dumps(data, indent=2" in result.final_code

    namespace = {}
    exec(result.final_code, namespace)
    assert '"key": "value"' in namespace["dumped"]


# =========================================================================
# TEST 8: Unresolvable / Non-existent Operation -> Refuses Guessing
# =========================================================================
def test_unresolvable_hallucination_rejects_guessing(corrector_engine: SelfCorrectionEngine):
    code = (
        "import numpy as np\n"
        "data = np.array([1, 2, 3])\n"
        "res = np.quantum_teleport(data)\n"
    )
    result = corrector_engine.run(code)

    # Engine must NOT guess or fabricate an arbitrary function; it must fail and request manual fix
    assert result.success is False
    assert "np.quantum_teleport" in result.final_code


# =========================================================================
# TEST 9: Comprehensive Multi-API User Snippet
# =========================================================================
def test_full_user_snippet_multi_correction(corrector_engine: SelfCorrectionEngine):
    user_code = (
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "data = np.array([10, 20, 30])\n\n"
        "# Calculate average\n"
        "avg = np.fast_average(data)\n\n"
        "# Convert array into DataFrame\n"
        "df = pd.create_dataframe(data, columns=[\"Values\"])\n\n"
        "# Normalize values\n"
        "normalized = np.normalize_array(data)\n\n"
        "print(avg)\n"
        "print(df)\n"
        "print(normalized)\n"
    )
    result = corrector_engine.run(user_code)

    assert result.success is True
    assert "np.average(data)" in result.final_code
    assert "pd.DataFrame(data" in result.final_code
    assert "(data - np.min(data)) / (np.max(data) - np.min(data))" in result.final_code
    assert "np.fast_average" not in result.final_code
    assert "pd.create_dataframe" not in result.final_code
    assert "np.normalize_array" not in result.final_code

    # Execute fully in python environment to guarantee zero runtime failures
    namespace = {}
    exec(result.final_code, namespace)
    assert namespace["avg"] == 20.0
    assert isinstance(namespace["df"], pd.DataFrame)
    np.testing.assert_allclose(namespace["normalized"], np.array([0.0, 0.5, 1.0]))

