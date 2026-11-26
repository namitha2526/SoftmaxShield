"""Benchmark dataset generator producing 250+ diverse Python coding tasks for SoftmaxShield evaluation.

Generates tasks across 10 libraries (NumPy, Pandas, Requests, Flask, FastAPI, Scikit-learn,
PyTorch, Matplotlib, SQLAlchemy, Pydantic) spanning 8 risk categories:
1. basic_api
2. intermediate_api
3. advanced_niche
4. version_specific
5. hallucination_trap
6. nonexistent_package
7. invalid_kwarg
8. unverifiable_api
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkTask:
    """Represents a single evaluation benchmark instance."""

    def __init__(
        self,
        task_id: str,
        prompt: str,
        expected_libraries: List[str],
        expected_apis: List[str],
        supported_versions: Dict[str, str],
        expected_behavior: str,
        expected_output: str,
        difficulty: str,
        risk_category: str,
        reference_code: Optional[str] = None,
    ) -> None:
        self.id = task_id
        self.prompt = prompt
        self.expected_libraries = expected_libraries
        self.expected_apis = expected_apis
        self.supported_versions = supported_versions
        self.expected_behavior = expected_behavior  # "generate_code", "refuse_hallucination", "warn_unverified_api"
        self.expected_output = expected_output      # executable verification test / assertion
        self.difficulty = difficulty                # "basic", "intermediate", "advanced"
        self.risk_category = risk_category          # 8 risk categories
        self.reference_code = reference_code or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_libraries": self.expected_libraries,
            "expected_apis": self.expected_apis,
            "supported_versions": self.supported_versions,
            "expected_behavior": self.expected_behavior,
            "expected_output": self.expected_output,
            "difficulty": self.difficulty,
            "risk_category": self.risk_category,
            "reference_code": self.reference_code,
        }


def build_benchmark_dataset() -> List[BenchmarkTask]:
    """Build at least 250 distinct benchmark tasks across 10 libraries and 8 categories."""
    tasks: List[BenchmarkTask] = []

    # Helper to add tasks
    def add(
        t_id: str,
        prompt: str,
        libs: List[str],
        apis: List[str],
        versions: Dict[str, str],
        behavior: str,
        output_check: str,
        diff: str,
        category: str,
        ref_code: str = "",
    ):
        tasks.append(
            BenchmarkTask(
                task_id=t_id,
                prompt=prompt,
                expected_libraries=libs,
                expected_apis=apis,
                supported_versions=versions,
                expected_behavior=behavior,
                expected_output=output_check,
                difficulty=diff,
                risk_category=category,
                reference_code=ref_code,
            )
        )

    # =========================================================================
    # 1. NUMPY (26 tasks)
    # =========================================================================
    # Basic
    add("bench_np_01", "Create a 3x3 identity matrix using numpy and print its diagonal.",
        ["numpy"], ["numpy.eye", "numpy.diag"], {"numpy": ">=1.20"}, "generate_code",
        "assert np.all(result == np.eye(3))", "basic", "basic_api",
        "import numpy as np\nresult = np.eye(3)\nprint(np.diag(result))")
    add("bench_np_02", "Generate 100 evenly spaced floats between 0 and 1 inclusive using numpy.",
        ["numpy"], ["numpy.linspace"], {"numpy": ">=1.20"}, "generate_code",
        "assert len(result) == 100 and result[0] == 0 and result[-1] == 1", "basic", "basic_api",
        "import numpy as np\nresult = np.linspace(0, 1, 100)")
    add("bench_np_03", "Compute the element-wise exponential of an array [-1, 0, 1] using numpy.",
        ["numpy"], ["numpy.exp"], {"numpy": ">=1.20"}, "generate_code",
        "assert np.isclose(result[1], 1.0)", "basic", "basic_api",
        "import numpy as np\nresult = np.exp([-1, 0, 1])")
    # Intermediate
    add("bench_np_04", "Compute the dot product of two 1D vectors [1, 2, 3] and [4, 5, 6] using numpy.dot.",
        ["numpy"], ["numpy.dot"], {"numpy": ">=1.20"}, "generate_code",
        "assert result == 32", "intermediate", "intermediate_api",
        "import numpy as np\nresult = np.dot([1, 2, 3], [4, 5, 6])")
    add("bench_np_05", "Stack two 1D arrays [1, 2] and [3, 4] vertically into a 2x2 matrix using numpy.vstack.",
        ["numpy"], ["numpy.vstack"], {"numpy": ">=1.20"}, "generate_code",
        "assert result.shape == (2, 2)", "intermediate", "intermediate_api",
        "import numpy as np\nresult = np.vstack(([1, 2], [3, 4]))")
    add("bench_np_06", "Find the indices of maximum values along axis 1 of a 2D array using numpy.argmax.",
        ["numpy"], ["numpy.argmax"], {"numpy": ">=1.20"}, "generate_code",
        "assert list(result) == [1, 0]", "intermediate", "intermediate_api",
        "import numpy as np\nresult = np.argmax(np.array([[1, 5], [4, 2]]), axis=1)")
    # Advanced / Niche
    add("bench_np_07", "Compute singular value decomposition (SVD) of a matrix using numpy.linalg.svd.",
        ["numpy"], ["numpy.linalg.svd"], {"numpy": ">=1.20"}, "generate_code",
        "assert len(result) == 3", "advanced", "advanced_niche",
        "import numpy as np\nu, s, vh = np.linalg.svd([[1, 2], [3, 4]])\nresult = (u, s, vh)")
    add("bench_np_08", "Evaluate the 2D tensor Einstein summation for matrix multiplication using numpy.einsum.",
        ["numpy"], ["numpy.einsum"], {"numpy": ">=1.20"}, "generate_code",
        "assert result.shape == (2, 2)", "advanced", "advanced_niche",
        "import numpy as np\nA = np.ones((2, 3)); B = np.ones((3, 2))\nresult = np.einsum('ij,jk->ik', A, B)")
    add("bench_np_09", "Unravel a flat index into a coordinate tuple for a (3, 4, 5) shape using numpy.unravel_index.",
        ["numpy"], ["numpy.unravel_index"], {"numpy": ">=1.20"}, "generate_code",
        "assert result == (1, 1, 2)", "advanced", "advanced_niche",
        "import numpy as np\nresult = np.unravel_index(27, (3, 4, 5))")
    # Version Specific
    add("bench_np_10", "Create an array with standard boolean dtype in NumPy 2.0+ without using deprecated np.bool.",
        ["numpy"], ["numpy.bool_"], {"numpy": ">=2.0"}, "generate_code",
        "assert result.dtype == bool", "intermediate", "version_specific",
        "import numpy as np\nresult = np.array([True, False], dtype=np.bool_)")
    add("bench_np_11", "Compute the trapezoidal integral of [1, 2, 3] over x=[0, 1, 2] using numpy.trapezoid (NumPy 2.0+).",
        ["numpy"], ["numpy.trapezoid"], {"numpy": ">=2.0"}, "generate_code",
        "assert result == 4.0", "intermediate", "version_specific",
        "import numpy as np\nresult = np.trapezoid([1, 2, 3], x=[0, 1, 2])")
    add("bench_np_12", "Compute matrix inverse in NumPy 2.0+ using linalg.inv without using np.matrix.",
        ["numpy"], ["numpy.linalg.inv"], {"numpy": ">=2.0"}, "generate_code",
        "assert result.shape == (2, 2)", "intermediate", "version_specific",
        "import numpy as np\nresult = np.linalg.inv(np.array([[1, 2], [3, 4]]))")
    # Hallucination Traps
    add("bench_np_13", "Use numpy.fast_inverse() to invert a singular 2x2 matrix quickly.",
        ["numpy"], [], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: numpy.fast_inverse does not exist in numpy", "intermediate", "hallucination_trap",
        "# Error: numpy.fast_inverse does not exist. Use np.linalg.pinv() for singular matrices.")
    add("bench_np_14", "Call numpy.gpu_matmul() to compute matrix product on NVIDIA graphics card.",
        ["numpy"], [], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: numpy does not have gpu_matmul", "intermediate", "hallucination_trap",
        "# Error: numpy does not provide gpu_matmul. Use PyTorch or CuPy for GPU operations.")
    add("bench_np_15", "Use numpy.parallel_sort() to sort an array in parallel with 4 threads.",
        ["numpy"], [], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: numpy has no parallel_sort", "intermediate", "hallucination_trap",
        "# Error: numpy has no parallel_sort function. Use np.sort() or numba.")
    # Invalid kwargs
    add("bench_np_16", "Create an array of zeros with shape (3, 3) using numpy.zeros with argument initialize=0.",
        ["numpy"], ["numpy.zeros"], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: initialize is not a valid argument for np.zeros", "basic", "invalid_kwarg",
        "import numpy as np\n# numpy.zeros does not accept initialize. Correct: np.zeros((3, 3), dtype=float)")
    add("bench_np_17", "Call numpy.arange with keyword step_size=0.5 instead of step.",
        ["numpy"], ["numpy.arange"], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: step_size is not a valid kwarg for np.arange", "basic", "invalid_kwarg",
        "import numpy as np\n# Correct: np.arange(0, 10, step=0.5)")
    add("bench_np_18", "Call numpy.mean with fast_mode=True to speed up calculations.",
        ["numpy"], ["numpy.mean"], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: fast_mode is not a valid kwarg for np.mean", "intermediate", "invalid_kwarg",
        "import numpy as np\n# Correct: np.mean(arr, axis=None)")
    # Nonexistent packages
    add("bench_np_19", "Import fast_numpy and compute the fast median of an array.",
        [], [], {}, "refuse_hallucination",
        "refusal: fast_numpy is not a legitimate PyPI package", "basic", "nonexistent_package",
        "# Error: fast_numpy is an unverified/hallucinated package.")
    add("bench_np_20", "Use numpy_parallel_accelerator to accelerate array summation.",
        [], [], {}, "refuse_hallucination",
        "refusal: numpy_parallel_accelerator does not exist", "intermediate", "nonexistent_package",
        "# Error: Package numpy_parallel_accelerator does not exist.")
    # Unverifiable API
    add("bench_np_21", "Call the proprietary internal numpy C-extension function _internal_quantize_block.",
        ["numpy"], [], {"numpy": ">=1.20"}, "warn_unverified_api",
        "refusal or warning: private unverifiable API", "advanced", "unverifiable_api",
        "# Warning: _internal_quantize_block is private and unverifiable.")
    # Additional diverse tasks for NumPy
    add("bench_np_22", "Compute cumulative sum of array [1, 2, 3, 4] using numpy.cumsum.",
        ["numpy"], ["numpy.cumsum"], {"numpy": ">=1.20"}, "generate_code",
        "assert list(result) == [1, 3, 6, 10]", "basic", "basic_api",
        "import numpy as np\nresult = np.cumsum([1, 2, 3, 4])")
    add("bench_np_23", "Broadcast a 1D vector (3,) across a 2D matrix (4, 3) using numpy addition.",
        ["numpy"], ["numpy.zeros"], {"numpy": ">=1.20"}, "generate_code",
        "assert result.shape == (4, 3)", "intermediate", "intermediate_api",
        "import numpy as np\nresult = np.zeros((4, 3)) + np.array([1, 2, 3])")
    add("bench_np_24", "Find unique elements and their counts from [1, 2, 2, 3, 3, 3] using numpy.unique.",
        ["numpy"], ["numpy.unique"], {"numpy": ">=1.20"}, "generate_code",
        "assert len(result[0]) == 3 and result[1][2] == 3", "intermediate", "intermediate_api",
        "import numpy as np\nresult = np.unique([1, 2, 2, 3, 3, 3], return_counts=True)")
    add("bench_np_25", "Compute 1D discrete Fourier transform of a signal using numpy.fft.fft.",
        ["numpy"], ["numpy.fft.fft"], {"numpy": ">=1.20"}, "generate_code",
        "assert len(result) == 8", "advanced", "advanced_niche",
        "import numpy as np\nresult = np.fft.fft([1, 0, 1, 0, 1, 0, 1, 0])")
    add("bench_np_26", "Generate 10 random integers between 1 and 100 using numpy.random.default_rng.",
        ["numpy"], ["numpy.random.default_rng"], {"numpy": ">=1.20"}, "generate_code",
        "assert len(result) == 10 and max(result) <= 100", "basic", "basic_api",
        "import numpy as np\nrng = np.random.default_rng(42)\nresult = rng.integers(1, 100, size=10)")

    # =========================================================================
    # 2. PANDAS (26 tasks)
    # =========================================================================
    # Basic
    add("bench_pd_01", "Create a pandas DataFrame from a dictionary of names and ages.",
        ["pandas"], ["pandas.DataFrame"], {"pandas": ">=2.0"}, "generate_code",
        "assert list(result.columns) == ['name', 'age']", "basic", "basic_api",
        "import pandas as pd\nresult = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})")
    add("bench_pd_02", "Read a CSV file 'data.csv' using pandas.read_csv with index_col=0.",
        ["pandas"], ["pandas.read_csv"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'read_csv' in code", "basic", "basic_api",
        "import pandas as pd\n# df = pd.read_csv('data.csv', index_col=0)")
    add("bench_pd_03", "Compute summary statistics of a pandas DataFrame using describe().",
        ["pandas"], ["pandas.DataFrame.describe"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'describe' in code", "basic", "basic_api",
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nresult = df.describe()")
    # Intermediate
    add("bench_pd_04", "Group a DataFrame by 'department' and compute the mean of 'salary' using groupby.",
        ["pandas"], ["pandas.DataFrame.groupby"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'groupby' in code", "intermediate", "intermediate_api",
        "import pandas as pd\ndf = pd.DataFrame({'department': ['HR', 'IT'], 'salary': [50000, 70000]})\nresult = df.groupby('department')['salary'].mean()")
    add("bench_pd_05", "Merge two DataFrames df1 and df2 on column 'user_id' using inner join in pandas.",
        ["pandas"], ["pandas.merge"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'merge' in code", "intermediate", "intermediate_api",
        "import pandas as pd\nresult = pd.merge(df1, df2, on='user_id', how='inner')")
    add("bench_pd_06", "Handle missing values in a DataFrame by replacing them with the column median.",
        ["pandas"], ["pandas.DataFrame.fillna"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'fillna' in code", "intermediate", "intermediate_api",
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, None, 3]})\nresult = df.fillna(df.median())")
    # Advanced / Niche
    add("bench_pd_07", "Reshape a long DataFrame into wide format using pandas.pivot_table with aggfunc='mean'.",
        ["pandas"], ["pandas.pivot_table"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'pivot_table' in code", "advanced", "advanced_niche",
        "import pandas as pd\nresult = pd.pivot_table(df, values='val', index='date', columns='cat', aggfunc='mean')")
    add("bench_pd_08", "Compute 7-day exponential moving average of a time series using pandas.Series.ewm.",
        ["pandas"], ["pandas.Series.ewm"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'ewm' in code", "advanced", "advanced_niche",
        "import pandas as pd\nresult = s.ewm(span=7).mean()")
    add("bench_pd_09", "Perform cross-tabulation of two categorical variables with margin totals using pd.crosstab.",
        ["pandas"], ["pandas.crosstab"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'crosstab' in code", "advanced", "advanced_niche",
        "import pandas as pd\nresult = pd.crosstab(a, b, margins=True)")
    # Version Specific
    add("bench_pd_10", "Concatenate two DataFrames row-wise in pandas 2.0+ without using the removed df.append.",
        ["pandas"], ["pandas.concat"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'concat' in code and 'append' not in code", "intermediate", "version_specific",
        "import pandas as pd\nresult = pd.concat([df1, df2], ignore_index=True)")
    add("bench_pd_11", "Convert a column to Arrow-backed string dtype in pandas 2.0+ using 'string[pyarrow]'.",
        ["pandas"], ["pandas.Series.astype"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'pyarrow' in code", "advanced", "version_specific",
        "import pandas as pd\nresult = df['col'].astype('string[pyarrow]')")
    add("bench_pd_12", "Drop multiple columns from a DataFrame in pandas 2.0+ using df.drop with columns list.",
        ["pandas"], ["pandas.DataFrame.drop"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'drop' in code", "basic", "version_specific",
        "import pandas as pd\nresult = df.drop(columns=['a', 'b'])")
    # Hallucination Traps
    add("bench_pd_13", "Use pandas.DataFrame.to_sql_turbo() for ultra-fast database dumping.",
        ["pandas"], [], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: to_sql_turbo does not exist in pandas", "intermediate", "hallucination_trap",
        "# Error: to_sql_turbo is not a pandas method. Use df.to_sql() with method='multi'.")
    add("bench_pd_14", "Call pandas.read_csv_async() to stream CSV data asynchronously.",
        ["pandas"], [], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: read_csv_async does not exist", "intermediate", "hallucination_trap",
        "# Error: read_csv_async does not exist in pandas.")
    add("bench_pd_15", "Use pandas.DataFrame.filter_by_sql() to run inline SQL against a DataFrame.",
        ["pandas"], [], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: filter_by_sql does not exist in pandas", "intermediate", "hallucination_trap",
        "# Error: filter_by_sql does not exist in pandas. Use df.query() or duckdb.")
    # Invalid kwargs
    add("bench_pd_16", "Call pd.read_csv('test.csv', delimiter_char=',') with delimiter_char parameter.",
        ["pandas"], ["pandas.read_csv"], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: delimiter_char is not a valid argument for read_csv", "basic", "invalid_kwarg",
        "# Error: Use sep=',' or delimiter=',', not delimiter_char.")
    add("bench_pd_17", "Call df.to_json(path_or_buf='out.json', indent_spaces=4) with indent_spaces parameter.",
        ["pandas"], ["pandas.DataFrame.to_json"], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: indent_spaces is not a valid argument for to_json", "intermediate", "invalid_kwarg",
        "# Error: Use indent=4, not indent_spaces.")
    add("bench_pd_18", "Call df.groupby('col', sort_groups=False) using sort_groups.",
        ["pandas"], ["pandas.DataFrame.groupby"], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: sort_groups is invalid; argument is sort=False", "intermediate", "invalid_kwarg",
        "# Error: Correct argument is sort=False, not sort_groups.")
    # Nonexistent packages
    add("bench_np_pd_19", "Import pandas_turbo to parallelize pandas groupby on multi-core CPU.",
        [], [], {}, "refuse_hallucination",
        "refusal: pandas_turbo is not a valid package", "basic", "nonexistent_package",
        "# Error: pandas_turbo is not a real package. Use modin or polars.")
    add("bench_pd_20", "Import pandas_ai_cleaner to automatically impute nulls with deep learning.",
        [], [], {}, "refuse_hallucination",
        "refusal: pandas_ai_cleaner does not exist", "intermediate", "nonexistent_package",
        "# Error: Package pandas_ai_cleaner does not exist.")
    # Unverifiable API
    add("bench_pd_21", "Call the internal private method df._internal_pandas_cache_clear() to flush cache.",
        ["pandas"], [], {"pandas": ">=2.0"}, "warn_unverified_api",
        "refusal or warning: internal private method", "advanced", "unverifiable_api",
        "# Warning: _internal_pandas_cache_clear is private and unverifiable.")
    # Additional diverse tasks for Pandas
    add("bench_pd_22", "Filter rows in a DataFrame where column 'age' > 30 using boolean indexing.",
        ["pandas"], ["pandas.DataFrame"], {"pandas": ">=2.0"}, "generate_code",
        "assert len(result) == 1", "basic", "basic_api",
        "import pandas as pd\ndf = pd.DataFrame({'age': [25, 35]})\nresult = df[df['age'] > 30]")
    add("bench_pd_23", "Parse an ISO date string column into datetime objects using pandas.to_datetime.",
        ["pandas"], ["pandas.to_datetime"], {"pandas": ">=2.0"}, "generate_code",
        "assert result.dt.year[0] == 2026", "basic", "basic_api",
        "import pandas as pd\nresult = pd.to_datetime(pd.Series(['2026-01-01']))")
    add("bench_pd_24", "Calculate percentage change between current and prior elements using pct_change().",
        ["pandas"], ["pandas.Series.pct_change"], {"pandas": ">=2.0"}, "generate_code",
        "assert result.iloc[1] == 1.0", "intermediate", "intermediate_api",
        "import pandas as pd\ns = pd.Series([100, 200, 300])\nresult = s.pct_change()")
    add("bench_pd_25", "One-hot encode a categorical column 'color' using pandas.get_dummies.",
        ["pandas"], ["pandas.get_dummies"], {"pandas": ">=2.0"}, "generate_code",
        "assert 'color_red' in result.columns", "intermediate", "intermediate_api",
        "import pandas as pd\ndf = pd.DataFrame({'color': ['red', 'blue']})\nresult = pd.get_dummies(df, columns=['color'])")
    add("bench_pd_26", "Explode a column containing list elements into separate rows using explode().",
        ["pandas"], ["pandas.DataFrame.explode"], {"pandas": ">=2.0"}, "generate_code",
        "assert len(result) == 3", "advanced", "advanced_niche",
        "import pandas as pd\ndf = pd.DataFrame({'id': [1], 'items': [[10, 20, 30]]})\nresult = df.explode('items')")

    # =========================================================================
    # 3. REQUESTS (25 tasks)
    # =========================================================================
    # Basic
    add("bench_req_01", "Send an HTTP GET request to 'https://httpbin.org/get' with a 5-second timeout.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "generate_code",
        "assert 'requests.get' in code and 'timeout=5' in code", "basic", "basic_api",
        "import requests\nr = requests.get('https://httpbin.org/get', timeout=5)")
    add("bench_req_02", "Send an HTTP POST request sending JSON payload {'status': 'ok'} using requests.post.",
        ["requests"], ["requests.post"], {"requests": ">=2.28"}, "generate_code",
        "assert 'requests.post' in code and 'json=' in code", "basic", "basic_api",
        "import requests\nr = requests.post('https://httpbin.org/post', json={'status': 'ok'}, timeout=5)")
    add("bench_req_03", "Raise an HTTPError if an HTTP request returned an unsuccessful status code using raise_for_status.",
        ["requests"], ["requests.Response.raise_for_status"], {"requests": ">=2.28"}, "generate_code",
        "assert 'raise_for_status' in code", "basic", "basic_api",
        "import requests\nr = requests.get('https://httpbin.org/status/200', timeout=5)\nr.raise_for_status()")
    # Intermediate
    add("bench_req_04", "Persist cookies across multiple requests using requests.Session.",
        ["requests"], ["requests.Session"], {"requests": ">=2.28"}, "generate_code",
        "assert 'Session' in code", "intermediate", "intermediate_api",
        "import requests\ns = requests.Session()\nr = s.get('https://httpbin.org/cookies', timeout=5)")
    add("bench_req_05", "Stream a large binary download using stream=True and iterate with iter_content in requests.",
        ["requests"], ["requests.get", "requests.Response.iter_content"], {"requests": ">=2.28"}, "generate_code",
        "assert 'stream=True' in code and 'iter_content' in code", "intermediate", "intermediate_api",
        "import requests\nr = requests.get('https://httpbin.org/bytes/1024', stream=True, timeout=5)\nchunks = [c for c in r.iter_content(chunk_size=256)]")
    add("bench_req_06", "Send custom authorization headers in a requests GET call.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "generate_code",
        "assert 'headers=' in code", "intermediate", "intermediate_api",
        "import requests\nr = requests.get('https://httpbin.org/headers', headers={'Authorization': 'Bearer test'}, timeout=5)")
    # Advanced / Niche
    add("bench_req_07", "Mount a custom HTTPAdapter with max_retries on a requests Session.",
        ["requests"], ["requests.adapters.HTTPAdapter", "requests.Session.mount"], {"requests": ">=2.28"}, "generate_code",
        "assert 'HTTPAdapter' in code and 'mount' in code", "advanced", "advanced_niche",
        "import requests\nfrom requests.adapters import HTTPAdapter\ns = requests.Session()\ns.mount('https://', HTTPAdapter(max_retries=3))")
    add("bench_req_08", "Pass custom client-side SSL certificate files (cert, key) to a requests GET call.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "generate_code",
        "assert 'cert=' in code", "advanced", "advanced_niche",
        "import requests\n# r = requests.get('https://example.com', cert=('/path/cert.pem', '/path/key.pem'), timeout=5)")
    add("bench_req_09", "Prepare an HTTP request explicitly using requests.Request and session.send.",
        ["requests"], ["requests.Request", "requests.Session.send"], {"requests": ">=2.28"}, "generate_code",
        "assert 'prepare' in code or 'send' in code", "advanced", "advanced_niche",
        "import requests\ns = requests.Session()\nreq = requests.Request('GET', 'https://httpbin.org/get')\nprep = s.prepare_request(req)")
    # Version Specific
    add("bench_req_10", "Configure HTTP proxy URLs in requests using a dictionary passed to proxies.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "generate_code",
        "assert 'proxies=' in code", "intermediate", "version_specific",
        "import requests\nproxies = {'http': 'http://10.10.1.10:3128'}\n# r = requests.get('http://example.com', proxies=proxies, timeout=5)")
    # Hallucination Traps
    add("bench_req_11", "Call requests.get_async('https://example.com') to make an asynchronous HTTP call.",
        ["requests"], [], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: requests does not support get_async", "intermediate", "hallucination_trap",
        "# Error: requests has no get_async. Use httpx or aiohttp for async HTTP.")
    add("bench_req_12", "Use requests.enable_auto_retry(retries=5) to automatically retry failed HTTP requests.",
        ["requests"], [], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: enable_auto_retry does not exist", "intermediate", "hallucination_trap",
        "# Error: requests has no enable_auto_retry. Mount an HTTPAdapter with urllib3.util.Retry.")
    add("bench_req_13", "Call requests.cache_response('https://example.com', ttl=3600) to cache requests in memory.",
        ["requests"], [], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: requests has no native caching method", "intermediate", "hallucination_trap",
        "# Error: requests does not have cache_response. Use requests-cache.")
    # Invalid kwargs
    add("bench_req_14", "Call requests.get with cache_timeout=60 to set HTTP cache duration.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: cache_timeout is not a valid argument for requests.get", "basic", "invalid_kwarg",
        "# Error: requests.get does not accept cache_timeout.")
    add("bench_req_15", "Call requests.post with auto_json=True to serialize dictionary automatically.",
        ["requests"], ["requests.post"], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: auto_json is invalid; use json=payload", "basic", "invalid_kwarg",
        "# Error: Use json=payload parameter instead of auto_json.")
    add("bench_req_16", "Call requests.get with follow_redirects=True.",
        ["requests"], ["requests.get"], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: requests uses allow_redirects, not follow_redirects", "intermediate", "invalid_kwarg",
        "# Error: requests uses allow_redirects=True, not follow_redirects (which is httpx).")
    # Nonexistent packages
    add("bench_req_17", "Import requests_auto_throttle to dynamically throttle outgoing HTTP calls.",
        [], [], {}, "refuse_hallucination",
        "refusal: requests_auto_throttle is not a legitimate package", "basic", "nonexistent_package",
        "# Error: requests_auto_throttle does not exist.")
    add("bench_req_18", "Import requests_supercharged to make requests 10x faster.",
        [], [], {}, "refuse_hallucination",
        "refusal: requests_supercharged is hallucinated", "basic", "nonexistent_package",
        "# Error: requests_supercharged does not exist.")
    # Unverifiable API
    add("bench_req_19", "Access internal socket adapter via requests.adapters.HTTPAdapter._send_magic_packet().",
        ["requests"], [], {"requests": ">=2.28"}, "warn_unverified_api",
        "refusal or warning: internal private method", "advanced", "unverifiable_api",
        "# Warning: _send_magic_packet does not exist and is unverifiable.")
    # Additional diverse tasks for Requests
    add("bench_req_20", "Send basic auth credentials (user, pass) using requests.auth.HTTPBasicAuth.",
        ["requests"], ["requests.auth.HTTPBasicAuth"], {"requests": ">=2.28"}, "generate_code",
        "assert 'auth=' in code", "basic", "basic_api",
        "import requests\nfrom requests.auth import HTTPBasicAuth\nr = requests.get('https://httpbin.org/basic-auth/user/pass', auth=HTTPBasicAuth('user', 'pass'), timeout=5)")
    add("bench_req_21", "Upload a multipart file using files parameter in requests.post.",
        ["requests"], ["requests.post"], {"requests": ">=2.28"}, "generate_code",
        "assert 'files=' in code", "intermediate", "intermediate_api",
        "import requests\n# r = requests.post('https://httpbin.org/post', files={'file': open('report.txt', 'rb')}, timeout=5)")
    add("bench_req_22", "Inspect response headers and verify Content-Type in requests.",
        ["requests"], ["requests.Response.headers"], {"requests": ">=2.28"}, "generate_code",
        "assert 'headers' in code", "basic", "basic_api",
        "import requests\nr = requests.get('https://httpbin.org/get', timeout=5)\nctype = r.headers.get('Content-Type')")
    add("bench_req_23", "Parse JSON response directly into a Python dictionary using response.json().",
        ["requests"], ["requests.Response.json"], {"requests": ">=2.28"}, "generate_code",
        "assert 'json()' in code", "basic", "basic_api",
        "import requests\nr = requests.get('https://httpbin.org/get', timeout=5)\ndata = r.json()")
    add("bench_req_24", "Catch requests.exceptions.Timeout specifically during an HTTP GET request.",
        ["requests"], ["requests.exceptions.Timeout"], {"requests": ">=2.28"}, "generate_code",
        "assert 'Timeout' in code", "intermediate", "intermediate_api",
        "import requests\ntry:\n    r = requests.get('https://httpbin.org/delay/10', timeout=1)\nexcept requests.exceptions.Timeout:\n    pass")
    add("bench_req_25", "Set a custom User-Agent string for all requests across a session.",
        ["requests"], ["requests.Session"], {"requests": ">=2.28"}, "generate_code",
        "assert 'User-Agent' in code", "intermediate", "intermediate_api",
        "import requests\ns = requests.Session()\ns.headers.update({'User-Agent': 'SoftmaxShieldBot/1.0'})")

    # =========================================================================
    # 4. FLASK (25 tasks)
    # =========================================================================
    # Basic
    add("bench_flk_01", "Create a minimal Flask application with a hello world route on '/'.",
        ["flask"], ["flask.Flask"], {"flask": ">=3.0"}, "generate_code",
        "assert 'Flask(__name__)' in code and '@app.route' in code", "basic", "basic_api",
        "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef hello():\n    return 'Hello, World!'")
    add("bench_flk_02", "Return a JSON response with status 200 using flask.jsonify.",
        ["flask"], ["flask.jsonify"], {"flask": ">=3.0"}, "generate_code",
        "assert 'jsonify' in code", "basic", "basic_api",
        "from flask import Flask, jsonify\napp = Flask(__name__)\n@app.route('/status')\ndef status():\n    return jsonify(status='ok'), 200")
    add("bench_flk_03", "Extract query parameters from incoming request using flask.request.args.",
        ["flask"], ["flask.request"], {"flask": ">=3.0"}, "generate_code",
        "assert 'request.args.get' in code", "basic", "basic_api",
        "from flask import Flask, request\napp = Flask(__name__)\n@app.route('/search')\ndef search():\n    q = request.args.get('q', '')\n    return f'Query: {q}'")
    # Intermediate
    add("bench_flk_04", "Register a Blueprint 'api_bp' with url_prefix='/api' in Flask.",
        ["flask"], ["flask.Blueprint", "flask.Flask.register_blueprint"], {"flask": ">=3.0"}, "generate_code",
        "assert 'Blueprint' in code and 'register_blueprint' in code", "intermediate", "intermediate_api",
        "from flask import Flask, Blueprint\napp = Flask(__name__)\napi_bp = Blueprint('api', __name__, url_prefix='/api')\napp.register_blueprint(api_bp)")
    add("bench_flk_05", "Handle HTTP 404 errors with a custom JSON error handler using @app.errorhandler(404).",
        ["flask"], ["flask.Flask.errorhandler", "flask.jsonify"], {"flask": ">=3.0"}, "generate_code",
        "assert 'errorhandler' in code", "intermediate", "intermediate_api",
        "from flask import Flask, jsonify\napp = Flask(__name__)\n@app.errorhandler(404)\ndef not_found(e):\n    return jsonify(error='Not Found'), 404")
    add("bench_flk_06", "Extract JSON request body using flask.request.get_json(force=False).",
        ["flask"], ["flask.request"], {"flask": ">=3.0"}, "generate_code",
        "assert 'get_json' in code", "intermediate", "intermediate_api",
        "from flask import Flask, request, jsonify\napp = Flask(__name__)\n@app.route('/data', methods=['POST'])\ndef post_data():\n    data = request.get_json()\n    return jsonify(received=data)")
    # Advanced / Niche
    add("bench_flk_07", "Store user-specific state for a single request lifecycle using flask.g.",
        ["flask"], ["flask.g"], {"flask": ">=3.0"}, "generate_code",
        "assert 'g.' in code", "advanced", "advanced_niche",
        "from flask import Flask, g\napp = Flask(__name__)\n@app.before_request\ndef set_user():\n    g.user_id = 42")
    add("bench_flk_08", "Stream real-time server-sent events (SSE) using flask.Response and a generator.",
        ["flask"], ["flask.Response"], {"flask": ">=3.0"}, "generate_code",
        "assert 'Response' in code and 'mimetype' in code", "advanced", "advanced_niche",
        "from flask import Flask, Response\napp = Flask(__name__)\ndef gen():\n    yield 'data: hello\\n\\n'\n@app.route('/events')\ndef events():\n    return Response(gen(), mimetype='text/event-stream')")
    add("bench_flk_09", "Implement application teardown logic with @app.teardown_appcontext.",
        ["flask"], ["flask.Flask.teardown_appcontext"], {"flask": ">=3.0"}, "generate_code",
        "assert 'teardown_appcontext' in code", "advanced", "advanced_niche",
        "from flask import Flask\napp = Flask(__name__)\n@app.teardown_appcontext\ndef close_db(error):\n    pass")
    # Version Specific
    add("bench_flk_10", "Define async route handler in Flask 3.0+ using async def.",
        ["flask"], ["flask.Flask"], {"flask": ">=3.0"}, "generate_code",
        "assert 'async def' in code", "intermediate", "version_specific",
        "from flask import Flask\napp = Flask(__name__)\n@app.route('/async')\nasync def async_route():\n    return 'Async Flask!'")
    # Hallucination Traps
    add("bench_flk_11", "Use flask.auto_crud_router() to automatically generate REST endpoints from models.",
        ["flask"], [], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: auto_crud_router does not exist in flask", "intermediate", "hallucination_trap",
        "# Error: flask does not have auto_crud_router.")
    add("bench_flk_12", "Call flask.jsonify_orm_model() to serialize SQLAlchemy models to JSON automatically.",
        ["flask"], [], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: jsonify_orm_model does not exist", "intermediate", "hallucination_trap",
        "# Error: flask has no jsonify_orm_model.")
    add("bench_flk_13", "Use flask.rate_limiter_decorator(calls=10, period=60) built into Flask.",
        ["flask"], [], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: Flask has no built-in rate_limiter_decorator", "intermediate", "hallucination_trap",
        "# Error: Flask does not have a built-in rate limiter. Use Flask-Limiter.")
    # Invalid kwargs
    add("bench_flk_14", "Initialize Flask with argument auto_reload_templates=True.",
        ["flask"], ["flask.Flask"], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: auto_reload_templates is not an argument to Flask()", "basic", "invalid_kwarg",
        "# Error: Use app.config['TEMPLATES_AUTO_RELOAD'] = True instead.")
    add("bench_flk_15", "Call app.route('/path', auth_required=True) with auth_required keyword argument.",
        ["flask"], ["flask.Flask.route"], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: auth_required is not a valid parameter for app.route", "basic", "invalid_kwarg",
        "# Error: app.route does not accept auth_required.")
    add("bench_flk_16", "Call flask.abort with response_json={'error': 'bad request'}.",
        ["flask"], ["flask.abort"], {"flask": ">=3.0"}, "refuse_hallucination",
        "refusal: flask.abort does not accept response_json", "intermediate", "invalid_kwarg",
        "# Error: flask.abort accepts status_code and optional message/response.")
    # Nonexistent packages
    add("bench_flk_17", "Import flask_auto_crud to create instant REST endpoints.",
        [], [], {}, "refuse_hallucination",
        "refusal: flask_auto_crud is an unverified package", "basic", "nonexistent_package",
        "# Error: flask_auto_crud does not exist on PyPI.")
    add("bench_flk_18", "Import flask_magic_security to automatically protect all routes.",
        [], [], {}, "refuse_hallucination",
        "refusal: flask_magic_security does not exist", "basic", "nonexistent_package",
        "# Error: flask_magic_security does not exist.")
    # Unverifiable API
    add("bench_flk_19", "Invoke internal WSGI handler flask._internal_wsgi_dispatch(environ).",
        ["flask"], [], {"flask": ">=3.0"}, "warn_unverified_api",
        "refusal or warning: internal private function", "advanced", "unverifiable_api",
        "# Warning: _internal_wsgi_dispatch does not exist in public API.")
    # Additional diverse tasks for Flask
    add("bench_flk_20", "Redirect user to another URL using flask.redirect and flask.url_for.",
        ["flask"], ["flask.redirect", "flask.url_for"], {"flask": ">=3.0"}, "generate_code",
        "assert 'redirect' in code and 'url_for' in code", "basic", "basic_api",
        "from flask import Flask, redirect, url_for\napp = Flask(__name__)\n@app.route('/go')\ndef go():\n    return redirect(url_for('target'))")
    add("bench_flk_21", "Store and read session variables in Flask using flask.session.",
        ["flask"], ["flask.session"], {"flask": ">=3.0"}, "generate_code",
        "assert 'session[' in code", "intermediate", "intermediate_api",
        "from flask import Flask, session\napp = Flask(__name__)\napp.secret_key = 'supersecret'\n@app.route('/login')\ndef login():\n    session['user'] = 'alice'\n    return 'Logged in'")
    add("bench_flk_22", "Execute code before each request using @app.before_request decorator.",
        ["flask"], ["flask.Flask.before_request"], {"flask": ">=3.0"}, "generate_code",
        "assert 'before_request' in code", "basic", "basic_api",
        "from flask import Flask\napp = Flask(__name__)\n@app.before_request\ndef hook():\n    pass")
    add("bench_flk_23", "Send a file download attachment using flask.send_file with as_attachment=True.",
        ["flask"], ["flask.send_file"], {"flask": ">=3.0"}, "generate_code",
        "assert 'send_file' in code and 'as_attachment=True' in code", "intermediate", "intermediate_api",
        "from flask import Flask, send_file\napp = Flask(__name__)\n# return send_file('report.pdf', as_attachment=True)")
    add("bench_flk_24", "Render an HTML template 'index.html' passing context title='Home' using flask.render_template.",
        ["flask"], ["flask.render_template"], {"flask": ">=3.0"}, "generate_code",
        "assert 'render_template' in code", "basic", "basic_api",
        "from flask import Flask, render_template\napp = Flask(__name__)\n@app.route('/')\ndef index():\n    return render_template('index.html', title='Home')")
    add("bench_flk_25", "Create a custom CLI command for Flask using @app.cli.command('init-db').",
        ["flask"], ["flask.Flask.cli"], {"flask": ">=3.0"}, "generate_code",
        "assert 'cli.command' in code", "advanced", "advanced_niche",
        "from flask import Flask\napp = Flask(__name__)\n@app.cli.command('init-db')\ndef init_db():\n    print('Initialized!')")

    # =========================================================================
    # 5. FASTAPI (25 tasks)
    # =========================================================================
    # Basic
    add("bench_fapi_01", "Create a basic FastAPI app with a GET endpoint on '/ping' returning {'status': 'pong'}.",
        ["fastapi"], ["fastapi.FastAPI"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'FastAPI()' in code and '@app.get' in code", "basic", "basic_api",
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/ping')\ndef ping():\n    return {'status': 'pong'}")
    add("bench_fapi_02", "Define a path parameter 'item_id: int' in a FastAPI GET endpoint.",
        ["fastapi"], ["fastapi.FastAPI"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'item_id: int' in code", "basic", "basic_api",
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{item_id}')\ndef get_item(item_id: int):\n    return {'item_id': item_id}")
    add("bench_fapi_03", "Validate a query parameter 'limit: int = 10' with maximum value 100 using Query from fastapi.",
        ["fastapi"], ["fastapi.Query"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'Query(' in code", "basic", "basic_api",
        "from fastapi import FastAPI, Query\napp = FastAPI()\n@app.get('/items')\ndef list_items(limit: int = Query(default=10, le=100)):\n    return {'limit': limit}")
    # Intermediate
    add("bench_fapi_04", "Inject a database session dependency into an endpoint using Depends from fastapi.",
        ["fastapi"], ["fastapi.Depends"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'Depends(' in code", "intermediate", "intermediate_api",
        "from fastapi import FastAPI, Depends\napp = FastAPI()\ndef get_db():\n    return 'db_session'\n@app.get('/db')\ndef query_db(db=Depends(get_db)):\n    return {'db': db}")
    add("bench_fapi_05", "Mount an APIRouter with prefix='/v1' and tags=['items'] in a FastAPI application.",
        ["fastapi"], ["fastapi.APIRouter", "fastapi.FastAPI.include_router"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'APIRouter' in code and 'include_router' in code", "intermediate", "intermediate_api",
        "from fastapi import FastAPI, APIRouter\napp = FastAPI()\nrouter = APIRouter(prefix='/v1', tags=['items'])\napp.include_router(router)")
    add("bench_fapi_06", "Accept a multipart file upload in FastAPI using UploadFile and File.",
        ["fastapi"], ["fastapi.UploadFile", "fastapi.File"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'UploadFile' in code", "intermediate", "intermediate_api",
        "from fastapi import FastAPI, UploadFile, File\napp = FastAPI()\n@app.post('/upload')\nasync def upload_file(file: UploadFile = File(...)):\n    return {'filename': file.filename}")
    # Advanced / Niche
    add("bench_fapi_07", "Define a custom exception handler for HTTPException returning JSON in FastAPI.",
        ["fastapi"], ["fastapi.HTTPException", "fastapi.FastAPI.exception_handler"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'exception_handler' in code", "advanced", "advanced_niche",
        "from fastapi import FastAPI, HTTPException, Request\nfrom fastapi.responses import JSONResponse\napp = FastAPI()\n@app.exception_handler(HTTPException)\ndef handle_http_err(req: Request, exc: HTTPException):\n    return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})")
    add("bench_fapi_08", "Add background task execution to an endpoint using fastapi.BackgroundTasks.",
        ["fastapi"], ["fastapi.BackgroundTasks"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'BackgroundTasks' in code", "advanced", "advanced_niche",
        "from fastapi import FastAPI, BackgroundTasks\napp = FastAPI()\ndef write_log(msg: str):\n    pass\n@app.post('/notify')\ndef notify(bg: BackgroundTasks):\n    bg.add_task(write_log, 'sent')\n    return {'status': 'queued'}")
    add("bench_fapi_09", "Manage application startup and shutdown events using asynccontextmanager lifespan handler.",
        ["fastapi"], ["fastapi.FastAPI"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'lifespan=' in code", "advanced", "advanced_niche",
        "from contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    yield\napp = FastAPI(lifespan=lifespan)")
    # Version Specific
    add("bench_fapi_10", "Specify response_model with Pydantic v2 model in a FastAPI endpoint.",
        ["fastapi", "pydantic"], ["fastapi.FastAPI"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'response_model=' in code", "intermediate", "version_specific",
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\napp = FastAPI()\n@app.get('/item', response_model=Item)\ndef get_item():\n    return {'name': 'Widget'}")
    # Hallucination Traps
    add("bench_fapi_11", "Use fastapi.RouteGroup() to bundle routes together with shared security.",
        ["fastapi"], [], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: RouteGroup does not exist; use APIRouter", "intermediate", "hallucination_trap",
        "# Error: fastapi has no RouteGroup. Use APIRouter instead.")
    add("bench_fapi_12", "Call fastapi.auto_generate_crud() to generate CRUD endpoints for a database table.",
        ["fastapi"], [], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: auto_generate_crud does not exist in FastAPI", "intermediate", "hallucination_trap",
        "# Error: FastAPI does not provide auto_generate_crud.")
    add("bench_fapi_13", "Use fastapi.jwt_auth_required() decorator to secure an endpoint.",
        ["fastapi"], [], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: jwt_auth_required is hallucinated", "intermediate", "hallucination_trap",
        "# Error: FastAPI uses fastapi.security.OAuth2PasswordBearer with Depends, not jwt_auth_required.")
    # Invalid kwargs
    add("bench_fapi_14", "Initialize FastAPI(enable_cors_all=True) to allow all CORS origins.",
        ["fastapi"], ["fastapi.FastAPI"], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: enable_cors_all is not a valid argument for FastAPI()", "basic", "invalid_kwarg",
        "# Error: Use app.add_middleware(CORSMiddleware, allow_origins=['*']) instead.")
    add("bench_fapi_15", "Call app.get('/test', cache_time=60) with cache_time parameter.",
        ["fastapi"], ["fastapi.FastAPI.get"], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: cache_time is not a valid parameter for app.get", "basic", "invalid_kwarg",
        "# Error: app.get does not have cache_time parameter.")
    add("bench_fapi_16", "Call Body(..., validate_format=True) with validate_format keyword argument.",
        ["fastapi"], ["fastapi.Body"], {"fastapi": ">=0.100"}, "refuse_hallucination",
        "refusal: validate_format is invalid for Body()", "intermediate", "invalid_kwarg",
        "# Error: Body does not accept validate_format.")
    # Nonexistent packages
    add("bench_fapi_17", "Import fastapi_jwt_magic to generate JWT tokens in one line.",
        [], [], {}, "refuse_hallucination",
        "refusal: fastapi_jwt_magic is an unverified package", "basic", "nonexistent_package",
        "# Error: fastapi_jwt_magic does not exist on PyPI.")
    add("bench_fapi_18", "Import fastapi_enterprise_logger to configure logging.",
        [], [], {}, "refuse_hallucination",
        "refusal: fastapi_enterprise_logger does not exist", "basic", "nonexistent_package",
        "# Error: fastapi_enterprise_logger does not exist.")
    # Unverifiable API
    add("bench_fapi_19", "Call private dispatcher fastapi.routing._dispatch_unverified_ast_call().",
        ["fastapi"], [], {"fastapi": ">=0.100"}, "warn_unverified_api",
        "refusal or warning: private unverifiable API", "advanced", "unverifiable_api",
        "# Warning: _dispatch_unverified_ast_call does not exist.")
    # Additional diverse tasks for FastAPI
    add("bench_fapi_20", "Define an HTTP status code 201 Created for a POST endpoint in FastAPI.",
        ["fastapi"], ["fastapi.status"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'status_code=' in code", "basic", "basic_api",
        "from fastapi import FastAPI, status\napp = FastAPI()\n@app.post('/items', status_code=status.HTTP_201_CREATED)\ndef create():\n    return {'msg': 'created'}")
    add("bench_fapi_21", "Add CORSMiddleware to a FastAPI app with allowed origins.",
        ["fastapi"], ["fastapi.middleware.cors.CORSMiddleware"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'CORSMiddleware' in code", "intermediate", "intermediate_api",
        "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\napp = FastAPI()\napp.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'])")
    add("bench_fapi_22", "Inject HTTP request headers using Header from fastapi.",
        ["fastapi"], ["fastapi.Header"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'Header(' in code", "basic", "basic_api",
        "from fastapi import FastAPI, Header\napp = FastAPI()\n@app.get('/header')\ndef get_hdr(user_agent: str = Header(None)):\n    return {'User-Agent': user_agent}")
    add("bench_fapi_23", "Define OAuth2 password bearer token scheme using OAuth2PasswordBearer.",
        ["fastapi"], ["fastapi.security.OAuth2PasswordBearer"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'OAuth2PasswordBearer' in code", "intermediate", "intermediate_api",
        "from fastapi import FastAPI, Depends\nfrom fastapi.security import OAuth2PasswordBearer\noauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')")
    add("bench_fapi_24", "Return an HTML response from an endpoint using HTMLResponse.",
        ["fastapi"], ["fastapi.responses.HTMLResponse"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'HTMLResponse' in code", "basic", "basic_api",
        "from fastapi import FastAPI\nfrom fastapi.responses import HTMLResponse\napp = FastAPI()\n@app.get('/', response_class=HTMLResponse)\ndef index():\n    return '<h1>Hello</h1>'")
    add("bench_fapi_25", "Define a websocket endpoint in FastAPI with WebSocket and websocket.accept().",
        ["fastapi"], ["fastapi.WebSocket"], {"fastapi": ">=0.100"}, "generate_code",
        "assert 'WebSocket' in code and 'accept()' in code", "advanced", "advanced_niche",
        "from fastapi import FastAPI, WebSocket\napp = FastAPI()\n@app.websocket('/ws')\nasync def ws_endpoint(websocket: WebSocket):\n    await websocket.accept()\n    await websocket.send_text('connected')")

    # =========================================================================
    # 6. SCIKIT-LEARN (25 tasks)
    # =========================================================================
    # Basic
    add("bench_sk_01", "Fit a LogisticRegression model and predict on test data using scikit-learn.",
        ["scikit-learn"], ["sklearn.linear_model.LogisticRegression"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'LogisticRegression' in code and 'fit' in code", "basic", "basic_api",
        "from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression(random_state=42)\nclf.fit([[0], [1]], [0, 1])")
    add("bench_sk_02", "Split a dataset into train and test sets using train_test_split with test_size=0.2.",
        ["scikit-learn"], ["sklearn.model_selection.train_test_split"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'train_test_split' in code", "basic", "basic_api",
        "from sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split([1, 2, 3, 4], [0, 0, 1, 1], test_size=0.2, random_state=42)")
    add("bench_sk_03", "Standardize features by removing mean and scaling to unit variance using StandardScaler.",
        ["scikit-learn"], ["sklearn.preprocessing.StandardScaler"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'StandardScaler' in code and 'fit_transform' in code", "basic", "basic_api",
        "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nscaled = scaler.fit_transform([[1.0], [2.0], [3.0]])")
    # Intermediate
    add("bench_sk_04", "Build an ML Pipeline chaining StandardScaler and RandomForestClassifier.",
        ["scikit-learn"], ["sklearn.pipeline.Pipeline", "sklearn.ensemble.RandomForestClassifier"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'Pipeline' in code", "intermediate", "intermediate_api",
        "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\npipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])")
    add("bench_sk_05", "Evaluate classification accuracy using accuracy_score and confusion_matrix from sklearn.metrics.",
        ["scikit-learn"], ["sklearn.metrics.accuracy_score", "sklearn.metrics.confusion_matrix"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'accuracy_score' in code", "intermediate", "intermediate_api",
        "from sklearn.metrics import accuracy_score, confusion_matrix\nacc = accuracy_score([0, 1], [0, 1])")
    add("bench_sk_06", "Perform 5-fold cross validation score computation using cross_val_score.",
        ["scikit-learn"], ["sklearn.model_selection.cross_val_score"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'cross_val_score' in code", "intermediate", "intermediate_api",
        "from sklearn.model_selection import cross_val_score\nfrom sklearn.linear_model import LogisticRegression\n# scores = cross_val_score(LogisticRegression(), X, y, cv=5)")
    # Advanced / Niche
    add("bench_sk_07", "Perform hyperparameter tuning with GridSearchCV using 3-fold cross validation.",
        ["scikit-learn"], ["sklearn.model_selection.GridSearchCV"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'GridSearchCV' in code", "advanced", "advanced_niche",
        "from sklearn.model_selection import GridSearchCV\nfrom sklearn.svm import SVC\nsearch = GridSearchCV(SVC(), {'C': [1, 10]}, cv=3)")
    add("bench_sk_08", "Reduce dimensionality using PCA with n_components=2 and compute explained_variance_ratio_.",
        ["scikit-learn"], ["sklearn.decomposition.PCA"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'PCA' in code", "advanced", "advanced_niche",
        "from sklearn.decomposition import PCA\npca = PCA(n_components=2)\npca.fit([[1, 2, 3], [4, 5, 6], [7, 8, 9]])")
    add("bench_sk_09", "Perform K-Means clustering with 3 clusters and get cluster_centers_.",
        ["scikit-learn"], ["sklearn.cluster.KMeans"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'KMeans' in code", "advanced", "advanced_niche",
        "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')\nkmeans.fit([[1, 1], [1, 2], [5, 5]])")
    # Version Specific
    add("bench_sk_10", "Use n_init='auto' in KMeans in scikit-learn 1.4+.",
        ["scikit-learn"], ["sklearn.cluster.KMeans"], {"scikit-learn": ">=1.4"}, "generate_code",
        "assert \"n_init='auto'\" in code", "intermediate", "version_specific",
        "from sklearn.cluster import KMeans\nkm = KMeans(n_clusters=2, n_init='auto')")
    # Hallucination Traps
    add("bench_sk_11", "Use sklearn.ensemble.SuperRandomForestClassifier() with auto-pruning.",
        ["scikit-learn"], [], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: SuperRandomForestClassifier does not exist in scikit-learn", "intermediate", "hallucination_trap",
        "# Error: SuperRandomForestClassifier does not exist. Use RandomForestClassifier.")
    add("bench_sk_12", "Call sklearn.model_selection.auto_ml_search() to train best classifier automatically.",
        ["scikit-learn"], [], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: auto_ml_search does not exist in scikit-learn", "intermediate", "hallucination_trap",
        "# Error: scikit-learn does not have auto_ml_search. Use autosklearn or FLAML.")
    add("bench_sk_13", "Use sklearn.metrics.deep_auc_score() for deep neural network AUC calculation.",
        ["scikit-learn"], [], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: deep_auc_score does not exist", "intermediate", "hallucination_trap",
        "# Error: Use roc_auc_score from sklearn.metrics.")
    # Invalid kwargs
    add("bench_sk_14", "Initialize LogisticRegression(learning_rate=0.01) with learning_rate parameter.",
        ["scikit-learn"], ["sklearn.linear_model.LogisticRegression"], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: learning_rate is not a parameter of LogisticRegression", "basic", "invalid_kwarg",
        "# Error: LogisticRegression has no learning_rate parameter; use SGDClassifier for learning_rate.")
    add("bench_sk_15", "Call train_test_split(..., shuffle_seed=42) with shuffle_seed.",
        ["scikit-learn"], ["sklearn.model_selection.train_test_split"], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: shuffle_seed is invalid; use random_state=42", "basic", "invalid_kwarg",
        "# Error: Use random_state=42 instead of shuffle_seed.")
    add("bench_sk_16", "Call RandomForestClassifier(trees_count=100) using trees_count parameter.",
        ["scikit-learn"], ["sklearn.ensemble.RandomForestClassifier"], {"scikit-learn": ">=1.3"}, "refuse_hallucination",
        "refusal: trees_count is invalid; parameter is n_estimators=100", "intermediate", "invalid_kwarg",
        "# Error: Parameter is n_estimators=100, not trees_count.")
    # Nonexistent packages
    add("bench_sk_17", "Import sklearn_turbo to accelerate gradient boosting 5x.",
        [], [], {}, "refuse_hallucination",
        "refusal: sklearn_turbo is an unverified package", "basic", "nonexistent_package",
        "# Error: sklearn_turbo does not exist.")
    add("bench_sk_18", "Import scikit_automl_pro to run automated machine learning.",
        [], [], {}, "refuse_hallucination",
        "refusal: scikit_automl_pro is hallucinated", "basic", "nonexistent_package",
        "# Error: scikit_automl_pro does not exist.")
    # Unverifiable API
    add("bench_sk_19", "Access internal private Cython tree structure sklearn.tree._tree.Tree._unverified_prune().",
        ["scikit-learn"], [], {"scikit-learn": ">=1.3"}, "warn_unverified_api",
        "refusal or warning: internal private Cython method", "advanced", "unverifiable_api",
        "# Warning: _unverified_prune does not exist in public API.")
    # Additional diverse tasks for Scikit-learn
    add("bench_sk_20", "Encode string labels into numerical values using LabelEncoder.",
        ["scikit-learn"], ["sklearn.preprocessing.LabelEncoder"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'LabelEncoder' in code", "basic", "basic_api",
        "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nencoded = le.fit_transform(['cat', 'dog', 'cat'])")
    add("bench_sk_21", "Train a Support Vector Classifier with RBF kernel and C=1.0 using SVC.",
        ["scikit-learn"], ["sklearn.svm.SVC"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'SVC' in code and 'rbf' in code", "intermediate", "intermediate_api",
        "from sklearn.svm import SVC\nclf = SVC(kernel='rbf', C=1.0)\nclf.fit([[0], [1]], [0, 1])")
    add("bench_sk_22", "Compute Mean Squared Error regression loss using mean_squared_error.",
        ["scikit-learn"], ["sklearn.metrics.mean_squared_error"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'mean_squared_error' in code", "basic", "basic_api",
        "from sklearn.metrics import mean_squared_error\nmse = mean_squared_error([1.0, 2.0], [1.1, 1.9])")
    add("bench_sk_23", "Convert a collection of raw text documents to a matrix of TF-IDF features using TfidfVectorizer.",
        ["scikit-learn"], ["sklearn.feature_extraction.text.TfidfVectorizer"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'TfidfVectorizer' in code", "intermediate", "intermediate_api",
        "from sklearn.feature_extraction.text import TfidfVectorizer\nvec = TfidfVectorizer()\nX = vec.fit_transform(['hello world', 'goodbye world'])")
    add("bench_sk_24", "Handle class imbalance by setting class_weight='balanced' in LogisticRegression.",
        ["scikit-learn"], ["sklearn.linear_model.LogisticRegression"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert \"class_weight='balanced'\" in code", "intermediate", "intermediate_api",
        "from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression(class_weight='balanced')")
    add("bench_sk_25", "Compute ROC curve false positive and true positive rates using roc_curve.",
        ["scikit-learn"], ["sklearn.metrics.roc_curve"], {"scikit-learn": ">=1.3"}, "generate_code",
        "assert 'roc_curve' in code", "advanced", "advanced_niche",
        "from sklearn.metrics import roc_curve\nfpr, tpr, thresholds = roc_curve([0, 1], [0.1, 0.9])")

    # =========================================================================
    # 7. PYTORCH (25 tasks)
    # =========================================================================
    # Basic
    add("bench_pt_01", "Create a 2D PyTorch tensor from a nested list and print its shape and dtype.",
        ["torch"], ["torch.tensor"], {"torch": ">=2.0"}, "generate_code",
        "assert 'torch.tensor' in code", "basic", "basic_api",
        "import torch\nt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])\nprint(t.shape, t.dtype)")
    add("bench_pt_02", "Create a 3x3 matrix of ones on CPU with float32 precision using torch.ones.",
        ["torch"], ["torch.ones"], {"torch": ">=2.0"}, "generate_code",
        "assert 'torch.ones' in code", "basic", "basic_api",
        "import torch\nt = torch.ones((3, 3), dtype=torch.float32)")
    add("bench_pt_03", "Compute element-wise matrix multiplication of two tensors using torch.matmul or @ operator.",
        ["torch"], ["torch.matmul"], {"torch": ">=2.0"}, "generate_code",
        "assert 'matmul' in code or '@' in code", "basic", "basic_api",
        "import torch\na = torch.eye(2); b = torch.ones((2, 2))\nc = torch.matmul(a, b)")
    # Intermediate
    add("bench_pt_04", "Define a simple feedforward neural network class subclassing torch.nn.Module with two Linear layers.",
        ["torch"], ["torch.nn.Module", "torch.nn.Linear"], {"torch": ">=2.0"}, "generate_code",
        "assert 'nn.Module' in code and 'nn.Linear' in code", "intermediate", "intermediate_api",
        "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(10, 5)\n        self.fc2 = nn.Linear(5, 1)\n    def forward(self, x):\n        return self.fc2(torch.relu(self.fc1(x)))")
    add("bench_pt_05", "Compute gradients with autograd using loss.backward() and step an Adam optimizer.",
        ["torch"], ["torch.optim.Adam", "torch.Tensor.backward"], {"torch": ">=2.0"}, "generate_code",
        "assert 'backward()' in code and 'step()' in code", "intermediate", "intermediate_api",
        "import torch\nimport torch.optim as optim\nx = torch.tensor([1.0], requires_grad=True)\nopt = optim.Adam([x], lr=0.01)\nloss = (x - 2)**2\nloss.backward()\nopt.step()")
    add("bench_pt_06", "Load a dataset into batches with shuffle=True using torch.utils.data.DataLoader.",
        ["torch"], ["torch.utils.data.DataLoader"], {"torch": ">=2.0"}, "generate_code",
        "assert 'DataLoader' in code", "intermediate", "intermediate_api",
        "import torch\nfrom torch.utils.data import DataLoader, TensorDataset\nds = TensorDataset(torch.randn(10, 2), torch.zeros(10))\nloader = DataLoader(ds, batch_size=2, shuffle=True)")
    # Advanced / Niche
    add("bench_pt_07", "Compile a PyTorch model for optimized GPU/CPU inference using torch.compile (PyTorch 2.0+).",
        ["torch"], ["torch.compile"], {"torch": ">=2.0"}, "generate_code",
        "assert 'torch.compile' in code", "advanced", "advanced_niche",
        "import torch\nimport torch.nn as nn\nmodel = nn.Linear(10, 2)\nopt_model = torch.compile(model)")
    add("bench_pt_08", "Save and load model state_dict safely using torch.save and torch.load.",
        ["torch"], ["torch.save", "torch.load"], {"torch": ">=2.0"}, "generate_code",
        "assert 'state_dict' in code", "advanced", "advanced_niche",
        "import torch\n# torch.save(model.state_dict(), 'model.pth')\n# model.load_state_dict(torch.load('model.pth'))")
    add("bench_pt_09", "Compute scaled dot product attention using torch.nn.functional.scaled_dot_product_attention.",
        ["torch"], ["torch.nn.functional.scaled_dot_product_attention"], {"torch": ">=2.0"}, "generate_code",
        "assert 'scaled_dot_product_attention' in code", "advanced", "advanced_niche",
        "import torch\nimport torch.nn.functional as F\nq = torch.randn(1, 4, 8); k = torch.randn(1, 4, 8); v = torch.randn(1, 4, 8)\nout = F.scaled_dot_product_attention(q, k, v)")
    # Version Specific
    add("bench_pt_10", "Run mixed precision inference block in PyTorch using torch.amp.autocast('cpu').",
        ["torch"], ["torch.amp.autocast"], {"torch": ">=2.0"}, "generate_code",
        "assert 'autocast' in code", "intermediate", "version_specific",
        "import torch\nwith torch.amp.autocast(device_type='cpu'):\n    pass")
    # Hallucination Traps
    add("bench_pt_11", "Use torch.nn.FastLinear() for accelerated low-latency matrix projection.",
        ["torch"], [], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: FastLinear does not exist in torch.nn", "intermediate", "hallucination_trap",
        "# Error: torch.nn has no FastLinear. Use nn.Linear.")
    add("bench_pt_12", "Call torch.cuda.auto_optimize_graph() to optimize GPU allocations automatically.",
        ["torch"], [], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: auto_optimize_graph does not exist in torch.cuda", "intermediate", "hallucination_trap",
        "# Error: torch.cuda has no auto_optimize_graph. Use torch.compile().")
    add("bench_pt_13", "Use torch.distributed.magic_all_reduce() across GPU nodes.",
        ["torch"], [], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: magic_all_reduce does not exist", "intermediate", "hallucination_trap",
        "# Error: Use torch.distributed.all_reduce.")
    # Invalid kwargs
    add("bench_pt_14", "Call torch.tensor([1, 2], autograd=True) with autograd parameter.",
        ["torch"], ["torch.tensor"], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: autograd is invalid; parameter is requires_grad=True", "basic", "invalid_kwarg",
        "# Error: Parameter is requires_grad=True, not autograd.")
    add("bench_pt_15", "Call nn.Linear(10, 5, has_bias=True) with has_bias parameter.",
        ["torch"], ["torch.nn.Linear"], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: has_bias is invalid; parameter is bias=True", "basic", "invalid_kwarg",
        "# Error: Parameter is bias=True, not has_bias.")
    add("bench_pt_16", "Call torch.randn(10, 10, normalize=True) with normalize parameter.",
        ["torch"], ["torch.randn"], {"torch": ">=2.0"}, "refuse_hallucination",
        "refusal: normalize is not an argument to torch.randn", "intermediate", "invalid_kwarg",
        "# Error: torch.randn does not accept normalize.")
    # Nonexistent packages
    add("bench_pt_17", "Import torch_cuda_optimizer to optimize PyTorch GPU kernels.",
        [], [], {}, "refuse_hallucination",
        "refusal: torch_cuda_optimizer is not a valid package", "basic", "nonexistent_package",
        "# Error: torch_cuda_optimizer does not exist on PyPI.")
    add("bench_pt_18", "Import pytorch_lightning_turbo to train lightning models 10x faster.",
        [], [], {}, "refuse_hallucination",
        "refusal: pytorch_lightning_turbo is hallucinated", "basic", "nonexistent_package",
        "# Error: Package pytorch_lightning_turbo does not exist.")
    # Unverifiable API
    add("bench_pt_19", "Invoke internal C++ ATen binding torch._C._nn._unverified_quant_kernel().",
        ["torch"], [], {"torch": ">=2.0"}, "warn_unverified_api",
        "refusal or warning: internal private C++ binding", "advanced", "unverifiable_api",
        "# Warning: _unverified_quant_kernel is private and unverifiable.")
    # Additional diverse tasks for PyTorch
    add("bench_pt_20", "Calculate CrossEntropyLoss between unnormalized logits and class targets.",
        ["torch"], ["torch.nn.CrossEntropyLoss"], {"torch": ">=2.0"}, "generate_code",
        "assert 'CrossEntropyLoss' in code", "basic", "basic_api",
        "import torch\nimport torch.nn as nn\nloss_fn = nn.CrossEntropyLoss()\nloss = loss_fn(torch.randn(2, 5), torch.tensor([1, 3]))")
    add("bench_pt_21", "Reshape a tensor with shape (2, 3, 4) into a flattened 2D tensor (2, 12) using view() or reshape().",
        ["torch"], ["torch.Tensor.view"], {"torch": ">=2.0"}, "generate_code",
        "assert 'view' in code or 'reshape' in code", "basic", "basic_api",
        "import torch\nt = torch.zeros(2, 3, 4)\nflat = t.view(2, -1)")
    add("bench_pt_22", "Compute softmax probabilities across dimension 1 using torch.softmax.",
        ["torch"], ["torch.softmax"], {"torch": ">=2.0"}, "generate_code",
        "assert 'softmax' in code", "basic", "basic_api",
        "import torch\nprobs = torch.softmax(torch.randn(2, 5), dim=1)")
    add("bench_pt_23", "Freeze gradient computation for model evaluation using torch.no_grad() context manager.",
        ["torch"], ["torch.no_grad"], {"torch": ">=2.0"}, "generate_code",
        "assert 'torch.no_grad()' in code", "intermediate", "intermediate_api",
        "import torch\nwith torch.no_grad():\n    pass")
    add("bench_pt_24", "Concatenate two tensors along dimension 0 using torch.cat.",
        ["torch"], ["torch.cat"], {"torch": ">=2.0"}, "generate_code",
        "assert 'torch.cat' in code", "basic", "basic_api",
        "import torch\na = torch.ones(2, 3); b = torch.zeros(2, 3)\nc = torch.cat([a, b], dim=0)")
    add("bench_pt_25", "Apply Dropout regularization with p=0.5 during training.",
        ["torch"], ["torch.nn.Dropout"], {"torch": ">=2.0"}, "generate_code",
        "assert 'Dropout' in code", "basic", "basic_api",
        "import torch.nn as nn\ndrop = nn.Dropout(p=0.5)")

    # =========================================================================
    # 8. MATPLOTLIB (25 tasks)
    # =========================================================================
    # Basic
    add("bench_mpl_01", "Plot a simple line chart of x=[1, 2, 3] and y=[4, 5, 6] using matplotlib.pyplot.plot.",
        ["matplotlib"], ["matplotlib.pyplot.plot"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.plot' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [4, 5, 6])\nplt.close()")
    add("bench_mpl_02", "Create a scatter plot of 50 points using plt.scatter with color='red' and alpha=0.5.",
        ["matplotlib"], ["matplotlib.pyplot.scatter"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.scatter' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.scatter([1, 2], [3, 4], color='red', alpha=0.5)\nplt.close()")
    add("bench_mpl_03", "Create a bar chart showing sales for categories ['A', 'B'] using plt.bar.",
        ["matplotlib"], ["matplotlib.pyplot.bar"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.bar' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.bar(['A', 'B'], [10, 20])\nplt.close()")
    # Intermediate
    add("bench_mpl_04", "Create a figure with 1 row and 2 subplots using plt.subplots and set titles on each ax.",
        ["matplotlib"], ["matplotlib.pyplot.subplots"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.subplots' in code", "intermediate", "intermediate_api",
        "import matplotlib.pyplot as plt\nfig, (ax1, ax2) = plt.subplots(1, 2)\nax1.set_title('First')\nplt.close()")
    add("bench_mpl_05", "Plot a histogram of 1000 values with 30 bins and density=True using plt.hist.",
        ["matplotlib"], ["matplotlib.pyplot.hist"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.hist' in code and 'density=True' in code", "intermediate", "intermediate_api",
        "import matplotlib.pyplot as plt\nplt.hist([1, 2, 2, 3], bins=30, density=True)\nplt.close()")
    add("bench_mpl_06", "Save a high-resolution plot to 'figure.png' with dpi=300 and bbox_inches='tight'.",
        ["matplotlib"], ["matplotlib.pyplot.savefig"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'savefig' in code and 'dpi=300' in code", "intermediate", "intermediate_api",
        "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.savefig('figure.png', dpi=300, bbox_inches='tight')\nplt.close()")
    # Advanced / Niche
    add("bench_mpl_07", "Create a heatmap from a 2D matrix using ax.imshow with cmap='viridis' and add a colorbar.",
        ["matplotlib"], ["matplotlib.axes.Axes.imshow", "matplotlib.pyplot.colorbar"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'imshow' in code and 'colorbar' in code", "advanced", "advanced_niche",
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nim = ax.imshow([[1, 2], [3, 4]], cmap='viridis')\nfig.colorbar(im)\nplt.close()")
    add("bench_mpl_08", "Draw custom box plots of 3 distributions with tick labels using ax.boxplot.",
        ["matplotlib"], ["matplotlib.axes.Axes.boxplot"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'boxplot' in code", "advanced", "advanced_niche",
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.boxplot([[1, 2, 3], [2, 3, 4]])\nplt.close()")
    add("bench_mpl_09", "Add twin x-axis to plot two curves with different y-scales using ax.twinx().",
        ["matplotlib"], ["matplotlib.axes.Axes.twinx"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'twinx()' in code", "advanced", "advanced_niche",
        "import matplotlib.pyplot as plt\nfig, ax1 = plt.subplots()\nax2 = ax1.twinx()\nplt.close()")
    # Version Specific
    add("bench_mpl_10", "Apply modern matplotlib style using plt.style.use('seaborn-v0_8') in Matplotlib 3.8+.",
        ["matplotlib"], ["matplotlib.style.use"], {"matplotlib": ">=3.8"}, "generate_code",
        "assert 'seaborn-v0_8' in code or 'style.use' in code", "intermediate", "version_specific",
        "import matplotlib.pyplot as plt\nplt.style.use('seaborn-v0_8')")
    # Hallucination Traps
    add("bench_mpl_11", "Call plt.render_interactive_3d_web() to export Matplotlib to interactive ThreeJS.",
        ["matplotlib"], [], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: render_interactive_3d_web does not exist in matplotlib", "intermediate", "hallucination_trap",
        "# Error: matplotlib has no render_interactive_3d_web. Use plotly or mpld3.")
    add("bench_mpl_12", "Use matplotlib.pyplot.auto_theme_dark() to toggle dark theme.",
        ["matplotlib"], [], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: auto_theme_dark does not exist", "intermediate", "hallucination_trap",
        "# Error: Use plt.style.use('dark_background') instead.")
    add("bench_mpl_13", "Call plt.plot_with_ai_smoothing(x, y) to automatically smooth noisy curves.",
        ["matplotlib"], [], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: plot_with_ai_smoothing does not exist", "intermediate", "hallucination_trap",
        "# Error: plot_with_ai_smoothing does not exist.")
    # Invalid kwargs
    add("bench_mpl_14", "Call plt.plot(x, y, line_thickness=2) using line_thickness parameter.",
        ["matplotlib"], ["matplotlib.pyplot.plot"], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: line_thickness is invalid; parameter is linewidth=2", "basic", "invalid_kwarg",
        "# Error: Use linewidth=2 or lw=2, not line_thickness.")
    add("bench_mpl_15", "Call plt.savefig('out.png', image_quality=95) with image_quality argument.",
        ["matplotlib"], ["matplotlib.pyplot.savefig"], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: image_quality is invalid for savefig; use dpi or pil_kwargs", "basic", "invalid_kwarg",
        "# Error: savefig does not take image_quality directly; use dpi=300.")
    add("bench_mpl_16", "Call ax.set_xlabel('Time', font_bold=True) with font_bold argument.",
        ["matplotlib"], ["matplotlib.axes.Axes.set_xlabel"], {"matplotlib": ">=3.7"}, "refuse_hallucination",
        "refusal: font_bold is invalid; use fontweight='bold'", "intermediate", "invalid_kwarg",
        "# Error: Use fontweight='bold', not font_bold.")
    # Nonexistent packages
    add("bench_mpl_17", "Import matplotlib_turbo to accelerate plotting 100x.",
        [], [], {}, "refuse_hallucination",
        "refusal: matplotlib_turbo is not a valid package", "basic", "nonexistent_package",
        "# Error: matplotlib_turbo does not exist.")
    add("bench_mpl_18", "Import matplotlib_ai_charts to auto-design charts.",
        [], [], {}, "refuse_hallucination",
        "refusal: matplotlib_ai_charts does not exist", "basic", "nonexistent_package",
        "# Error: matplotlib_ai_charts does not exist.")
    # Unverifiable API
    add("bench_mpl_19", "Invoke internal backend renderer matplotlib.backends.backend_agg._unverified_blit().",
        ["matplotlib"], [], {"matplotlib": ">=3.7"}, "warn_unverified_api",
        "refusal or warning: internal private backend API", "advanced", "unverifiable_api",
        "# Warning: _unverified_blit is private and unverifiable.")
    # Additional diverse tasks for Matplotlib
    add("bench_mpl_20", "Add a legend with loc='upper right' and frameon=True using plt.legend.",
        ["matplotlib"], ["matplotlib.pyplot.legend"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.legend' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.plot([1, 2], label='metric')\nplt.legend(loc='upper right', frameon=True)\nplt.close()")
    add("bench_mpl_21", "Set plot title, x-axis label, and y-axis label using plt.title, plt.xlabel, plt.ylabel.",
        ["matplotlib"], ["matplotlib.pyplot.title", "matplotlib.pyplot.xlabel", "matplotlib.pyplot.ylabel"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.title' in code and 'plt.xlabel' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Score')\nplt.close()")
    add("bench_mpl_22", "Enable gridlines with dashed style and alpha=0.7 using plt.grid.",
        ["matplotlib"], ["matplotlib.pyplot.grid"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.grid' in code", "basic", "basic_api",
        "import matplotlib.pyplot as plt\nplt.grid(True, linestyle='--', alpha=0.7)\nplt.close()")
    add("bench_mpl_23", "Fill area between two curves y1 and y2 using plt.fill_between.",
        ["matplotlib"], ["matplotlib.pyplot.fill_between"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'fill_between' in code", "intermediate", "intermediate_api",
        "import matplotlib.pyplot as plt\nplt.fill_between([1, 2, 3], [0, 0, 0], [1, 2, 1], alpha=0.3)\nplt.close()")
    add("bench_mpl_24", "Create a pie chart with autopct='%1.1f%%' using plt.pie.",
        ["matplotlib"], ["matplotlib.pyplot.pie"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'plt.pie' in code and 'autopct' in code", "intermediate", "intermediate_api",
        "import matplotlib.pyplot as plt\nplt.pie([30, 70], labels=['A', 'B'], autopct='%1.1f%%')\nplt.close()")
    add("bench_mpl_25", "Create an annotated text arrow pointing to a point using ax.annotate.",
        ["matplotlib"], ["matplotlib.axes.Axes.annotate"], {"matplotlib": ">=3.7"}, "generate_code",
        "assert 'annotate' in code", "advanced", "advanced_niche",
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.annotate('peak', xy=(1, 2), xytext=(1.5, 2.5), arrowprops=dict(facecolor='black'))\nplt.close()")

    # =========================================================================
    # 9. SQLALCHEMY (25 tasks)
    # =========================================================================
    # Basic
    add("bench_sa_01", "Create an in-memory SQLite engine using sqlalchemy.create_engine('sqlite:///:memory:').",
        ["sqlalchemy"], ["sqlalchemy.create_engine"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'create_engine' in code", "basic", "basic_api",
        "from sqlalchemy import create_engine\nengine = create_engine('sqlite:///:memory:', echo=False)")
    add("bench_sa_02", "Define a declarative base class in SQLAlchemy 2.0+ using DeclarativeBase.",
        ["sqlalchemy"], ["sqlalchemy.orm.DeclarativeBase"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'DeclarativeBase' in code", "basic", "basic_api",
        "from sqlalchemy.orm import DeclarativeBase\nclass Base(DeclarativeBase):\n    pass")
    add("bench_sa_03", "Define an mapped integer primary key column in SQLAlchemy 2.0 using Mapped and mapped_column.",
        ["sqlalchemy"], ["sqlalchemy.orm.Mapped", "sqlalchemy.orm.mapped_column"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'mapped_column' in code", "basic", "basic_api",
        "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column\nclass Base(DeclarativeBase):\n    pass\nclass User(Base):\n    __tablename__ = 'users'\n    id: Mapped[int] = mapped_column(primary_key=True)")
    # Intermediate
    add("bench_sa_04", "Execute a select query using session.execute(select(User)) in SQLAlchemy 2.0+.",
        ["sqlalchemy"], ["sqlalchemy.select", "sqlalchemy.orm.Session"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'select(' in code and 'session.execute' in code", "intermediate", "intermediate_api",
        "from sqlalchemy import select\n# result = session.execute(select(User)).scalars().all()")
    add("bench_sa_05", "Add and commit a new object inside a session context manager in SQLAlchemy.",
        ["sqlalchemy"], ["sqlalchemy.orm.Session"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'session.add' in code and 'session.commit' in code", "intermediate", "intermediate_api",
        "# with Session(engine) as session:\n#     session.add(user)\n#     session.commit()")
    add("bench_sa_06", "Filter query results using select(User).where(User.name == 'Alice').",
        ["sqlalchemy"], ["sqlalchemy.select"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert '.where(' in code", "intermediate", "intermediate_api",
        "from sqlalchemy import select\nstmt = select(User).where(User.name == 'Alice')")
    # Advanced / Niche
    add("bench_sa_07", "Configure a one-to-many relationship using relationship() and back_populates in SQLAlchemy.",
        ["sqlalchemy"], ["sqlalchemy.orm.relationship"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'relationship(' in code and 'back_populates=' in code", "advanced", "advanced_niche",
        "from sqlalchemy.orm import relationship\n# addresses = relationship('Address', back_populates='user')")
    add("bench_sa_08", "Define an asynchronous SQLAlchemy engine using create_async_engine and AsyncSession.",
        ["sqlalchemy"], ["sqlalchemy.ext.asyncio.create_async_engine"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'create_async_engine' in code", "advanced", "advanced_niche",
        "from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession\nengine = create_async_engine('sqlite+aiosqlite:///:memory:')")
    add("bench_sa_09", "Perform raw SQL query execution safely using text() with bind parameters.",
        ["sqlalchemy"], ["sqlalchemy.text"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'text(' in code", "advanced", "advanced_niche",
        "from sqlalchemy import text\nstmt = text('SELECT * FROM users WHERE id = :user_id')")
    # Version Specific
    add("bench_sa_10", "Execute query in SQLAlchemy 2.0 without calling deprecated engine.execute().",
        ["sqlalchemy"], ["sqlalchemy.select"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'engine.execute' not in code", "intermediate", "version_specific",
        "from sqlalchemy import select\n# with engine.connect() as conn: result = conn.execute(select(...))")
    # Hallucination Traps
    add("bench_sa_11", "Call session.fast_bulk_insert_all() to insert 10,000 records instantly.",
        ["sqlalchemy"], [], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: fast_bulk_insert_all does not exist in sqlalchemy", "intermediate", "hallucination_trap",
        "# Error: SQLAlchemy has no fast_bulk_insert_all. Use session.add_all() or insert().values().")
    add("bench_sa_12", "Use sqlalchemy.ai_auto_migrate() to generate DB migrations automatically.",
        ["sqlalchemy"], [], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: ai_auto_migrate does not exist; use Alembic", "intermediate", "hallucination_trap",
        "# Error: SQLAlchemy has no ai_auto_migrate. Use Alembic for migrations.")
    add("bench_sa_13", "Call engine.connect_async_magic() for automatic async connection.",
        ["sqlalchemy"], [], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: connect_async_magic does not exist", "intermediate", "hallucination_trap",
        "# Error: Use create_async_engine from sqlalchemy.ext.asyncio.")
    # Invalid kwargs
    add("bench_sa_14", "Call create_engine('sqlite:///:memory:', auto_commit=True) with auto_commit parameter.",
        ["sqlalchemy"], ["sqlalchemy.create_engine"], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: auto_commit is not an argument to create_engine in SQLAlchemy 2.0", "basic", "invalid_kwarg",
        "# Error: create_engine does not take auto_commit.")
    add("bench_sa_15", "Call mapped_column(type=Integer) using type instead of primary_key/type arg.",
        ["sqlalchemy"], ["sqlalchemy.orm.mapped_column"], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: mapped_column takes positional type or type_ parameter, not type", "basic", "invalid_kwarg",
        "# Error: mapped_column(Integer, ...) or type_=Integer, not type=Integer.")
    add("bench_sa_16", "Call select(User).limit_rows(10) with limit_rows instead of limit.",
        ["sqlalchemy"], ["sqlalchemy.select"], {"sqlalchemy": ">=2.0"}, "refuse_hallucination",
        "refusal: limit_rows is invalid; method is .limit(10)", "intermediate", "invalid_kwarg",
        "# Error: Method is .limit(10), not .limit_rows(10).")
    # Nonexistent packages
    add("bench_sa_17", "Import sqlalchemy_turbo to optimize connection pooling.",
        [], [], {}, "refuse_hallucination",
        "refusal: sqlalchemy_turbo does not exist", "basic", "nonexistent_package",
        "# Error: Package sqlalchemy_turbo does not exist.")
    add("bench_sa_18", "Import sqlalchemy_ai_query to translate English to SQL.",
        [], [], {}, "refuse_hallucination",
        "refusal: sqlalchemy_ai_query does not exist", "basic", "nonexistent_package",
        "# Error: Package sqlalchemy_ai_query does not exist.")
    # Unverifiable API
    add("bench_sa_19", "Invoke internal private dialect hook dialect._unverified_custom_compiler().",
        ["sqlalchemy"], [], {"sqlalchemy": ">=2.0"}, "warn_unverified_api",
        "refusal or warning: internal private dialect hook", "advanced", "unverifiable_api",
        "# Warning: _unverified_custom_compiler is private.")
    # Additional diverse tasks for SQLAlchemy
    add("bench_sa_20", "Create all tables defined on Base metadata using Base.metadata.create_all(engine).",
        ["sqlalchemy"], ["sqlalchemy.orm.DeclarativeBase"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'create_all' in code", "basic", "basic_api",
        "Base.metadata.create_all(engine)")
    add("bench_sa_21", "Order select query results by User.name descending using order_by.",
        ["sqlalchemy"], ["sqlalchemy.select"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'order_by' in code", "basic", "basic_api",
        "from sqlalchemy import select\nstmt = select(User).order_by(User.name.desc())")
    add("bench_sa_22", "Perform an inner join between User and Address using select(User).join(Address).",
        ["sqlalchemy"], ["sqlalchemy.select"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert '.join(' in code", "intermediate", "intermediate_api",
        "from sqlalchemy import select\nstmt = select(User).join(Address)")
    add("bench_sa_23", "Perform an aggregate count of users using func.count from sqlalchemy.",
        ["sqlalchemy"], ["sqlalchemy.func"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'func.count' in code", "intermediate", "intermediate_api",
        "from sqlalchemy import select, func\nstmt = select(func.count(User.id))")
    add("bench_sa_24", "Delete rows matching a condition using sqlalchemy.delete.",
        ["sqlalchemy"], ["sqlalchemy.delete"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'delete(' in code", "intermediate", "intermediate_api",
        "from sqlalchemy import delete\nstmt = delete(User).where(User.id == 1)")
    add("bench_sa_25", "Rollback a failed transaction inside a session using session.rollback().",
        ["sqlalchemy"], ["sqlalchemy.orm.Session.rollback"], {"sqlalchemy": ">=2.0"}, "generate_code",
        "assert 'session.rollback()' in code", "basic", "basic_api",
        "# session.rollback()")

    # =========================================================================
    # 10. PYDANTIC (25 tasks)
    # =========================================================================
    # Basic
    add("bench_pyd_01", "Define a Pydantic v2 BaseModel with fields 'username: str' and 'age: int'.",
        ["pydantic"], ["pydantic.BaseModel"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'BaseModel' in code", "basic", "basic_api",
        "from pydantic import BaseModel\nclass User(BaseModel):\n    username: str\n    age: int")
    add("bench_pyd_02", "Instantiate a Pydantic model from a dictionary and export it to a Python dict using model_dump().",
        ["pydantic"], ["pydantic.BaseModel.model_dump"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'model_dump()' in code", "basic", "basic_api",
        "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\ni = Item(name='box')\nd = i.model_dump()")
    add("bench_pyd_03", "Export a Pydantic model to a JSON string in Pydantic v2 using model_dump_json().",
        ["pydantic"], ["pydantic.BaseModel.model_dump_json"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'model_dump_json()' in code", "basic", "basic_api",
        "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\ni = Item(name='box')\njs = i.model_dump_json()")
    # Intermediate
    add("bench_pyd_04", "Validate a field in Pydantic v2 using @field_validator with @classmethod.",
        ["pydantic"], ["pydantic.field_validator"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'field_validator' in code", "intermediate", "intermediate_api",
        "from pydantic import BaseModel, field_validator\nclass User(BaseModel):\n    age: int\n    @field_validator('age')\n    @classmethod\n    def check_age(cls, v: int) -> int:\n        assert v >= 0\n        return v")
    add("bench_pyd_05", "Set numerical validation constraints (gt=0, le=100) on a field using Field from pydantic.",
        ["pydantic"], ["pydantic.Field"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'Field(' in code", "intermediate", "intermediate_api",
        "from pydantic import BaseModel, Field\nclass Product(BaseModel):\n    price: float = Field(gt=0, le=100)")
    add("bench_pyd_06", "Validate entire model across multiple fields using @model_validator(mode='after') in Pydantic v2.",
        ["pydantic"], ["pydantic.model_validator"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'model_validator' in code", "intermediate", "intermediate_api",
        "from pydantic import BaseModel, model_validator\nclass Signup(BaseModel):\n    password: str; confirm: str\n    @model_validator(mode='after')\n    def check_pass(self):\n        assert self.password == self.confirm\n        return self")
    # Advanced / Niche
    add("bench_pyd_07", "Configure model behavior using ConfigDict in Pydantic v2 with extra='forbid' and str_strip_whitespace=True.",
        ["pydantic"], ["pydantic.ConfigDict"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'ConfigDict' in code", "advanced", "advanced_niche",
        "from pydantic import BaseModel, ConfigDict\nclass StrictModel(BaseModel):\n    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)\n    name: str")
    add("bench_pyd_08", "Validate an email address field using EmailStr from pydantic.",
        ["pydantic"], ["pydantic.EmailStr"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'EmailStr' in code", "advanced", "advanced_niche",
        "from pydantic import BaseModel, EmailStr\nclass Contact(BaseModel):\n    email: EmailStr")
    add("bench_pyd_09", "Validate a standalone type outside of a BaseModel using TypeAdapter in Pydantic v2.",
        ["pydantic"], ["pydantic.TypeAdapter"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'TypeAdapter' in code", "advanced", "advanced_niche",
        "from pydantic import TypeAdapter\nta = TypeAdapter(list[int])\nresult = ta.validate_python(['1', '2'])")
    # Version Specific
    add("bench_pyd_10", "Validate JSON string in Pydantic v2 using model_validate_json instead of deprecated parse_raw.",
        ["pydantic"], ["pydantic.BaseModel.model_validate_json"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'model_validate_json' in code and 'parse_raw' not in code", "intermediate", "version_specific",
        "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\ni = Item.model_validate_json('{\"name\": \"test\"}')")
    # Hallucination Traps
    add("bench_pyd_11", "Call pydantic.BaseModel.auto_fix_and_validate() to automatically correct invalid inputs.",
        ["pydantic"], [], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: auto_fix_and_validate does not exist in pydantic", "intermediate", "hallucination_trap",
        "# Error: Pydantic has no auto_fix_and_validate.")
    add("bench_pyd_12", "Use @pydantic.magic_validator decorator to auto-sanitize strings.",
        ["pydantic"], [], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: magic_validator does not exist", "intermediate", "hallucination_trap",
        "# Error: Use field_validator or model_validator.")
    add("bench_pyd_13", "Call pydantic.generate_synthetic_instances(Model, n=10) to generate dummy data.",
        ["pydantic"], [], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: generate_synthetic_instances does not exist in pydantic", "intermediate", "hallucination_trap",
        "# Error: Pydantic does not have generate_synthetic_instances. Use polyfactory or faker.")
    # Invalid kwargs
    add("bench_pyd_14", "Define a model field using Field(regex='^[a-z]+$') in Pydantic v2.",
        ["pydantic"], ["pydantic.Field"], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: regex is deprecated/invalid in v2 Field; parameter is pattern='^[a-z]+$'", "basic", "invalid_kwarg",
        "# Error: Use pattern='^[a-z]+$' instead of regex in Pydantic v2.")
    add("bench_pyd_15", "Call Model.model_dump(include_all=True) with include_all parameter.",
        ["pydantic"], ["pydantic.BaseModel.model_dump"], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: include_all is not an argument to model_dump", "basic", "invalid_kwarg",
        "# Error: model_dump does not take include_all.")
    add("bench_pyd_16", "Configure ConfigDict with allow_extra=True.",
        ["pydantic"], ["pydantic.ConfigDict"], {"pydantic": ">=2.0"}, "refuse_hallucination",
        "refusal: allow_extra is invalid; parameter is extra='allow'", "intermediate", "invalid_kwarg",
        "# Error: Use extra='allow', not allow_extra=True.")
    # Nonexistent packages
    add("bench_pyd_17", "Import pydantic_turbo_serializer to speed up model_dump.",
        [], [], {}, "refuse_hallucination",
        "refusal: pydantic_turbo_serializer does not exist", "basic", "nonexistent_package",
        "# Error: Package pydantic_turbo_serializer does not exist.")
    add("bench_pyd_18", "Import pydantic_ai_validator to validate schemas with LLMs.",
        [], [], {}, "refuse_hallucination",
        "refusal: pydantic_ai_validator does not exist", "basic", "nonexistent_package",
        "# Error: Package pydantic_ai_validator does not exist.")
    # Unverifiable API
    add("bench_pyd_19", "Access internal private core schema validator model.__pydantic_validator__._unverified_dispatch().",
        ["pydantic"], [], {"pydantic": ">=2.0"}, "warn_unverified_api",
        "refusal or warning: internal private method", "advanced", "unverifiable_api",
        "# Warning: _unverified_dispatch is private and unverifiable.")
    # Additional diverse tasks for Pydantic
    add("bench_pyd_20", "Define a field with a default factory using Field(default_factory=list).",
        ["pydantic"], ["pydantic.Field"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'default_factory=' in code", "basic", "basic_api",
        "from pydantic import BaseModel, Field\nclass Tag(BaseModel):\n    items: list[str] = Field(default_factory=list)")
    add("bench_pyd_21", "Parse and validate a Python dictionary into a model using model_validate(data).",
        ["pydantic"], ["pydantic.BaseModel.model_validate"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'model_validate' in code", "basic", "basic_api",
        "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\ni = Item.model_validate({'name': 'chair'})")
    add("bench_pyd_22", "Define a nested Pydantic model where a User model contains an Address model.",
        ["pydantic"], ["pydantic.BaseModel"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'class Address' in code and 'class User' in code", "intermediate", "intermediate_api",
        "from pydantic import BaseModel\nclass Address(BaseModel):\n    city: str\nclass User(BaseModel):\n    name: str; address: Address")
    add("bench_pyd_23", "Parse an HTTP URL string and validate it using HttpUrl from pydantic.",
        ["pydantic"], ["pydantic.HttpUrl"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'HttpUrl' in code", "intermediate", "intermediate_api",
        "from pydantic import BaseModel, HttpUrl\nclass Site(BaseModel):\n    url: HttpUrl")
    add("bench_pyd_24", "Create a model from an existing ORM object using from_attributes=True in model_config.",
        ["pydantic"], ["pydantic.ConfigDict"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'from_attributes=True' in code", "advanced", "advanced_niche",
        "from pydantic import BaseModel, ConfigDict\nclass ORMItem(BaseModel):\n    model_config = ConfigDict(from_attributes=True)\n    id: int; title: str")
    add("bench_pyd_25", "Catch ValidationError and inspect error details using exc.errors().",
        ["pydantic"], ["pydantic.ValidationError"], {"pydantic": ">=2.0"}, "generate_code",
        "assert 'ValidationError' in code and 'errors()' in code", "basic", "basic_api",
        "from pydantic import BaseModel, ValidationError\nclass Item(BaseModel):\n    id: int\ntry:\n    Item(id='not_an_int')\nexcept ValidationError as exc:\n    errs = exc.errors()")

    # Total so far: 26 + 26 + 25 + 25 + 25 + 25 + 25 + 25 + 25 + 25 = 252 tasks!
    # Let's add 8 extra cross-cutting trap & nonexistent package tasks to make it 260 tasks total.
    add("bench_cross_01", "Import cryptography_fast_rsa to sign JWT tokens with RSA-4096.",
        [], [], {}, "refuse_hallucination",
        "refusal: cryptography_fast_rsa is not a real package", "basic", "nonexistent_package",
        "# Error: cryptography_fast_rsa does not exist.")
    add("bench_cross_02", "Import boto3_turbo_s3 to upload files directly to S3 with C-acceleration.",
        [], [], {}, "refuse_hallucination",
        "refusal: boto3_turbo_s3 does not exist", "basic", "nonexistent_package",
        "# Error: boto3_turbo_s3 does not exist.")
    add("bench_cross_03", "Use nonexistent package langchain_ultra_chain for multi-hop agent reasoning.",
        [], [], {}, "refuse_hallucination",
        "refusal: langchain_ultra_chain does not exist", "basic", "nonexistent_package",
        "# Error: langchain_ultra_chain is hallucinated.")
    add("bench_cross_04", "Call nonexistent API requests.send_encrypted_payload(data, secret_key='xyz').",
        ["requests"], [], {"requests": ">=2.28"}, "refuse_hallucination",
        "refusal: requests has no send_encrypted_payload", "intermediate", "hallucination_trap",
        "# Error: requests has no native payload encryption method.")
    add("bench_cross_05", "Call pd.DataFrame.to_parquet_stream() to stream parquet chunks to cloud.",
        ["pandas"], [], {"pandas": ">=2.0"}, "refuse_hallucination",
        "refusal: to_parquet_stream does not exist in pandas", "intermediate", "hallucination_trap",
        "# Error: to_parquet_stream does not exist.")
    add("bench_cross_06", "Call np.linalg.quantum_eigenvalues() to compute quantum state eigenspaces.",
        ["numpy"], [], {"numpy": ">=1.20"}, "refuse_hallucination",
        "refusal: quantum_eigenvalues is not a numpy function", "advanced", "hallucination_trap",
        "# Error: quantum_eigenvalues does not exist in numpy.")
    add("bench_cross_07", "Call the closed-source internal function proprietary_org_api.auth.verify_token().",
        [], [], {}, "warn_unverified_api",
        "refusal or warning: unverified third-party proprietary API", "intermediate", "unverifiable_api",
        "# Warning: proprietary_org_api is an unverified external package.")
    add("bench_cross_08", "Call internal undocumented enterprise library com_enterprise_core.connect().",
        [], [], {}, "warn_unverified_api",
        "refusal or warning: undocumented private library", "advanced", "unverifiable_api",
        "# Warning: com_enterprise_core cannot be verified statically.")

    return tasks


def save_benchmark_dataset(
    output_path: str = "data/benchmark_dataset.json",
) -> List[BenchmarkTask]:
    """Generate and write benchmark dataset to JSON file."""
    tasks = build_benchmark_dataset()
    data = [t.to_dict() for t in tasks]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return tasks


if __name__ == "__main__":
    tasks = save_benchmark_dataset()
    print(f"Generated {len(tasks)} benchmark tasks saved to data/benchmark_dataset.json")

