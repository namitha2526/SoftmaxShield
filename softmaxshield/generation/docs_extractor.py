"""Documentation and signature extractor for the 10 popular target libraries with metadata filtering and version update mechanisms."""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DocChunk:
    """A granular documentation chunk representing a verified function or class signature with rich metadata."""

    library: str
    version: str
    api_name: str  # Fully qualified symbol name, e.g. 'numpy.array'
    symbol_type: str  # 'function', 'class', 'method', 'decorator'
    signature: str
    docstring: str
    source_url: str = ""
    doc_timestamp: str = "2024-04-01T00:00:00Z"
    parameters: List[str] = field(default_factory=list)
    return_type: str = "Any"
    examples: str = ""
    deprecated_in: Optional[str] = None
    added_in: Optional[str] = None

    @property
    def symbol(self) -> str:
        """Alias for api_name to support legacy symbol lookups."""
        return self.api_name

    def to_embedding_text(self) -> str:
        """Construct dense textual representation optimized for semantic vector retrieval."""
        lines = [
            f"Library: {self.library} (version {self.version})",
            f"API: {self.api_name} ({self.symbol_type})",
            f"Signature: {self.signature}",
            f"Summary: {self.docstring}",
            f"Parameters: {', '.join(self.parameters) if self.parameters else 'None'}",
            f"Returns: {self.return_type}",
            f"Source URL: {self.source_url}",
        ]
        if self.examples:
            lines.append(f"Usage Example:\n{self.examples}")
        if self.deprecated_in:
            lines.append(f"DEPRECATION WARNING: Deprecated in version {self.deprecated_in}")
        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """Compact representation injected into LLM context."""
        dep_str = f" [DEPRECATED in {self.deprecated_in}]" if self.deprecated_in else ""
        return (
            f"[{self.library}=={self.version}{dep_str}] {self.signature}\n"
            f"  Doc: {self.docstring}\n"
            f"  Parameters: {', '.join(self.parameters)}\n"
            f"  Returns: {self.return_type}\n"
            f"  Source: {self.source_url}"
        )


# Comprehensive curated signature dataset for the 10 target libraries with official documentation URLs and timestamps
TEN_LIBRARIES_SIGNATURES: List[Dict[str, Any]] = [
    # 1. NumPy
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.array",
        "symbol_type": "function",
        "signature": "numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)",
        "docstring": "Create an array from an array-like object or sequence.",
        "source_url": "https://numpy.org/doc/1.26/reference/generated/numpy.array.html",
        "doc_timestamp": "2024-02-06T00:00:00Z",
        "parameters": ["object", "dtype", "copy", "order", "subok", "ndmin", "like"],
        "return_type": "numpy.ndarray",
        "examples": "import numpy as np\narr = np.array([1, 2, 3], dtype=np.float32)",
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.linspace",
        "symbol_type": "function",
        "signature": "numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)",
        "docstring": "Return evenly spaced numbers over a specified closed or half-open interval.",
        "source_url": "https://numpy.org/doc/1.26/reference/generated/numpy.linspace.html",
        "doc_timestamp": "2024-02-06T00:00:00Z",
        "parameters": ["start", "stop", "num", "endpoint", "retstep", "dtype", "axis"],
        "return_type": "numpy.ndarray",
        "examples": "x = np.linspace(0.0, 1.0, num=100)",
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.mean",
        "symbol_type": "function",
        "signature": "numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True)",
        "docstring": "Compute the arithmetic mean along the specified axis.",
        "source_url": "https://numpy.org/doc/1.26/reference/generated/numpy.mean.html",
        "doc_timestamp": "2024-02-06T00:00:00Z",
        "parameters": ["a", "axis", "dtype", "out", "keepdims", "where"],
        "return_type": "scalar or numpy.ndarray",
        "examples": "avg = np.mean(arr, axis=0, keepdims=True)",
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.matmul",
        "symbol_type": "function",
        "signature": "numpy.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True)",
        "docstring": "Matrix product of two arrays. Positional-only inputs x1 and x2.",
        "source_url": "https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html",
        "doc_timestamp": "2024-02-06T00:00:00Z",
        "parameters": ["x1", "x2", "out", "casting", "order", "dtype", "subok"],
        "return_type": "numpy.ndarray",
        "examples": "result = np.matmul(A, B)",
    },

    # 2. Pandas
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.read_csv",
        "symbol_type": "function",
        "signature": "pandas.read_csv(filepath_or_buffer, *, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None, dtype=None, engine=None, encoding=None)",
        "docstring": "Read a comma-separated values (csv) file into a DataFrame.",
        "source_url": "https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.read_csv.html",
        "doc_timestamp": "2024-04-10T00:00:00Z",
        "parameters": ["filepath_or_buffer", "sep", "delimiter", "header", "names", "index_col", "usecols", "dtype", "engine", "encoding"],
        "return_type": "pandas.DataFrame",
        "examples": "df = pd.read_csv('data.csv', usecols=['id', 'val'], dtype={'id': int})",
    },
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.DataFrame.merge",
        "symbol_type": "method",
        "signature": "pandas.DataFrame.merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None)",
        "docstring": "Merge DataFrame or named Series objects with a database-style join.",
        "source_url": "https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.merge.html",
        "doc_timestamp": "2024-04-10T00:00:00Z",
        "parameters": ["right", "how", "on", "left_on", "right_on", "left_index", "right_index", "sort", "suffixes", "copy", "indicator", "validate"],
        "return_type": "pandas.DataFrame",
        "examples": "merged = df1.merge(df2, on='user_id', how='left')",
    },
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.DataFrame.groupby",
        "symbol_type": "method",
        "signature": "pandas.DataFrame.groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True)",
        "docstring": "Group DataFrame using a mapper or by a Series of columns.",
        "source_url": "https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.groupby.html",
        "doc_timestamp": "2024-04-10T00:00:00Z",
        "parameters": ["by", "axis", "level", "as_index", "sort", "group_keys", "observed", "dropna"],
        "return_type": "pandas.core.groupby.DataFrameGroupBy",
        "examples": "grouped = df.groupby('category')['sales'].sum()",
    },
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.concat",
        "symbol_type": "function",
        "signature": "pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None)",
        "docstring": "Concatenate pandas objects along a particular axis. Replaces deprecated DataFrame.append().",
        "source_url": "https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.concat.html",
        "doc_timestamp": "2024-04-10T00:00:00Z",
        "parameters": ["objs", "axis", "join", "ignore_index", "keys", "levels", "names", "verify_integrity", "sort", "copy"],
        "return_type": "pandas.DataFrame or pandas.Series",
        "examples": "df = pd.concat([df1, df2], ignore_index=True)",
    },

    # 3. Requests
    {
        "library": "requests",
        "version": "2.31.0",
        "api_name": "requests.get",
        "symbol_type": "function",
        "signature": "requests.get(url, params=None, *, headers=None, cookies=None, auth=None, timeout=None, allow_redirects=True, proxies=None, verify=True, stream=False, cert=None)",
        "docstring": "Sends a synchronous HTTP GET request to the specified URL.",
        "source_url": "https://requests.readthedocs.io/en/latest/api/#requests.get",
        "doc_timestamp": "2023-05-22T00:00:00Z",
        "parameters": ["url", "params", "headers", "cookies", "auth", "timeout", "allow_redirects", "proxies", "verify", "stream", "cert"],
        "return_type": "requests.Response",
        "examples": "resp = requests.get('https://api.github.com', timeout=10.0, headers={'Accept': 'application/json'})",
    },
    {
        "library": "requests",
        "version": "2.31.0",
        "api_name": "requests.post",
        "symbol_type": "function",
        "signature": "requests.post(url, data=None, json=None, *, headers=None, cookies=None, auth=None, timeout=None, allow_redirects=True, proxies=None, verify=True, stream=False, cert=None)",
        "docstring": "Sends a synchronous HTTP POST request to the specified URL with data or JSON.",
        "source_url": "https://requests.readthedocs.io/en/latest/api/#requests.post",
        "doc_timestamp": "2023-05-22T00:00:00Z",
        "parameters": ["url", "data", "json", "headers", "cookies", "auth", "timeout", "allow_redirects", "proxies", "verify", "stream", "cert"],
        "return_type": "requests.Response",
        "examples": "resp = requests.post('https://httpbin.org/post', json={'name': 'alice'}, timeout=5.0)",
    },
    {
        "library": "requests",
        "version": "2.31.0",
        "api_name": "requests.Session",
        "symbol_type": "class",
        "signature": "requests.Session()",
        "docstring": "Provides persistent cookie-storage, connection pooling, and configuration across requests.",
        "source_url": "https://requests.readthedocs.io/en/latest/api/#requests.Session",
        "doc_timestamp": "2023-05-22T00:00:00Z",
        "parameters": [],
        "return_type": "requests.Session",
        "examples": "with requests.Session() as s:\n    s.headers.update({'Authorization': 'Bearer token'})\n    r = s.get('https://api.example.com/me')",
    },

    # 4. Flask
    {
        "library": "flask",
        "version": "3.0.2",
        "api_name": "flask.Flask",
        "symbol_type": "class",
        "signature": "flask.Flask(import_name, static_url_path=None, static_folder='static', template_folder='templates', instance_path=None, instance_relative_config=False, root_path=None)",
        "docstring": "The flask object implements a WSGI application and acts as the central object.",
        "source_url": "https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask",
        "doc_timestamp": "2024-02-04T00:00:00Z",
        "parameters": ["import_name", "static_url_path", "static_folder", "template_folder", "instance_path", "instance_relative_config", "root_path"],
        "return_type": "flask.Flask",
        "examples": "app = Flask(__name__)\n@app.route('/health')\ndef health():\n    return {'status': 'healthy'}",
    },
    {
        "library": "flask",
        "version": "3.0.2",
        "api_name": "flask.jsonify",
        "symbol_type": "function",
        "signature": "flask.jsonify(*args, **kwargs)",
        "docstring": "Serialize data to JSON and wrap it in a Response object with application/json mimetype.",
        "source_url": "https://flask.palletsprojects.com/en/3.0.x/api/#flask.json.jsonify",
        "doc_timestamp": "2024-02-04T00:00:00Z",
        "parameters": ["*args", "**kwargs"],
        "return_type": "flask.Response",
        "examples": "return jsonify(user_id=123, role='admin')",
    },

    # 5. FastAPI
    {
        "library": "fastapi",
        "version": "0.115.0",
        "api_name": "fastapi.FastAPI",
        "symbol_type": "class",
        "signature": "fastapi.FastAPI(*, title='FastAPI', description='', version='0.1.0', openapi_url='/openapi.json', docs_url='/docs', redoc_url='/redoc', lifespan=None)",
        "docstring": "FastAPI modern, high-performance web framework for building APIs with Python 3.8+.",
        "source_url": "https://fastapi.tiangolo.com/reference/fastapi/",
        "doc_timestamp": "2024-09-12T00:00:00Z",
        "parameters": ["title", "description", "version", "openapi_url", "docs_url", "redoc_url", "lifespan"],
        "return_type": "fastapi.FastAPI",
        "examples": "app = FastAPI(title='SoftmaxShield API', version='1.0.0')",
    },
    {
        "library": "fastapi",
        "version": "0.115.0",
        "api_name": "fastapi.Depends",
        "symbol_type": "function",
        "signature": "fastapi.Depends(dependency=None, *, use_cache=True)",
        "docstring": "Declare a sub-dependency parameter in a path operation function.",
        "source_url": "https://fastapi.tiangolo.com/reference/dependencies/",
        "doc_timestamp": "2024-09-12T00:00:00Z",
        "parameters": ["dependency", "use_cache"],
        "return_type": "fastapi.params.Depends",
        "examples": "@app.get('/items/')\ndef get_items(db: Session = Depends(get_db)):\n    ...",
    },
    {
        "library": "fastapi",
        "version": "0.115.0",
        "api_name": "fastapi.HTTPException",
        "symbol_type": "class",
        "signature": "fastapi.HTTPException(status_code, detail=None, headers=None)",
        "docstring": "HTTP exception to return HTTP error responses with status codes and details.",
        "source_url": "https://fastapi.tiangolo.com/reference/exceptions/",
        "doc_timestamp": "2024-09-12T00:00:00Z",
        "parameters": ["status_code", "detail", "headers"],
        "return_type": "fastapi.exceptions.HTTPException",
        "examples": "raise HTTPException(status_code=404, detail='Resource not found')",
    },

    # 6. Scikit-Learn
    {
        "library": "scikit-learn",
        "version": "1.4.0",
        "api_name": "sklearn.linear_model.LogisticRegression",
        "symbol_type": "class",
        "signature": "sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)",
        "docstring": "Logistic Regression (aka logit, MaxEnt) classifier.",
        "source_url": "https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "doc_timestamp": "2024-01-18T00:00:00Z",
        "parameters": ["penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling", "class_weight", "random_state", "solver", "max_iter", "multi_class", "n_jobs", "l1_ratio"],
        "return_type": "sklearn.linear_model.LogisticRegression",
        "examples": "clf = LogisticRegression(C=0.5, max_iter=200, random_state=42)\nclf.fit(X_train, y_train)",
    },
    {
        "library": "scikit-learn",
        "version": "1.4.0",
        "api_name": "sklearn.model_selection.train_test_split",
        "symbol_type": "function",
        "signature": "sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)",
        "docstring": "Split arrays or matrices into random train and test subsets.",
        "source_url": "https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.train_test_split.html",
        "doc_timestamp": "2024-01-18T00:00:00Z",
        "parameters": ["*arrays", "test_size", "train_size", "random_state", "shuffle", "stratify"],
        "return_type": "list of splits",
        "examples": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
    },

    # 7. PyTorch
    {
        "library": "torch",
        "version": "2.2.0",
        "api_name": "torch.nn.Linear",
        "symbol_type": "class",
        "signature": "torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)",
        "docstring": "Applies an affine linear transformation to incoming data: y = xA^T + b.",
        "source_url": "https://pytorch.org/docs/2.2/generated/torch.nn.Linear.html",
        "doc_timestamp": "2024-01-30T00:00:00Z",
        "parameters": ["in_features", "out_features", "bias", "device", "dtype"],
        "return_type": "torch.nn.Module",
        "examples": "layer = torch.nn.Linear(768, 128, bias=True)",
    },
    {
        "library": "torch",
        "version": "2.2.0",
        "api_name": "torch.optim.AdamW",
        "symbol_type": "class",
        "signature": "torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)",
        "docstring": "Implements AdamW optimizer with decoupled weight decay as described in Loshchilov & Hutter.",
        "source_url": "https://pytorch.org/docs/2.2/generated/torch.optim.AdamW.html",
        "doc_timestamp": "2024-01-30T00:00:00Z",
        "parameters": ["params", "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "fused"],
        "return_type": "torch.optim.Optimizer",
        "examples": "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)",
    },
    {
        "library": "torch",
        "version": "2.2.0",
        "api_name": "torch.cuda.empty_cache",
        "symbol_type": "function",
        "signature": "torch.cuda.empty_cache()",
        "docstring": "Releases all unoccupied cached memory currently held by the caching allocator.",
        "source_url": "https://pytorch.org/docs/2.2/generated/torch.cuda.empty_cache.html",
        "doc_timestamp": "2024-01-30T00:00:00Z",
        "parameters": [],
        "return_type": "None",
        "examples": "torch.cuda.empty_cache()",
    },

    # 8. Matplotlib
    {
        "library": "matplotlib",
        "version": "3.8.0",
        "api_name": "matplotlib.pyplot.subplots",
        "symbol_type": "function",
        "signature": "matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw)",
        "docstring": "Create a figure and a set of subplots.",
        "source_url": "https://matplotlib.org/3.8.0/api/_as_gen/matplotlib.pyplot.subplots.html",
        "doc_timestamp": "2023-09-14T00:00:00Z",
        "parameters": ["nrows", "ncols", "sharex", "sharey", "squeeze", "width_ratios", "height_ratios", "subplot_kw", "gridspec_kw", "**fig_kw"],
        "return_type": "tuple of (Figure, Axes)",
        "examples": "fig, ax = plt.subplots(figsize=(8, 6), dpi=100)\nax.plot(x, y, label='Accuracy')",
    },

    # 9. SQLAlchemy
    {
        "library": "sqlalchemy",
        "version": "2.0.28",
        "api_name": "sqlalchemy.orm.Session",
        "symbol_type": "class",
        "signature": "sqlalchemy.orm.Session(bind=None, autoflush=True, expire_on_commit=True, info=None)",
        "docstring": "Manages persistence operations for ORM-mapped objects in SQLAlchemy 2.0+.",
        "source_url": "https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.Session",
        "doc_timestamp": "2024-03-05T00:00:00Z",
        "parameters": ["bind", "autoflush", "expire_on_commit", "info"],
        "return_type": "sqlalchemy.orm.Session",
        "examples": "with Session(engine) as session:\n    result = session.scalars(select(User).where(User.active == True)).all()",
    },
    {
        "library": "sqlalchemy",
        "version": "2.0.28",
        "api_name": "sqlalchemy.select",
        "symbol_type": "function",
        "signature": "sqlalchemy.select(*entities)",
        "docstring": "Construct a new Select statement in SQLAlchemy 2.0 style.",
        "source_url": "https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.select",
        "doc_timestamp": "2024-03-05T00:00:00Z",
        "parameters": ["*entities"],
        "return_type": "sqlalchemy.sql.Select",
        "examples": "stmt = select(User).where(User.email == 'test@example.com')",
    },

    # 10. Pydantic
    {
        "library": "pydantic",
        "version": "2.7.0",
        "api_name": "pydantic.BaseModel",
        "symbol_type": "class",
        "signature": "pydantic.BaseModel(**data)",
        "docstring": "Primary base class for data validation and settings management in Pydantic v2.",
        "source_url": "https://docs.pydantic.dev/2.7/api/base_model/",
        "doc_timestamp": "2024-04-11T00:00:00Z",
        "parameters": ["**data"],
        "return_type": "pydantic.BaseModel",
        "examples": "class UserSchema(BaseModel):\n    id: int\n    username: str\n    email: str",
    },
    {
        "library": "pydantic",
        "version": "2.7.0",
        "api_name": "pydantic.BaseModel.model_dump",
        "symbol_type": "method",
        "signature": "pydantic.BaseModel.model_dump(self, *, mode='python', include=None, exclude=None, context=None, by_alias=False, exclude_unset=False, exclude_defaults=False, exclude_none=False, round_trip=False, warnings=True, serialize_as_any=False)",
        "docstring": "Generate a dictionary representation of the model (Pydantic v2 replacement for .dict()).",
        "source_url": "https://docs.pydantic.dev/2.7/api/base_model/#pydantic.BaseModel.model_dump",
        "doc_timestamp": "2024-04-11T00:00:00Z",
        "parameters": ["mode", "include", "exclude", "context", "by_alias", "exclude_unset", "exclude_defaults", "exclude_none", "round_trip", "warnings", "serialize_as_any"],
        "return_type": "dict",
        "examples": "user_dict = user.model_dump(exclude_none=True)",
    },
    {
        "library": "pydantic",
        "version": "2.7.0",
        "api_name": "pydantic.field_validator",
        "symbol_type": "decorator",
        "signature": "pydantic.field_validator(*fields, mode='after', check_fields=None)",
        "docstring": "Decorate methods on BaseModel to validate fields in Pydantic v2 (replaces v1 @validator).",
        "source_url": "https://docs.pydantic.dev/2.7/api/functional_validators/#pydantic.functional_validators.field_validator",
        "doc_timestamp": "2024-04-11T00:00:00Z",
        "parameters": ["*fields", "mode", "check_fields"],
        "return_type": "classmethod validator decorator",
        "examples": "@field_validator('email')\n@classmethod\ndef validate_email(cls, v: str) -> str:\n    if '@' not in v:\n        raise ValueError('Invalid email')\n    return v.lower()",
    },
]


class LibraryDocsCatalog:
    """Manages version-tagged signature chunks for the 10 target libraries with update mechanisms."""

    def __init__(self, chunks: Optional[List[DocChunk]] = None) -> None:
        self.chunks: List[DocChunk] = chunks or [DocChunk(**item) for item in TEN_LIBRARIES_SIGNATURES]

    def add_chunk(self, chunk: DocChunk) -> None:
        """Register a new documentation signature chunk."""
        self.chunks.append(chunk)

    def update_library_docs(self, library: str, version: str, new_chunks: List[DocChunk]) -> int:
        """Update or register documentation when a library version changes.

        Removes outdated chunks for the given (library, version) pair and registers new ones.
        Returns the updated count of chunks.
        """
        # Filter out existing chunks for this specific version
        self.chunks = [
            c for c in self.chunks
            if not (c.library.lower() == library.lower() and c.version == version)
        ]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for chunk in new_chunks:
            chunk.doc_timestamp = timestamp
            self.chunks.append(chunk)
        return len(self.chunks)

    def get_supported_versions(self, library: str) -> List[str]:
        """Return list of distinct registered versions for a given library."""
        versions: Set[str] = {c.version for c in self.chunks if c.library.lower() == library.lower()}
        return sorted(list(versions))

    def find_api(
        self,
        library: str,
        api_name: str,
        version: Optional[str] = None,
    ) -> Optional[DocChunk]:
        """Locate exact API documentation chunk or return None without inventing references."""
        for c in self.chunks:
            if c.library.lower() == library.lower():
                if version and not c.version.startswith(version):
                    continue
                if c.api_name.lower() == api_name.lower():
                    return c
        return None

    def filter_by_metadata(
        self,
        library: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[DocChunk]:
        """Filter chunks based on library name and optional version prefix."""
        results = self.chunks
        if library:
            results = [c for c in results if c.library.lower() == library.lower()]
        if version:
            results = [c for c in results if c.version.startswith(version)]
        return results

    def save_json(self, path: str) -> None:
        """Serialize all chunks to JSON file."""
        data = [asdict(c) for c in self.chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> LibraryDocsCatalog:
        """Load catalog from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [DocChunk(**item) for item in data]
        return cls(chunks=chunks)


def get_ten_libraries_docs() -> LibraryDocsCatalog:
    """Factory creating the default catalog for the 10 target libraries."""
    return LibraryDocsCatalog()
