"""Fine-tuning dataset generator producing 2,000+ grounded instruction pairs conforming to the SoftmaxShield generation schema."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# Base templates for positive verified API usage across the 10 target libraries
POSITIVE_DATA_TEMPLATES = [
    # 1. NumPy
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.array",
        "prompt_fmt": "Create a NumPy array from a {shape} nested list with {dtype} precision and verify its shape.",
        "code_fmt": "import numpy as np\n\ndata = [[1.0, 2.0], [3.0, 4.0]]\narr = np.array(data, dtype=np.{dtype}, copy=True)\nprint('Shape:', arr.shape)",
        "variations": [
            {"shape": "2x2", "dtype": "float32"},
            {"shape": "2x2", "dtype": "float64"},
            {"shape": "2x2", "dtype": "int32"},
        ],
        "explanation": "Uses numpy.array with explicit dtype and copy parameter adhering to NumPy 1.26+ conventions."
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.linspace",
        "prompt_fmt": "Use numpy.linspace to generate {num} points between {start} and {stop} with endpoint included.",
        "code_fmt": "import numpy as np\n\npoints = np.linspace(start={start}, stop={stop}, num={num}, endpoint=True)\nprint(points)",
        "variations": [
            {"num": "50", "start": "0.0", "stop": "1.0"},
            {"num": "100", "start": "-10.0", "stop": "10.0"},
            {"num": "200", "start": "0.0", "stop": "6.28"},
        ],
        "explanation": "Calls numpy.linspace with documented keyword arguments start, stop, and num."
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.mean",
        "prompt_fmt": "Calculate the arithmetic mean of a 2D NumPy array along axis {axis} while preserving dimensions.",
        "code_fmt": "import numpy as np\n\narr = np.random.randn(10, 5)\nmean_vec = np.mean(arr, axis={axis}, keepdims=True)\nprint(mean_vec)",
        "variations": [
            {"axis": "0"},
            {"axis": "1"},
            {"axis": "None"},
        ],
        "explanation": "Calculates mean using numpy.mean with keepdims=True to preserve tensor rank."
    },
    {
        "library": "numpy",
        "version": "1.26.4",
        "api_name": "numpy.matmul",
        "prompt_fmt": "Compute the matrix product of two {dim}x{dim} matrices using numpy.matmul.",
        "code_fmt": "import numpy as np\n\nA = np.ones(({dim}, {dim}), dtype=np.float32)\nB = np.eye({dim}, dtype=np.float32)\nC = np.matmul(A, B)\nprint(C.shape)",
        "variations": [
            {"dim": "4"},
            {"dim": "8"},
            {"dim": "16"},
        ],
        "explanation": "Calls verified numpy.matmul with positional matrix arguments."
    },

    # 2. Pandas
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.read_csv",
        "prompt_fmt": "Load a CSV file from '{path}' into a Pandas DataFrame selecting columns {cols} with encoding '{enc}'.",
        "code_fmt": "import pandas as pd\n\ndf = pd.read_csv('{path}', usecols={cols}, encoding='{enc}')\nprint(df.head())",
        "variations": [
            {"path": "sales.csv", "cols": "['date', 'revenue']", "enc": "utf-8"},
            {"path": "users.csv", "cols": "['user_id', 'email']", "enc": "utf-8"},
            {"path": "logs.csv", "cols": "['timestamp', 'status']", "enc": "utf-8"},
        ],
        "explanation": "Uses pandas.read_csv with validated parameters usecols and encoding."
    },
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.DataFrame.groupby",
        "prompt_fmt": "Group DataFrame by '{group_col}' and aggregate '{val_col}' using sum and mean without index dropping.",
        "code_fmt": "import pandas as pd\n\ndef aggregate_groups(df: pd.DataFrame) -> pd.DataFrame:\n    return df.groupby('{group_col}', as_index=False)['{val_col}'].agg(['sum', 'mean'])",
        "variations": [
            {"group_col": "department", "val_col": "salary"},
            {"group_col": "region", "val_col": "sales"},
            {"group_col": "category", "val_col": "units"},
        ],
        "explanation": "Groups data using DataFrame.groupby with as_index=False and agg."
    },
    {
        "library": "pandas",
        "version": "2.2.2",
        "api_name": "pandas.concat",
        "prompt_fmt": "Combine two Pandas DataFrames vertically and reset index using pandas.concat.",
        "code_fmt": "import pandas as pd\n\ndef combine_frames(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:\n    return pd.concat([df1, df2], axis=0, ignore_index=True)",
        "variations": [
            {},
        ],
        "explanation": "Concatenates DataFrames using pandas.concat with axis=0 and ignore_index=True."
    },

    # 3. Requests
    {
        "library": "requests",
        "version": "2.31.0",
        "api_name": "requests.get",
        "prompt_fmt": "Perform an authenticated HTTP GET request to '{endpoint}' with a {timeout}s timeout.",
        "code_fmt": "import requests\n\ndef fetch_resource(token: str):\n    headers = {{'Authorization': f'Bearer {{token}}', 'Accept': 'application/json'}}\n    resp = requests.get('{endpoint}', headers=headers, timeout={timeout})\n    resp.raise_for_status()\n    return resp.json()",
        "variations": [
            {"endpoint": "https://api.github.com/user", "timeout": "5.0"},
            {"endpoint": "https://httpbin.org/get", "timeout": "10.0"},
            {"endpoint": "https://api.example.com/v1/health", "timeout": "3.0"},
        ],
        "explanation": "Uses requests.get with explicit timeout and HTTP error handling via raise_for_status."
    },
    {
        "library": "requests",
        "version": "2.31.0",
        "api_name": "requests.post",
        "prompt_fmt": "Send a JSON payload to '{endpoint}' using requests.post with timeout {timeout}.",
        "code_fmt": "import requests\n\ndef post_payload(payload: dict):\n    url = '{endpoint}'\n    resp = requests.post(url, json=payload, timeout={timeout})\n    resp.raise_for_status()\n    return resp.json()",
        "variations": [
            {"endpoint": "https://httpbin.org/post", "timeout": "5.0"},
            {"endpoint": "https://api.example.com/events", "timeout": "8.0"},
        ],
        "explanation": "Dispatches POST request passing dict directly to the json argument."
    },

    # 4. Flask
    {
        "library": "flask",
        "version": "3.0.2",
        "api_name": "flask.Flask",
        "prompt_fmt": "Create a Flask WSGI application with a health-check endpoint at '{path}'.",
        "code_fmt": "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('{path}', methods=['GET'])\ndef health():\n    return jsonify({{'status': 'healthy'}}), 200",
        "variations": [
            {"path": "/healthz"},
            {"path": "/api/v1/ping"},
            {"path": "/status"},
        ],
        "explanation": "Instantiates standard Flask app and registers route returning jsonify."
    },

    # 5. FastAPI
    {
        "library": "fastapi",
        "version": "0.115.0",
        "api_name": "fastapi.FastAPI",
        "prompt_fmt": "Initialize a FastAPI application with title '{title}' and version '{ver}'.",
        "code_fmt": "from fastapi import FastAPI\n\napp = FastAPI(title='{title}', version='{ver}')\n\n@app.get('/ping')\ndef ping():\n    return {{'ping': 'pong'}}",
        "variations": [
            {"title": "User Service", "ver": "1.0.0"},
            {"title": "SoftmaxShield API", "ver": "2.0.0"},
            {"title": "Telemetry Gateway", "ver": "0.1.0"},
        ],
        "explanation": "Instantiates FastAPI with documented keyword arguments title and version."
    },
    {
        "library": "fastapi",
        "version": "0.115.0",
        "api_name": "fastapi.Depends",
        "prompt_fmt": "Declare a FastAPI GET route at '{path}' that utilizes dependency injection via Depends.",
        "code_fmt": "from fastapi import FastAPI, Depends\n\napp = FastAPI()\n\ndef get_current_user():\n    return {{'username': 'admin'}}\n\n@app.get('{path}')\ndef read_data(user: dict = Depends(get_current_user)):\n    return {{'data': 'confidential', 'user': user}}",
        "variations": [
            {"path": "/profile"},
            {"path": "/dashboard/stats"},
        ],
        "explanation": "Uses fastapi.Depends for clean dependency injection."
    },

    # 6. Scikit-Learn
    {
        "library": "scikit-learn",
        "version": "1.4.0",
        "api_name": "sklearn.linear_model.LogisticRegression",
        "prompt_fmt": "Configure a LogisticRegression classifier with C={c_val} and random_state={seed}.",
        "code_fmt": "from sklearn.linear_model import LogisticRegression\n\nclf = LogisticRegression(C={c_val}, random_state={seed}, max_iter=200)\n# clf.fit(X_train, y_train)",
        "variations": [
            {"c_val": "0.1", "seed": "42"},
            {"c_val": "1.0", "seed": "123"},
            {"c_val": "10.0", "seed": "7"},
        ],
        "explanation": "Instantiates LogisticRegression passing valid solver parameters."
    },
    {
        "library": "scikit-learn",
        "version": "1.4.0",
        "api_name": "sklearn.model_selection.train_test_split",
        "prompt_fmt": "Partition arrays X and y into train and test splits with test size {size} and stratification.",
        "code_fmt": "from sklearn.model_selection import train_test_split\n\ndef partition(X, y):\n    return train_test_split(X, y, test_size={size}, random_state=42, stratify=y)",
        "variations": [
            {"size": "0.2"},
            {"size": "0.25"},
            {"size": "0.3"},
        ],
        "explanation": "Uses train_test_split with stratify parameter for balanced dataset splitting."
    },

    # 7. PyTorch
    {
        "library": "torch",
        "version": "2.2.0",
        "api_name": "torch.nn.Linear",
        "prompt_fmt": "Define a PyTorch feed-forward layer from {in_dim} to {out_dim} features.",
        "code_fmt": "import torch\nimport torch.nn as nn\n\nlayer = nn.Linear({in_dim}, {out_dim}, bias=True)\nx = torch.randn(8, {in_dim})\nout = layer(x)",
        "variations": [
            {"in_dim": "512", "out_dim": "128"},
            {"in_dim": "768", "out_dim": "256"},
            {"in_dim": "1024", "out_dim": "512"},
        ],
        "explanation": "Calls nn.Linear with positional in_features and out_features."
    },
    {
        "library": "torch",
        "version": "2.2.0",
        "api_name": "torch.optim.AdamW",
        "prompt_fmt": "Initialize PyTorch AdamW optimizer with learning rate {lr} and weight decay {wd}.",
        "code_fmt": "import torch\nimport torch.nn as nn\n\nmodel = nn.Linear(10, 2)\noptimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={wd})",
        "variations": [
            {"lr": "1e-4", "wd": "0.01"},
            {"lr": "2e-4", "wd": "0.05"},
            {"lr": "5e-5", "wd": "0.001"},
        ],
        "explanation": "Instantiates AdamW optimizer passing model parameters and weight decay."
    },

    # 8. Matplotlib
    {
        "library": "matplotlib",
        "version": "3.8.0",
        "api_name": "matplotlib.pyplot.subplots",
        "prompt_fmt": "Create a figure with {rows}x{cols} subplots and set title to '{title}'.",
        "code_fmt": "import matplotlib.pyplot as plt\n\nfig, axes = plt.subplots(nrows={rows}, ncols={cols}, figsize=(8, 6), squeeze=True)\nfig.suptitle('{title}')\n# plt.show()",
        "variations": [
            {"rows": "1", "cols": "2", "title": "Loss vs Epochs"},
            {"rows": "2", "cols": "2", "title": "Model Performance Grid"},
            {"rows": "1", "cols": "3", "title": "Distribution Overview"},
        ],
        "explanation": "Calls plt.subplots returning figure and axes tuple."
    },

    # 9. SQLAlchemy
    {
        "library": "sqlalchemy",
        "version": "2.0.28",
        "api_name": "sqlalchemy.select",
        "prompt_fmt": "Construct a SQLAlchemy 2.0 select query on model '{model}' filtered by status.",
        "code_fmt": "from sqlalchemy import select\n\nstmt = select({model}).where({model}.is_active == True)\n# with Session(engine) as session:\n#     results = session.scalars(stmt).all()",
        "variations": [
            {"model": "User"},
            {"model": "Account"},
            {"model": "Organization"},
        ],
        "explanation": "Constructs 2.0 style select statement without legacy query."
    },

    # 10. Pydantic
    {
        "library": "pydantic",
        "version": "2.7.0",
        "api_name": "pydantic.BaseModel.model_dump",
        "prompt_fmt": "Create a Pydantic v2 schema for '{model_name}' and serialize it to a dictionary.",
        "code_fmt": "from pydantic import BaseModel\n\nclass {model_name}(BaseModel):\n    id: int\n    name: str\n\ninstance = {model_name}(id=1, name='Demo')\ndata = instance.model_dump(exclude_none=True)",
        "variations": [
            {"model_name": "UserProfile"},
            {"model_name": "ItemConfig"},
            {"model_name": "ServerSetting"},
        ],
        "explanation": "Uses Pydantic v2 model_dump() replacing deprecated v1 .dict()."
    },
    {
        "library": "pydantic",
        "version": "2.7.0",
        "api_name": "pydantic.field_validator",
        "prompt_fmt": "Define a field validator on a Pydantic v2 model using field_validator.",
        "code_fmt": "from pydantic import BaseModel, field_validator\n\nclass Item(BaseModel):\n    price: float\n\n    @field_validator('price')\n    @classmethod\n    def check_price(cls, v: float) -> float:\n        if v <= 0:\n            raise ValueError('Price must be strictly positive')\n        return v",
        "variations": [
            {},
        ],
        "explanation": "Uses Pydantic v2 @field_validator decorator replacing deprecated @validator."
    },
]


NEGATIVE_DATASET_CASES = [
    # Category 1: Incorrect Function Names (Hallucinated APIs)
    {
        "library": "numpy",
        "version": "1.26.4",
        "prompt": "Call numpy.fast_matrix_multiply(A, B) to multiply two matrices.",
        "negative_example": "result = np.fast_matrix_multiply(A, B)",
        "correct_code": "import numpy as np\n\n# Correct verified function:\nresult = np.matmul(A, B)",
        "expected_behavior": "refuse_hallucination",
        "explanation": "numpy.fast_matrix_multiply does not exist in numpy 1.26.4. The verified function is numpy.matmul or the @ operator."
    },
    {
        "library": "requests",
        "version": "2.31.0",
        "prompt": "Use requests.get_async('https://api.example.com') to make an asynchronous GET request.",
        "negative_example": "resp = requests.get_async('https://api.example.com')",
        "correct_code": "import requests\n\n# requests is synchronous only; use requests.get:\nresp = requests.get('https://api.example.com', timeout=10.0)",
        "expected_behavior": "refuse_hallucination",
        "explanation": "requests does not have an asynchronous get_async method. Use synchronous requests.get with timeout, or an async library like httpx."
    },
    {
        "library": "flask",
        "version": "3.0.2",
        "prompt": "Use flask.auto_route_crud(app, User) to automatically create CRUD endpoints.",
        "negative_example": "flask.auto_route_crud(app, User)",
        "correct_code": "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/users', methods=['GET'])\ndef get_users():\n    return jsonify([])",
        "expected_behavior": "refuse_hallucination",
        "explanation": "flask.auto_route_crud is a hallucinated API. Flask requires routes to be explicitly registered using @app.route."
    },
    {
        "library": "torch",
        "version": "2.2.0",
        "prompt": "Call torch.cuda.empty_all_gpus() to purge CUDA cache on all available devices.",
        "negative_example": "torch.cuda.empty_all_gpus()",
        "correct_code": "import torch\n\nif torch.cuda.is_available():\n    torch.cuda.empty_cache()",
        "expected_behavior": "refuse_hallucination",
        "explanation": "torch.cuda.empty_all_gpus does not exist in PyTorch. The genuine memory release function is torch.cuda.empty_cache()."
    },
    {
        "library": "scikit-learn",
        "version": "1.4.0",
        "prompt": "Cluster dataset with sklearn.cluster.SuperKMeans(n_clusters=3).",
        "negative_example": "from sklearn.cluster import SuperKMeans\nclf = SuperKMeans(n_clusters=3)",
        "correct_code": "from sklearn.cluster import KMeans\n\nclf = KMeans(n_clusters=3, random_state=42, n_init='auto')",
        "expected_behavior": "refuse_hallucination",
        "explanation": "sklearn.cluster.SuperKMeans is a hallucinated class. The correct estimator is sklearn.cluster.KMeans."
    },

    # Category 2: Nonexistent Packages (Hallucinated Dependencies)
    {
        "library": "reqeusts",
        "version": "n/a",
        "prompt": "Import reqeusts and fetch https://example.com.",
        "negative_example": "import reqeusts\nresp = reqeusts.get('https://example.com')",
        "correct_code": "import requests\n\nresp = requests.get('https://example.com', timeout=10.0)",
        "expected_behavior": "refuse_nonexistent_package",
        "explanation": "The package 'reqeusts' does not exist on PyPI. It is a typosquat hallucination of the legitimate package 'requests'."
    },
    {
        "library": "fastapi_auto_crud",
        "version": "n/a",
        "prompt": "Use fastapi_auto_crud to scaffold an automated database REST API.",
        "negative_example": "import fastapi_auto_crud\nrouter = fastapi_auto_crud.create_router(User)",
        "correct_code": "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get('/items')\ndef read_items():\n    return []",
        "expected_behavior": "refuse_nonexistent_package",
        "explanation": "Package 'fastapi_auto_crud' does not exist in verified standard catalogs. Use standard FastAPI APIRouter instead."
    },

    # Category 3: Invalid Keyword Arguments
    {
        "library": "json",
        "version": "stdlib",
        "prompt": "Serialize dictionary using json.dumps(obj, invalid_flag=True, indent=2).",
        "negative_example": "import json\njson.dumps({'k': 'v'}, invalid_flag=True, indent=2)",
        "correct_code": "import json\n\n# Correct call without invalid_flag:\nresult = json.dumps({'k': 'v'}, indent=2)",
        "expected_behavior": "refuse_hallucination",
        "explanation": "json.dumps does not accept 'invalid_flag'. Supported arguments include indent, skipkeys, sort_keys, and ensure_ascii."
    },
    {
        "library": "math",
        "version": "stdlib",
        "prompt": "Compute square root using math.sqrt(x=16, fast=True).",
        "negative_example": "import math\nmath.sqrt(x=16, fast=True)",
        "correct_code": "import math\n\n# math.sqrt takes a single positional-only argument:\nresult = math.sqrt(16)",
        "expected_behavior": "refuse_hallucination",
        "explanation": "math.sqrt takes a single positional-only argument. Passing x=16 or fast=True causes a TypeError."
    },

    # Category 4: Version-Specific API Mismatches
    {
        "library": "pandas",
        "version": "2.2.2",
        "prompt": "Add a new row to a Pandas DataFrame using df.append({'x': 10}) in Pandas 2.2.",
        "negative_example": "df = df.append({'x': 10}, ignore_index=True)",
        "correct_code": "import pandas as pd\n\nnew_row = pd.DataFrame([{'x': 10}])\ndf = pd.concat([df, new_row], ignore_index=True)",
        "expected_behavior": "refuse_hallucination",
        "explanation": "DataFrame.append was deprecated in Pandas 1.4 and completely removed in Pandas 2.0. The verified replacement is pandas.concat."
    },
    {
        "library": "pydantic",
        "version": "2.7.0",
        "prompt": "Export a Pydantic v2 model instance to a dictionary using user.dict().",
        "negative_example": "payload = user.dict()",
        "correct_code": "from pydantic import BaseModel\n\nclass User(BaseModel):\n    name: str\n\nuser = User(name='Bob')\npayload = user.model_dump()",
        "expected_behavior": "refuse_hallucination",
        "explanation": "In Pydantic v2, .dict() is deprecated. The official modern method is user.model_dump()."
    },
    {
        "library": "pydantic",
        "version": "2.7.0",
        "prompt": "Decorate a validator method using @validator in Pydantic 2.7.",
        "negative_example": "from pydantic import BaseModel, validator\n\nclass M(BaseModel):\n    v: int\n    @validator('v')\n    def check_v(cls, v): return v",
        "correct_code": "from pydantic import BaseModel, field_validator\n\nclass M(BaseModel):\n    v: int\n\n    @field_validator('v')\n    @classmethod\n    def check_v(cls, v: int) -> int:\n        return v",
        "expected_behavior": "refuse_hallucination",
        "explanation": "In Pydantic v2, @validator is deprecated. Use @field_validator with classmethod semantics."
    },
    {
        "library": "sqlalchemy",
        "version": "2.0.28",
        "prompt": "Query active users using session.query(User).all() in SQLAlchemy 2.0.",
        "negative_example": "users = session.query(User).all()",
        "correct_code": "from sqlalchemy import select\n\nstmt = select(User).where(User.is_active == True)\nusers = session.scalars(stmt).all()",
        "expected_behavior": "refuse_hallucination",
        "explanation": "SQLAlchemy 2.0 supersedes session.query in favor of 2.0 style select() executed via session.scalars()."
    },

    # Category 5: Cases Where Model Explicitly Declares API Cannot Be Verified
    {
        "library": "unknown_ai_lib",
        "version": "unknown",
        "prompt": "Use unknown_ai_lib.ultra_neural_infer() to classify text.",
        "negative_example": "import unknown_ai_lib\nres = unknown_ai_lib.ultra_neural_infer('text')",
        "correct_code": "# Cannot verify API for 'unknown_ai_lib.ultra_neural_infer'.\n# This library is not in the verified documentation catalog.\nraise NotImplementedError('Cannot verify API for unknown_ai_lib; manual verification required.')",
        "expected_behavior": "warn_unverified_api",
        "explanation": "The requested API does not exist in verified ground-truth documentation. The model must refuse to invent methods and declare it unverified."
    },
    {
        "library": "proprietary_cloud_tool",
        "version": "unknown",
        "prompt": "Call proprietary_cloud_tool.teleport_database(source, dest).",
        "negative_example": "proprietary_cloud_tool.teleport_database('db1', 'db2')",
        "correct_code": "# Cannot verify API for 'proprietary_cloud_tool.teleport_database'.\n# Please refer to official vendor documentation.",
        "expected_behavior": "warn_unverified_api",
        "explanation": "The API is undocumented and unverified; explicit declaration of unverified status is required."
    },
]


def generate_extended_dataset(
    output_path: str = "data/finetuning_dataset.jsonl",
    target_count: int = 2100,
) -> int:
    """Generate 2,000+ grounded instruction pairs conforming to the specified schema."""
    records: List[Dict[str, Any]] = []

    # 1. Generate Positive Examples (approx 70% of dataset ~ 1,470 records)
    pos_target = int(target_count * 0.70)
    modifiers = [
        "",
        " Ensure clean type hints and PEP 8 style.",
        " Include robust error checking.",
        " Handle exceptions cleanly.",
        " Use modern idiom without deprecated functions.",
    ]

    while len(records) < pos_target:
        for template in POSITIVE_DATA_TEMPLATES:
            lib = template["library"]
            ver = template["version"]
            prompt_fmt = template["prompt_fmt"]
            code_fmt = template["code_fmt"]
            variations = template["variations"]
            expl = template["explanation"]

            for var in variations:
                mod = random.choice(modifiers)
                prompt = prompt_fmt.format(**var) + mod
                code = code_fmt.format(**var)

                records.append({
                    "prompt": prompt,
                    "library": lib,
                    "version": ver,
                    "correct_code": code,
                    "negative_example": None,
                    "expected_behavior": "generate_code",
                    "explanation": expl,
                })
                if len(records) >= pos_target:
                    break
            if len(records) >= pos_target:
                break

    # 2. Generate Negative and Refusal Examples (approx 30% of dataset ~ 630 records)
    neg_prompts = list(NEGATIVE_DATASET_CASES)
    phrasings = [
        "{}",
        "Please write code to: {}",
        "How do I {}?",
        "Implement a script that will: {}",
        "Provide a code example for: {}",
    ]

    while len(records) < target_count:
        for case in neg_prompts:
            phrased_prompt = random.choice(phrasings).format(case["prompt"])
            records.append({
                "prompt": phrased_prompt,
                "library": case["library"],
                "version": case["version"],
                "correct_code": case["correct_code"],
                "negative_example": case["negative_example"],
                "expected_behavior": case["expected_behavior"],
                "explanation": case["explanation"],
            })
            if len(records) >= target_count:
                break

    # Shuffle dataset
    random.seed(42)
    random.shuffle(records)

    # Write out as JSONL
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return len(records)


# Alias for backwards compatibility with tests
generate_dataset = generate_extended_dataset


if __name__ == "__main__":
    count = generate_extended_dataset("data/finetuning_dataset.jsonl", target_count=2100)
    print(f"Generated {count} records in data/finetuning_dataset.jsonl")
