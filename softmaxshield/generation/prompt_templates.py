"""Prompt engineering templates for generation-side hallucination prevention: Constrained, Few-Shot, and Chain-of-Verification."""

from __future__ import annotations

import enum
from typing import List, Optional

from softmaxshield.generation.docs_extractor import DocChunk


class PromptStrategy(str, enum.Enum):
    """Enumeration of generation-side prompt defense strategies."""

    CONSTRAINED = "CONSTRAINED"
    FEW_SHOT = "FEW_SHOT"
    CHAIN_OF_VERIFICATION = "CHAIN_OF_VERIFICATION"


def format_prompt_with_retrieved_docs(retrieved_chunks: List[DocChunk]) -> str:
    """Format retrieved documentation chunks into a context block with explicit metadata and missing doc guidance."""
    if not retrieved_chunks:
        return (
            "=== RETRIEVED TRUSTED API DOCUMENTATION ===\n"
            "NOTICE: No specific documentation chunks were found matching this query.\n"
            "RULE: Do NOT invent fictional APIs or parameter names. If an API is unknown, use standard library primitives or declare it unverified.\n"
            "============================================"
        )
    lines = ["=== RETRIEVED TRUSTED API DOCUMENTATION (Use these exact signatures) ==="]
    for idx, c in enumerate(retrieved_chunks, 1):
        lines.append(f"[{idx}] {c.to_prompt_context()}")
    lines.append("=========================================================================")
    return "\n".join(lines)


# 1. Constrained / Closed-World Prompt Template
def build_constrained_prompt(
    user_query: str,
    allowed_libraries: List[str],
    retrieved_chunks: Optional[List[DocChunk]] = None,
) -> str:
    """Generate a constrained closed-world prompt restricting the LLM to verified libraries and documented APIs."""
    allowed_list_str = ", ".join(f"`{lib}`" for lib in allowed_libraries)
    docs_block = format_prompt_with_retrieved_docs(retrieved_chunks or [])

    return f"""You are a secure Python software engineer operating under the SoftmaxShield verification boundary.

### STRICT GENERATION CONSTRAINTS (CLOSED-WORLD INSTRUCTION)
1. You are strictly restricted to using ONLY the following approved libraries: {allowed_list_str} and the standard Python library.
2. Do NOT invent, assume, or hallucinate nonexistent packages, submodules, functions, classes, or keyword arguments.
3. Every external function call must correspond to an actual, documented API matching the retrieved documentation below.
4. UNVERIFIED API CLAUSE: If you cannot verify an API from the documentation or standard library, you are EXPLICITLY PERMITTED and REQUIRED to state:
   "Cannot verify API for [capability]; manual verification required" rather than inventing a hallucinated method.
5. Return ONLY executable Python source code without conversational filler.

{docs_block}

### USER CODING REQUEST:
{user_query}

### PYTHON IMPLEMENTATION:
"""


# 2. Few-Shot Grounded Prompt Template with Incorrect vs Corrected API Examples
FEW_SHOT_DEMONSTRATIONS = """
### EXAMPLE 1: Pydantic v2 Migration & Field Validation
User: Create a Pydantic v2 schema for an API user with email validation and export it to a dictionary.
Retrieved Signature: [pydantic==2.7.0] pydantic.BaseModel.model_dump(self, *, mode='python', include=None, exclude=None, ...)
Retrieved Signature: [pydantic==2.7.0] pydantic.field_validator(*fields, mode='after', check_fields=None)
[INCORRECT API PATTERN TO AVOID]:
- user.dict()  # DEPRECATED/REMOVED in Pydantic v2
- @validator('email')  # DEPRECATED/REMOVED in Pydantic v2
[CORRECTED VERIFIED CODE]:
```python
from pydantic import BaseModel, field_validator

class UserCreate(BaseModel):
    id: int
    email: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

user = UserCreate(id=1, email='User@Example.com')
# Correct Pydantic v2 method:
payload = user.model_dump()
```

### EXAMPLE 2: FastAPI + SQLAlchemy 2.0 Modern ORM Querying
User: Write a FastAPI route to fetch active users using SQLAlchemy 2.0.
Retrieved Signature: [fastapi==0.115.0] fastapi.FastAPI(*, title='FastAPI', ...)
Retrieved Signature: [sqlalchemy==2.0.28] sqlalchemy.select(*entities)
Retrieved Signature: [sqlalchemy==2.0.28] sqlalchemy.orm.Session.scalars(self, statement, ...)
[INCORRECT API PATTERN TO AVOID]:
- session.query(User).filter(User.is_active == True).all()  # Legacy 1.x pattern, superseded in 2.0
[CORRECTED VERIFIED CODE]:
```python
from fastapi import FastAPI, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

app = FastAPI(title="User API")

@app.get("/users")
def get_active_users(db: Session = Depends(get_db)):
    # Correct SQLAlchemy 2.0 select and scalars pattern:
    stmt = select(User).where(User.is_active == True)
    return db.scalars(stmt).all()
```

### EXAMPLE 3: Pandas 2.x Safe Row Operations
User: Append a new observation dictionary to an existing DataFrame in Pandas 2.2.
Retrieved Signature: [pandas==2.2.2] pandas.concat(objs, *, axis=0, ignore_index=False, ...)
[INCORRECT API PATTERN TO AVOID]:
- df = df.append({'metric': 42}, ignore_index=True)  # REMOVED from Pandas 2.0+
[CORRECTED VERIFIED CODE]:
```python
import pandas as pd

def append_observation(df: pd.DataFrame, new_data: dict) -> pd.DataFrame:
    # Correct replacement for removed DataFrame.append():
    new_row = pd.DataFrame([new_data])
    return pd.concat([df, new_row], ignore_index=True)
```

### EXAMPLE 4: Synchronous vs Asynchronous Network Calls
User: Asynchronously fetch data from a JSON endpoint using requests.
Retrieved Signature: [requests==2.31.0] requests.get(url, params=None, *, timeout=None, ...)
[INCORRECT API PATTERN TO AVOID]:
- resp = await requests.get_async('https://api.example.com')  # requests does NOT have get_async
[CORRECTED VERIFIED CODE]:
```python
import requests

def fetch_json(url: str) -> dict:
    # requests is synchronous only; specify timeout to avoid hanging
    resp = requests.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.json()
```
"""


def build_few_shot_prompt(
    user_query: str,
    retrieved_chunks: Optional[List[DocChunk]] = None,
) -> str:
    """Generate a few-shot grounded prompt providing verified modern API patterns and contrastive corrections."""
    docs_block = format_prompt_with_retrieved_docs(retrieved_chunks or [])

    return f"""You are an expert Python engineer adhering strictly to verified, modern API conventions.
Observe the demonstrations below, noting how every API call uses genuine signatures and explicitly avoids hallucinated or deprecated functions.

{FEW_SHOT_DEMONSTRATIONS}

{docs_block}

### TARGET TASK:
User: {user_query}
[CORRECTED VERIFIED CODE]:
"""


# 3. Chain-of-Verification (CoVe) Prompt Template
def build_chain_of_verification_prompt(
    user_query: str,
    retrieved_chunks: Optional[List[DocChunk]] = None,
) -> str:
    """Generate a Chain-of-Verification prompt enforcing a 4-step self-audit before emitting code."""
    docs_block = format_prompt_with_retrieved_docs(retrieved_chunks or [])

    return f"""You are a verified code generation agent running under the SoftmaxShield protocol.
Before outputting code, you must execute a 4-phase Chain-of-Verification (CoVe) audit to prevent hallucinations.

{docs_block}

Follow this exact structure:

PHASE 1: DRAFT IMPLEMENTATION PLAN
- Outline the logic and packages you plan to use.

PHASE 2: IDENTIFIER INVENTORY & CHECKLIST
- List every external import you plan to include.
- List every function, class, or method call you plan to invoke.
- List all keyword arguments you plan to pass.

PHASE 3: STATIC API VERIFICATION AUDIT
- For each item in Phase 2, verify it against the retrieved documentation above.
- Confirm whether the function exists in the specified library version.
- Confirm whether each keyword argument is valid or disallowed (e.g. check for positional-only arguments).
- If any function or argument is missing from the docs, state: "DISALLOWED: Replacing with standard verified alternative."

PHASE 4: FINAL VERIFIED PYTHON CODE
- Produce only the clean, verified Python code incorporating the audit findings.

### USER QUERY:
{user_query}
"""
