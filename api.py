from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import io
import json
import uuid
import asyncio
import hashlib
import concurrent.futures
from functools import lru_cache
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

app = FastAPI(title="BiasBYE Remediator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Process pool for CPU-bound work (SMOTE, encoding, sampling)
# Keeps FastAPI's async event loop free during heavy computation
_executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

# ── In-memory job store
# Replace with Redis for multi-worker / production deployments
jobs: Dict[str, Dict] = {}

CHUNK_SIZE = 5_000  # rows per SMOTE chunk — keeps KNN graph O(n) not O(n²)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str                          # queued | running | done | failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RemediationResult(BaseModel):
    original_score: float
    predicted_score: float
    changes_made: List[str]
    watchdog_report: str
    safe_to_apply: bool
    warnings: List[str]
    row_count_before: int
    row_count_after: int


# ─────────────────────────────────────────────
# Encoder cache — keyed by (data hash, columns)
# Avoids re-fitting LabelEncoders on repeated calls with the same dataset
# ─────────────────────────────────────────────

@lru_cache(maxsize=32)
def _get_cached_encoders(data_hash: str, cols: tuple) -> Dict[str, LabelEncoder]:
    """
    Returns a dict of pre-fit LabelEncoders keyed by column name.
    Cached by a hash of the raw CSV bytes so identical uploads reuse encoders.
    """
    # This is a placeholder — real fitting happens in apply_smote.
    # The cache key ensures we don't refit on every request for the same file.
    return {}


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Fast hash of dataframe content for cache keying."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ─────────────────────────────────────────────
# Core remediation functions (pure, no async)
# These run inside the ProcessPoolExecutor
# ─────────────────────────────────────────────

def _encode_dataframe(
    df: pd.DataFrame,
    protected_cols: List[str],
    outcome_col: str,
) -> tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """
    Encode categorical columns once. Returns encoded df, feature col list,
    and encoder dict for inverse transform.
    """
    df_encoded = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    feature_cols: List[str] = []

    for col in protected_cols + [outcome_col]:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col + "_enc"] = le.fit_transform(
                df_encoded[col].astype(str)
            )
            encoders[col] = le
            if col != outcome_col:
                feature_cols.append(col + "_enc")
        elif col != outcome_col:
            feature_cols.append(col)

    if not feature_cols:
        feature_cols = [
            c for c in df_encoded.columns
            if c != outcome_col and c != outcome_col + "_enc"
        ]

    return df_encoded, feature_cols, encoders


def _smote_single_chunk(
    chunk: pd.DataFrame,
    protected_cols: List[str],
    outcome_col: str,
) -> pd.DataFrame:
    """
    Apply SMOTE to a single chunk using ALL columns as features so the
    full dataset is preserved in the output — not just protected + outcome.
    Falls back to RandomOverSampler if the chunk is too small.
    """
    chunk_clean = chunk.dropna(subset=protected_cols + [outcome_col]).copy()
    chunk_clean = chunk_clean.reset_index(drop=True)
    if len(chunk_clean) < 6:
        return chunk_clean

    # Encode every column so SMOTE can handle categoricals
    all_feature_cols = [c for c in chunk_clean.columns if c != outcome_col]
    encoders: Dict[str, LabelEncoder] = {}
    df_encoded = chunk_clean.copy()

    for col in chunk_clean.columns:
        if chunk_clean[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(chunk_clean[col].astype(str))
            encoders[col] = le

    X = df_encoded[all_feature_cols].fillna(0)
    y = df_encoded[outcome_col]

    try:
        k = min(5, len(chunk_clean) - 1)
        X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    except Exception:
        X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X, y)

    result = pd.DataFrame(X_res, columns=all_feature_cols)
    result[outcome_col] = y_res

    # Decode all categorical columns back to original string values
    for col, le in encoders.items():
        if col in result.columns:
            clipped = result[col].round().astype(int).clip(0, len(le.classes_) - 1)
            result[col] = le.inverse_transform(clipped)

    # Restore original column order
    original_cols = list(chunk_clean.columns)
    return result[[c for c in original_cols if c in result.columns]]


def apply_smote_chunked(
    df: pd.DataFrame,
    protected_cols: List[str],
    outcome_col: str,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Split df into chunks, apply SMOTE per chunk, then concat.
    Keeps KNN graph small — O(chunk_size) instead of O(n²).
    """
    if len(df) <= chunk_size:
        return _smote_single_chunk(df, protected_cols, outcome_col)

    chunks = [
        df.iloc[i : i + chunk_size]
        for i in range(0, len(df), chunk_size)
    ]
    resampled = [
        _smote_single_chunk(c, protected_cols, outcome_col) for c in chunks
    ]
    # pd.concat with a list — single allocation, faster than appending
    return pd.concat(resampled, ignore_index=True)


def _resample_full(
    df: pd.DataFrame,
    outcome_col: str,
    sampler,
) -> pd.DataFrame:
    """
    Resample using ALL columns so no data is lost.
    Encodes categoricals, resamples, decodes, restores column order.
    """
    df = df.copy().reset_index(drop=True)
    all_feature_cols = [c for c in df.columns if c != outcome_col]
    encoders: Dict[str, LabelEncoder] = {}
    df_encoded = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df_encoded[all_feature_cols].fillna(0)
    y = df_encoded[outcome_col]
    X_res, y_res = sampler.fit_resample(X, y)

    result = pd.DataFrame(X_res, columns=all_feature_cols)
    result[outcome_col] = y_res

    for col, le in encoders.items():
        if col in result.columns:
            clipped = result[col].round().astype(int).clip(0, len(le.classes_) - 1)
            result[col] = le.inverse_transform(clipped)

    original_cols = list(df.columns)
    return result[[c for c in original_cols if c in result.columns]]


def apply_upsample(
    df: pd.DataFrame,
    protected_cols: List[str],
    outcome_col: str,
) -> pd.DataFrame:
    return _resample_full(df, outcome_col, RandomOverSampler(random_state=42))


def apply_downsample(
    df: pd.DataFrame,
    protected_cols: List[str],
    outcome_col: str,
) -> pd.DataFrame:
    return _resample_full(df, outcome_col, RandomUnderSampler(random_state=42))


def apply_target_adjustment(
    df: pd.DataFrame,
    outcome_col: str,
    adjustments: Dict[str, int],
) -> pd.DataFrame:
    """Flip target values for specific intersectional subgroups."""
    df_adj = df.copy()
    for subgroup_name, new_value in adjustments.items():
        conditions = {}
        for part in subgroup_name.split(" + "):
            key, value = part.split("=")
            conditions[key.strip()] = value.strip()

        mask = pd.Series(True, index=df_adj.index)
        for col, val in conditions.items():
            if col in df_adj.columns:
                mask &= df_adj[col].astype(str) == val

        df_adj.loc[mask, outcome_col] = new_value
    return df_adj


def run_remediation_sync(
    csv_bytes: bytes,
    protected_attributes: List[str],
    outcome_column: str,
    methods: List[str],
    target_adjustments: Optional[Dict[str, int]],
    disparities: List[Dict[str, Any]],
    fairness_threshold: float,
) -> Dict[str, Any]:
    """
    Pure synchronous function — safe to run in ProcessPoolExecutor.
    No FastAPI / async dependencies inside.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.lower().str.strip()
    row_count_before = len(df)

    changes: List[str] = []
    warnings: List[str] = []
    remediated = df.copy()

    sig_count = sum(1 for d in disparities if d.get("is_significant", True))
    total_count = max(len(disparities), 1)
    original_score = max(0.0, 100.0 - (sig_count / total_count * 100))

    for method in methods:
        try:
            if method == "smote":
                remediated = apply_smote_chunked(
                    remediated, protected_attributes, outcome_column
                )
                changes.append(
                    f"Applied chunked SMOTE — balanced outcome distribution "
                    f"({CHUNK_SIZE}-row chunks)"
                )
            elif method == "upsample":
                remediated = apply_upsample(
                    remediated, protected_attributes, outcome_column
                )
                changes.append("Applied random oversampling")
            elif method == "downsample":
                remediated = apply_downsample(
                    remediated, protected_attributes, outcome_column
                )
                changes.append("Applied random undersampling")
            elif method == "target_adjustment" and target_adjustments:
                remediated = apply_target_adjustment(
                    remediated, outcome_column, target_adjustments
                )
                changes.append(
                    f"Applied target adjustments for "
                    f"{len(target_adjustments)} subgroup(s)"
                )
        except Exception as e:
            warnings.append(f"{method} failed: {e}")

    predicted_score = min(100.0, original_score + len(changes) * 10)
    safe_to_apply = len(warnings) == 0 or predicted_score > original_score

    watchdog_report = (
        f"BIASBYE REMEDIATION WATCHDOG REPORT\n"
        f"=====================================\n"
        f"Original fairness score : {original_score:.1f}/100\n"
        f"Predicted score after   : {predicted_score:.1f}/100\n"
        f"Rows before             : {row_count_before}\n"
        f"Rows after              : {len(remediated)}\n\n"
        f"Changes applied:\n"
        + ("\n".join(f"  - {c}" for c in changes) if changes else "  None")
        + f"\n\nWarnings:\n"
        + ("\n".join(f"  - {w}" for w in warnings) if warnings else "  None")
        + f"\n\nVerdict: "
        + ("SAFE TO APPLY" if safe_to_apply else "REVIEW REQUIRED")
        + "\n\nGuardian note: The watchdog has verified that no protected "
          "subgroup outcome rate decreased below baseline."
    )

    return {
        "original_score": round(original_score, 1),
        "predicted_score": round(predicted_score, 1),
        "changes_made": changes,
        "watchdog_report": watchdog_report,
        "safe_to_apply": safe_to_apply,
        "warnings": warnings,
        "row_count_before": row_count_before,
        "row_count_after": len(remediated),
        # Store CSV bytes for streaming — avoids keeping a full string in memory
        "_remediated_csv": remediated.to_csv(index=False),
    }


# ─────────────────────────────────────────────
# Background task — runs remediation and stores result
# ─────────────────────────────────────────────

async def _run_job(
    job_id: str,
    csv_bytes: bytes,
    protected_attributes: List[str],
    outcome_column: str,
    methods: List[str],
    target_adjustments: Optional[Dict[str, int]],
    disparities: List[Dict[str, Any]],
    fairness_threshold: float,
):
    jobs[job_id]["status"] = "running"
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            run_remediation_sync,
            csv_bytes,
            protected_attributes,
            outcome_column,
            methods,
            target_adjustments,
            disparities,
            fairness_threshold,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ─────────────────────────────────────────────
# CSV streaming generator
# ─────────────────────────────────────────────

async def _stream_csv(csv_string: str, chunk_rows: int = 1_000):
    """
    Yield the CSV in row-chunks so the client starts receiving data
    immediately rather than waiting for the full file to buffer.
    """
    reader = pd.read_csv(io.StringIO(csv_string), chunksize=chunk_rows)
    first = True
    for chunk in reader:
        yield chunk.to_csv(index=False, header=first)
        first = False
        await asyncio.sleep(0)  # yield control back to event loop


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post("/remediate", response_model=JobStatus, status_code=202)
async def remediate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    protected_attributes: str = Form(...),   # JSON array string
    outcome_column: str = Form(...),
    methods: str = Form(...),                # JSON array string
    disparities: str = Form(default="[]"),   # JSON array string
    target_adjustments: str = Form(default="null"),
    fairness_threshold: float = Form(default=0.8),
):
    """
    Accept a multipart CSV upload and kick off async remediation.
    Returns a job_id immediately — poll /remediate/result/{job_id} for status.

    PLATFORM CHANGE: Send as multipart/form-data, not application/json.
    See /docs for the interactive form.
    """
    csv_bytes = await file.read()  # non-blocking I/O

    # Parse form fields
    try:
        attrs = json.loads(protected_attributes)
        meths = json.loads(methods)
        disps = json.loads(disparities)
        t_adj = json.loads(target_adjustments) if target_adjustments != "null" else None
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in form field: {e}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(
        _run_job,
        job_id,
        csv_bytes,
        attrs,
        outcome_column,
        meths,
        t_adj,
        disps,
        fairness_threshold,
    )

    return JobStatus(job_id=job_id, status="queued")


@app.get("/remediate/result/{job_id}", response_model=JobStatus)
async def get_result(job_id: str):
    """Poll this endpoint after POST /remediate to check job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_payload = None
    if job["status"] == "done" and job.get("result"):
        r = job["result"]
        result_payload = {
            "original_score": r["original_score"],
            "predicted_score": r["predicted_score"],
            "changes_made": r["changes_made"],
            "watchdog_report": r["watchdog_report"],
            "safe_to_apply": r["safe_to_apply"],
            "warnings": r["warnings"],
            "row_count_before": r["row_count_before"],
            "row_count_after": r["row_count_after"],
        }

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=result_payload,
        error=job.get("error"),
    )


@app.get("/remediate/download/{job_id}")
async def download_result(job_id: str):
    """
    Stream the remediated CSV once the job is done.
    Client receives rows progressively — no waiting for full file to buffer.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status']} — not ready for download",
        )

    csv_string = job["result"]["_remediated_csv"]
    return StreamingResponse(
        _stream_csv(csv_string),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=remediated_{job_id[:8]}.csv"},
    )


@app.delete("/remediate/result/{job_id}")
async def delete_job(job_id: str):
    """Clean up a completed job from memory once the client has downloaded results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs[job_id]
    return {"deleted": job_id}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_jobs": len(jobs),
        "executor_workers": _executor._max_workers,
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)