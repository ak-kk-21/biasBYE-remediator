# BiasBYE Remediator API

**Version:** 2.0  
**Base URL:** `http://localhost:8000`  
**Interactive docs:** `http://localhost:8000/docs`  
**Part of:** [Invisible at the Intersection — Unified Intersectional Fairness Framework]

---

## Table of Contents

1. [Overview](#1-overview)
2. [How It Works — The Async Job Pattern](#2-how-it-works--the-async-job-pattern)
3. [Installation & Running](#3-installation--running)
4. [Authentication](#4-authentication)
5. [Endpoints](#5-endpoints)
   - [POST /remediate](#51-post-remediate)
   - [GET /remediate/result/{job_id}](#52-get-remediateresultjob_id)
   - [GET /remediate/download/{job_id}](#53-get-remediatedownloadjob_id)
   - [DELETE /remediate/result/{job_id}](#54-delete-remediateresultjob_id)
   - [GET /health](#55-get-health)
6. [Remediation Methods](#6-remediation-methods)
   - [smote](#61-smote)
   - [upsample](#62-upsample)
   - [downsample](#63-downsample)
   - [target_adjustment](#64-target_adjustment)
7. [Watchdog Report](#7-watchdog-report)
8. [Error Reference](#8-error-reference)
9. [Complete Worked Examples](#9-complete-worked-examples)
   - [cURL — COMPAS dataset](#91-curl--compas-dataset)
   - [Python](#92-python)
   - [JavaScript / Fetch](#93-javascript--fetch)
10. [Migration from v1](#10-migration-from-v1)
11. [Performance Notes](#11-performance-notes)
12. [Production Considerations](#12-production-considerations)

---

## 1. Overview

The BiasBYE Remediator API accepts biased datasets and applies intersectional bias mitigation techniques. It is one of seven pipeline stages in the **Invisible at the Intersection** fairness framework, sitting between the Subgroup Discovery engine and the Causal DAG analysis layer.

**What it does:**
- Accepts any CSV dataset via multipart file upload
- Applies one or more bias mitigation methods: SMOTE, oversampling, undersampling, or direct target adjustment
- Returns the full remediated dataset with all original columns preserved
- Provides a Watchdog Report validating that changes improve — not harm — vulnerable groups
- Streams the output CSV so large files download progressively

**What it does not do:**
- It does not train or evaluate ML models
- It does not perform subgroup discovery (that is the upstream stage)
- It does not persist data between server restarts (in-memory job store)

---

## 2. How It Works — The Async Job Pattern

Version 2.0 uses a **submit → poll → download → cleanup** pattern. This keeps the API responsive for large files.

```
Client                          Server
  |                               |
  |  POST /remediate (file)       |
  |------------------------------>|
  |                               |  Queues background job
  |  202 { job_id }               |
  |<------------------------------|
  |                               |
  |  GET /result/{job_id}  ×N    |  Job runs in ProcessPoolExecutor
  |------------------------------>|  (chunked SMOTE, encoding, sampling)
  |  { status: "running" }        |
  |<------------------------------|
  |                               |
  |  GET /result/{job_id}         |
  |------------------------------>|
  |  { status: "done", result }   |
  |<------------------------------|
  |                               |
  |  GET /download/{job_id}       |
  |------------------------------>|
  |  Streaming CSV (1k rows/chunk)|
  |<------------------------------|
  |                               |
  |  DELETE /result/{job_id}      |
  |------------------------------>|
  |  { deleted: job_id }          |
  |<------------------------------|
```

**Why async?** SMOTE and sampling are CPU-bound. Running them synchronously blocks FastAPI's event loop, preventing any other requests from being served. By offloading to a `ProcessPoolExecutor`, the server stays responsive to concurrent requests while remediation runs.

---

## 3. Installation & Running

### Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn imbalanced-learn
```

### Start the server

```bash
python api.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ **Use `--workers 1`** when running with the in-process job store. Multiple workers will not share the `jobs` dict, causing poll requests to return 404. For multi-worker deployments, replace the `jobs` dict with Redis. See [Production Considerations](#12-production-considerations).

### Environment variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `PORT` | integer | `8000` | Port to bind the server to |

---

## 4. Authentication

Version 2.0 does not enforce authentication. CORS is open (`allow_origins: ["*"]`) for development purposes.

> ⚠️ Before exposing this service externally, restrict CORS origins and add JWT or API key authentication via FastAPI's `SecurityScopes`.

---

## 5. Endpoints

### 5.1 `POST /remediate`

Submit a dataset for bias remediation. Returns a `job_id` immediately with HTTP 202. Computation runs in the background.

**Request format:** `multipart/form-data`

> Do **not** set `Content-Type` manually. Let the HTTP client set it automatically with the correct multipart boundary.

#### Form fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | ✅ Yes | — | CSV file upload. Must have a header row. Column names are lowercased and stripped automatically. |
| `protected_attributes` | string (JSON array) | ✅ Yes | — | JSON-encoded list of protected attribute column names. Example: `["race","sex","age"]` |
| `outcome_column` | string | ✅ Yes | — | Name of the binary outcome column to balance. Must exist in the CSV. |
| `methods` | string (JSON array) | ✅ Yes | — | Ordered list of remediation methods to apply. Options: `smote`, `upsample`, `downsample`, `target_adjustment`. Applied in sequence. |
| `disparities` | string (JSON array) | No | `[]` | JSON-encoded list of disparity objects from the subgroup discovery stage. Each object may contain `is_significant: true\|false`. Used to compute the baseline fairness score. |
| `target_adjustments` | string (JSON object) | No | `null` | Map of subgroup name → new outcome value. Only used when `target_adjustment` is in `methods`. See [Section 6.4](#64-target_adjustment). |
| `fairness_threshold` | float | No | `0.8` | Minimum acceptable fairness score (0.0–1.0). Used in the Watchdog verdict. |

#### Response — `202 Accepted`

```json
{
  "job_id": "a3f2c1d4-88b2-4e10-9f3a-bc12d4567890",
  "status": "queued",
  "result": null,
  "error": null
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | Unique job identifier. Use in all subsequent requests. |
| `status` | string | Always `"queued"` at submission. |
| `result` | null | Always null at submission. |
| `error` | null | Always null at submission. |

#### Status codes

| Code | Meaning |
|---|---|
| `202` | Job queued. Poll `/remediate/result/{job_id}` for progress. |
| `422` | Invalid JSON in a form field, or a required field is missing. |
| `500` | Unexpected server error during job setup. |

---

### 5.2 `GET /remediate/result/{job_id}`

Poll for job status. When `status` is `"done"`, also returns the remediation metadata. The remediated CSV is **not** in this response — use the [download endpoint](#53-get-remediatedownloadjob_id).

#### Path parameters

| Parameter | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The `job_id` returned from `POST /remediate` |

#### Response — `200 OK`

```json
{
  "job_id": "a3f2c1d4-88b2-4e10-9f3a-bc12d4567890",
  "status": "done",
  "result": {
    "original_score": 42.0,
    "predicted_score": 52.0,
    "changes_made": [
      "Applied chunked SMOTE — balanced outcome distribution (5000-row chunks)"
    ],
    "watchdog_report": "BIASBYE REMEDIATION WATCHDOG REPORT\n...",
    "safe_to_apply": true,
    "warnings": [],
    "row_count_before": 7214,
    "row_count_after": 9108
  },
  "error": null
}
```

#### Result fields (when `status == "done"`)

| Field | Type | Description |
|---|---|---|
| `original_score` | float | Baseline fairness score (0–100) computed from the input `disparities` list before any remediation. |
| `predicted_score` | float | Estimated fairness score after remediation. Increases by ~10 per successful method applied, capped at 100. This is a heuristic — validate by re-running subgroup discovery on the remediated dataset. |
| `changes_made` | string[] | Human-readable list of changes applied. Empty if no methods succeeded. |
| `watchdog_report` | string | Full plain-text watchdog report. See [Section 7](#7-watchdog-report). |
| `safe_to_apply` | boolean | `true` if no warnings were raised, or if `predicted_score > original_score`. |
| `warnings` | string[] | Non-fatal issues encountered (e.g. a method that fell back or failed). |
| `row_count_before` | integer | Number of rows in the original uploaded dataset. |
| `row_count_after` | integer | Number of rows in the remediated dataset. Higher for oversampling methods, lower for undersampling. |

#### Status values

| Value | Meaning |
|---|---|
| `queued` | Job is waiting to enter the process pool. |
| `running` | Remediation is actively running in the process pool. |
| `done` | Remediation complete. Result is available. |
| `failed` | An unrecoverable error occurred. Check `error` field. |

#### Response when running

```json
{
  "job_id": "a3f2c1d4-88b2-4e10-9f3a-bc12d4567890",
  "status": "running",
  "result": null,
  "error": null
}
```

#### Response when failed

```json
{
  "job_id": "a3f2c1d4-88b2-4e10-9f3a-bc12d4567890",
  "status": "failed",
  "result": null,
  "error": "SMOTE failed: Not enough samples in minority class"
}
```

#### Status codes

| Code | Meaning |
|---|---|
| `200` | Status returned. Check the `status` field. |
| `404` | No job with this `job_id` exists. May have been deleted or never created. |

> **Polling recommendation:** Poll every 1,500ms for datasets under 10k rows. Increase to 3,000ms for larger datasets. Stop when `status` is `"done"` or `"failed"`.

---

### 5.3 `GET /remediate/download/{job_id}`

Stream the remediated CSV. The server yields the file in 1,000-row chunks so the client starts receiving data immediately. Only available when `status == "done"`.

The downloaded CSV contains **all original columns** from the uploaded dataset — not just protected attributes and the outcome column.

#### Path parameters

| Parameter | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The `job_id` of a completed job |

#### Response — `200 OK`

```
Content-Type: text/csv
Content-Disposition: attachment; filename=remediated_{first 8 chars of job_id}.csv
Transfer-Encoding: chunked
```

The body is a valid CSV streamed in 1,000-row increments. The header row is included once at the start.

#### Status codes

| Code | Meaning |
|---|---|
| `200` | Streaming CSV. |
| `404` | No job with this `job_id` exists. |
| `409` | Job exists but is not yet `done`. Wait and retry. |

---

### 5.4 `DELETE /remediate/result/{job_id}`

Remove a completed job and its CSV from server memory. Call this after the client has successfully downloaded the file. The server holds the full CSV in memory between job completion and deletion, so calling this promptly matters for large files.

#### Path parameters

| Parameter | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The `job_id` to delete |

#### Response — `200 OK`

```json
{ "deleted": "a3f2c1d4-88b2-4e10-9f3a-bc12d4567890" }
```

#### Status codes

| Code | Meaning |
|---|---|
| `200` | Job deleted. All data freed from memory. |
| `404` | No job with this `job_id` exists. |

---

### 5.5 `GET /health`

Service health check. Use for load balancer probes and monitoring.

#### Response — `200 OK`

```json
{
  "status": "healthy",
  "active_jobs": 3,
  "executor_workers": 4
}
```

| Field | Description |
|---|---|
| `status` | Always `"healthy"` if the server is running. |
| `active_jobs` | Number of jobs currently held in memory (queued + running + done but not yet deleted). |
| `executor_workers` | Number of processes in the `ProcessPoolExecutor`. Default is 4. |

---

## 6. Remediation Methods

Pass one or more method names in the `methods` form field as a JSON array. Methods are applied **in sequence** — each method operates on the output of the previous one.

```
methods=["smote", "target_adjustment"]
```

In this example, SMOTE runs first, then target adjustment runs on the SMOTE output.

---

### 6.1 `smote`

Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the outcome column. Uses chunked processing to avoid O(n²) KNN scaling on large datasets. Preserves all columns.

**How it works internally:**
1. Dataset is split into chunks of 5,000 rows
2. Each chunk is encoded (all categorical columns → integers)
3. SMOTE is applied per chunk with `k_neighbors = min(5, chunk_size - 1)`
4. If a chunk is too small for SMOTE (< 6 rows), `RandomOverSampler` is used as fallback
5. All columns are decoded back to original values
6. Chunks are concatenated with a single `pd.concat` call

| Property | Value |
|---|---|
| Effect on row count | Increases (synthetic rows added) |
| Chunk size | 5,000 rows |
| Fallback | `RandomOverSampler` for small chunks |
| Columns preserved | All original columns |
| Best for | Imbalanced outcome distributions where one class is significantly underrepresented |

> ⚠️ SMOTE generates **synthetic** rows by interpolating between existing samples. The output dataset will contain rows that did not exist in the original data.

---

### 6.2 `upsample`

Applies random oversampling (`RandomOverSampler`) by duplicating existing minority class rows. Simpler and faster than SMOTE — no synthetic data generation. Preserves all columns.

| Property | Value |
|---|---|
| Effect on row count | Increases (rows duplicated) |
| Synthetic data | No — duplicates real rows |
| Columns preserved | All original columns |
| Best for | Very small datasets where SMOTE's KNN cannot be reliably computed, or when synthetic interpolation is not appropriate |

---

### 6.3 `downsample`

Applies random undersampling (`RandomUnderSampler`) by removing majority class rows. Decreases total row count. Preserves all columns.

| Property | Value |
|---|---|
| Effect on row count | Decreases (majority rows removed) |
| Data loss | Yes — majority class rows are permanently removed |
| Columns preserved | All original columns |
| Best for | Very large datasets where oversampling would be computationally prohibitive |

> ⚠️ Downsampling loses data. Only use when the majority class is severely overrepresented and dataset size is a constraint.

---

### 6.4 `target_adjustment`

Directly flips the outcome value for specified intersectional subgroups. No rows are added or removed. Requires the `target_adjustments` form field.

#### Subgroup name format

Subgroup names are attribute=value pairs joined by ` + ` (space-plus-space):

```
"race=African-American + sex=Female"
```

Each condition is matched against the dataset using string comparison. Multiple conditions use AND logic — all conditions must be true for a row to be matched.

#### `target_adjustments` format

A JSON object mapping subgroup names to the new outcome integer value:

```json
{
  "race=African-American + sex=Female": 0,
  "race=Caucasian + sex=Male": 1
}
```

| Property | Value |
|---|---|
| Effect on row count | None |
| Data loss | No |
| Columns preserved | All original columns |
| Best for | Correcting known historically biased labels for specific intersectional subgroups |

> ⚠️ This method directly modifies ground truth labels. It should be used only when there is a domain-justified reason to believe the original labels are incorrect for a specific subgroup — not as a general balancing technique.

---

## 7. Watchdog Report

Every job produces a plain-text watchdog report returned in `result.watchdog_report` and used to set `result.safe_to_apply`.

### Report format

```
BIASBYE REMEDIATION WATCHDOG REPORT
=====================================
Original fairness score : 42.0/100
Predicted score after   : 52.0/100
Rows before             : 7214
Rows after              : 9108

Changes applied:
  - Applied chunked SMOTE — balanced outcome distribution (5000-row chunks)
  - Applied target adjustments for 2 subgroup(s)

Warnings:
  None

Verdict: SAFE TO APPLY

Guardian note: The watchdog has verified that no protected subgroup outcome
rate decreased below baseline.
```

### Fairness score calculation

The `original_score` is a proxy computed from the `disparities` input:

```
significant_count = count of disparities where is_significant == true
original_score    = max(0, 100 - (significant_count / total_count × 100))
```

The `predicted_score` adds 10 points per successful method applied, capped at 100:

```
predicted_score = min(100, original_score + len(changes_made) × 10)
```

> This is a **heuristic estimate**. For accurate post-remediation fairness measurement, re-run the subgroup discovery pipeline on the remediated dataset.

### `safe_to_apply` logic

```python
safe_to_apply = (len(warnings) == 0) or (predicted_score > original_score)
```

A remediation is considered safe even if some methods produced warnings, as long as the net predicted score improved. Always perform a human review when warnings are present.

### Verdict values

| Verdict | Meaning |
|---|---|
| `SAFE TO APPLY` | No warnings raised, or score improved despite warnings. |
| `REVIEW REQUIRED` | Warnings were raised and score did not improve. Human validation required before applying. |

---

## 8. Error Reference

All error responses follow FastAPI's default schema:

```json
{ "detail": "Description of what went wrong" }
```

For `422` errors, `detail` is an array:

```json
{
  "detail": [
    {
      "loc": ["body", "outcome_column"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### HTTP status codes

| Code | Endpoint(s) | Meaning |
|---|---|---|
| `202` | `POST /remediate` | Job queued successfully. |
| `200` | All others | Success. |
| `404` | Result, download, delete | No job with this `job_id`. May have been deleted or never created. |
| `409` | Download | Job exists but is not yet `done`. Cannot download. |
| `422` | `POST /remediate` | Invalid JSON in a form field, or a required field is missing. |
| `500` | Any | Unexpected server error. |

### Common issues

**I get a 404 when polling after submitting.**  
You are running multiple uvicorn workers. The `jobs` dict is not shared between workers. Use `--workers 1` or migrate the job store to Redis.

**The download returns only 3 columns instead of the full dataset.**  
This was a bug in v2.0 (fixed in the current version). The sampling functions were rebuilding the dataframe from only `protected_cols + outcome_col`. The fix encodes and resamples all columns, then decodes and restores the original column order.

**SMOTE failed: Not enough samples.**  
The minority class has fewer than 6 samples after dropping NaN rows in the protected attribute or outcome columns. The API automatically falls back to `RandomOverSampler`. If this appears in `warnings`, the fallback was used.

**422: Invalid JSON in form field.**  
The `protected_attributes` and `methods` fields must be valid JSON array strings. The `target_adjustments` field must be a valid JSON object string or the literal string `"null"`. Do not pass raw Python lists — always `JSON.stringify()` or `json.dumps()` them first.

---

## 9. Complete Worked Examples

### 9.1 cURL — COMPAS dataset

```bash
# Step 1 — submit
curl -X POST http://localhost:8000/remediate \
  -F "file=@compas-scores.csv;type=text/csv" \
  -F 'protected_attributes=["race","sex"]' \
  -F "outcome_column=two_year_recid" \
  -F 'methods=["smote","target_adjustment"]' \
  -F 'target_adjustments={"race=African-American + sex=Female": 0, "race=Caucasian + sex=Male": 1}' \
  -F "fairness_threshold=0.8"

# Response:
# { "job_id": "e30bd8ae-d635-4928-94a3-450da173d622", "status": "queued", ... }

# Step 2 — poll (run repeatedly until status == "done")
curl http://localhost:8000/remediate/result/e30bd8ae-d635-4928-94a3-450da173d622

# Step 3 — download the full remediated dataset
curl -O http://localhost:8000/remediate/download/e30bd8ae-d635-4928-94a3-450da173d622

# Step 4 — free server memory
curl -X DELETE http://localhost:8000/remediate/result/e30bd8ae-d635-4928-94a3-450da173d622
```

**Windows (curl.exe) note:** Escape inner double quotes with backslash:

```cmd
curl.exe -X POST http://localhost:8000/remediate ^
  -F "file=@compas-scores.csv;type=text/csv" ^
  -F "protected_attributes=[\"race\",\"sex\"]" ^
  -F "outcome_column=two_year_recid" ^
  -F "methods=[\"smote\",\"target_adjustment\"]" ^
  -F "target_adjustments={\"race=African-American + sex=Female\": 0}" ^
  -F "fairness_threshold=0.8"
```

---

### 9.2 Python

```python
import requests
import json
import time

BASE = "http://localhost:8000"

# ── Step 1: Submit
with open("compas-scores.csv", "rb") as f:
    resp = requests.post(
        f"{BASE}/remediate",
        data={
            "protected_attributes": json.dumps(["race", "sex"]),
            "outcome_column": "two_year_recid",
            "methods": json.dumps(["smote", "target_adjustment"]),
            "target_adjustments": json.dumps({
                "race=African-American + sex=Female": 0,
                "race=Caucasian + sex=Male": 1
            }),
            "disparities": json.dumps([
                {"is_significant": True},
                {"is_significant": True},
                {"is_significant": False},
            ]),
            "fairness_threshold": "0.8",
        },
        files={"file": ("compas.csv", f, "text/csv")}
    )

resp.raise_for_status()
job_id = resp.json()["job_id"]
print(f"Job submitted: {job_id}")

# ── Step 2: Poll
while True:
    time.sleep(1.5)
    status = requests.get(f"{BASE}/remediate/result/{job_id}").json()
    print(f"  Status: {status['status']}")

    if status["status"] == "done":
        r = status["result"]
        print(f"  Score: {r['original_score']} → {r['predicted_score']}")
        print(f"  Rows:  {r['row_count_before']} → {r['row_count_after']}")
        print(f"  Safe:  {r['safe_to_apply']}")
        print(f"\n{r['watchdog_report']}")
        break

    if status["status"] == "failed":
        raise RuntimeError(f"Job failed: {status['error']}")

# ── Step 3: Download (streaming)
with requests.get(f"{BASE}/remediate/download/{job_id}", stream=True) as r:
    r.raise_for_status()
    with open("remediated_compas.csv", "wb") as out:
        for chunk in r.iter_content(chunk_size=8192):
            out.write(chunk)

print("Downloaded: remediated_compas.csv")

# ── Step 4: Cleanup
requests.delete(f"{BASE}/remediate/result/{job_id}")
print("Cleaned up.")
```

---

### 9.3 JavaScript / Fetch

```javascript
const BASE = "http://localhost:8000";

async function remediateDataset(csvFile) {
  // ── Step 1: Submit
  const form = new FormData();
  form.append("file", csvFile);
  form.append("protected_attributes", JSON.stringify(["race", "sex"]));
  form.append("outcome_column", "two_year_recid");
  form.append("methods", JSON.stringify(["smote", "target_adjustment"]));
  form.append("target_adjustments", JSON.stringify({
    "race=African-American + sex=Female": 0,
    "race=Caucasian + sex=Male": 1
  }));
  form.append("fairness_threshold", "0.8");

  // Do NOT set Content-Type — browser sets it with boundary automatically
  const submitRes = await fetch(`${BASE}/remediate`, {
    method: "POST",
    body: form,
  });

  if (!submitRes.ok) throw new Error(`Submit failed: ${submitRes.status}`);
  const { job_id } = await submitRes.json();
  console.log(`Job submitted: ${job_id}`);

  // ── Step 2: Poll
  let result;
  while (true) {
    await new Promise(r => setTimeout(r, 1500));

    const pollRes = await fetch(`${BASE}/remediate/result/${job_id}`);
    const status = await pollRes.json();
    console.log(`  Status: ${status.status}`);

    if (status.status === "done") {
      result = status.result;
      console.log(`  Score: ${result.original_score} → ${result.predicted_score}`);
      console.log(`  Rows:  ${result.row_count_before} → ${result.row_count_after}`);
      break;
    }

    if (status.status === "failed") {
      throw new Error(`Job failed: ${status.error}`);
    }
  }

  // ── Step 3a: Simple file download (browser)
  window.location.href = `${BASE}/remediate/download/${job_id}`;

  // ── Step 3b: Stream into memory (e.g. to parse and preview in-app)
  // const stream = await fetch(`${BASE}/remediate/download/${job_id}`);
  // const reader = stream.body.getReader();
  // const decoder = new TextDecoder();
  // let csvText = "";
  // while (true) {
  //   const { done, value } = await reader.read();
  //   if (done) break;
  //   csvText += decoder.decode(value);
  //   // Call your table preview function here as rows arrive
  // }

  // ── Step 4: Cleanup
  await fetch(`${BASE}/remediate/result/${job_id}`, { method: "DELETE" });
  console.log("Cleaned up.");

  return result;
}
```

---

## 10. Migration from v1

v2.0 has three breaking changes from v1.

### Breaking change 1 — Request format

| | v1 | v2 |
|---|---|---|
| Content-Type | `application/json` | `multipart/form-data` |
| Dataset field | `dataset_csv` string in JSON body | `file` as File upload |
| Other fields | JSON body fields | Form fields (arrays/objects as JSON strings) |

### Breaking change 2 — Response is async

| | v1 | v2 |
|---|---|---|
| `POST /remediate` response | Full remediation result (blocking) | `job_id` + `202` (immediate) |
| `remediated_csv` in response | Yes (base64 string) | No — use download endpoint |
| Polling required | No | Yes |

### Breaking change 3 — Separate download endpoint

In v1, the remediated CSV was returned as a base64 string inside the JSON response body. In v2, it is streamed from a dedicated endpoint once the job is complete.

### Before / After

**v1:**
```javascript
const res = await fetch("/remediate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    dataset_csv: csvString,
    protected_attributes: ["race", "sex"],
    outcome_column: "two_year_recid",
    methods: ["smote"],
  }),
});
const { remediated_csv } = await res.json();
```

**v2:**
```javascript
const form = new FormData();
form.append("file", csvFile);
form.append("protected_attributes", JSON.stringify(["race", "sex"]));
form.append("outcome_column", "two_year_recid");
form.append("methods", JSON.stringify(["smote"]));

const { job_id } = await fetch("/remediate", { method: "POST", body: form })
  .then(r => r.json());

// poll, then download separately
```

---

## 11. Performance Notes

| Optimization | What it does |
|---|---|
| Chunked SMOTE (5k rows/chunk) | Keeps KNN graph O(chunk_size) instead of O(n²). Critical for datasets over 10k rows. |
| `ProcessPoolExecutor` (4 workers) | SMOTE and encoding run in separate processes. Event loop stays free for concurrent API requests. |
| `multipart/form-data` upload | Avoids JSON serialization overhead of embedding a large CSV string in the request body. |
| `StreamingResponse` | Client receives first 1,000 rows in ~1s instead of waiting for the full file to buffer. |
| LRU encoder cache (32 slots) | Repeated requests with the same dataset reuse pre-fit `LabelEncoder` objects. Keyed by MD5 hash. |

### Benchmark — COMPAS dataset (~7,200 rows)

| Version | Method | Time |
|---|---|---|
| v1 | Synchronous, full CSV in JSON body | ~18–25s |
| v2 | Async, multipart, chunked SMOTE | ~4–6s |

---

## 12. Production Considerations

### Replace the in-memory job store

The current `jobs: Dict[str, Dict] = {}` is in-process and does not survive restarts or scale across workers.

**For production:** Replace with Redis using `arq` or Celery:

```python
import redis
r = redis.Redis()

# Store job
r.setex(f"job:{job_id}", 3600, json.dumps(job_data))  # TTL: 1 hour

# Retrieve job
job = json.loads(r.get(f"job:{job_id}"))
```

### Restrict CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com"],  # not ["*"]
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Add authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

security = HTTPBearer()

@app.post("/remediate")
async def remediate(
    credentials: HTTPAuthorizationCredentials = Security(security),
    ...
):
    verify_token(credentials.credentials)  # your JWT/API key check
    ...
```

### Set file size limits

Add to uvicorn startup or nginx proxy:

```bash
uvicorn api:app --limit-concurrency 20 --limit-max-requests 1000
```

Or in nginx:
```nginx
client_max_body_size 500M;
```

### Job TTL and cleanup

The server currently holds job data in memory indefinitely until the client calls `DELETE`. Add a background cleanup task for jobs that are never explicitly deleted:

```python
import asyncio
from datetime import datetime, timedelta

async def cleanup_stale_jobs():
    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        cutoff = datetime.utcnow() - timedelta(hours=1)
        stale = [jid for jid, j in jobs.items() if j.get("created_at", datetime.utcnow()) < cutoff]
        for jid in stale:
            del jobs[jid]
```

---

*BiasBYE Remediator API v2.0 — Invisible at the Intersection Framework*  
