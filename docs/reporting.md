# Quality Control Research Report v1

The report projects the completed immutable `BatchReport` into scalar-only frozen
`ResearchReport`, `ModelInfo`, `Counts` and `ReportRow` objects. It does not import
Streamlit, load checkpoints, decode images or run inference. Model names, frozen
thresholds and artifact identities come from existing installed approved metadata.
This is an export contract, not a general JSON deserializer or a model registry.

## Workflow and lifecycle

Open Batch inspection, select a model, upload permitted JPEG/PNG images, confirm
permitted use and explicitly run. The report follows the processing summary.
Review issues, timings, provenance, scores and limitations before selecting
**Download research CSV** or **Download research JSON**. Both use in-memory bytes
cached only with that session's completed snapshot. Details, download clicks and
ordinary reruns do not infer again. Changing category, model, files, configured
limit, consent or clearing invalidates the relevant report state. Navigation
between Batch Inspection and Reports preserves the immutable snapshot and its
exports; leaving those two pages clears it. Returning from Reports exposes a
read-only retained batch until Clear batch starts a new selection.
Validation failure removes exports and requires clearing/retrying.

The existing maximum 20 images, 32 MiB accepted total, 8 MiB per image and decoded
image constraints remain. Streamlit receives uploads before aggregate validation;
this is not a hard HTTP transport memory limit. This local app has no authentication
and is for one trusted owner. Downloads are explicit browser saves, with no server
output files. Clearing releases references; it does not guarantee secure erasure
or remove copies already downloaded or retained by the browser.

## JSON schema

UTF-8, fixed insertion order, two-space indentation and final LF. No timestamp is
included. `schema_version` is integer `1`; unsupported versions are rejected.
Changes to field types or meanings require a new schema version and compatibility
tests. `report_id` is `fvr1-` plus SHA-256 of the exact canonical v1 JSON document
without the `report_id` field. `batch_id` is the existing batch execution identity.
The same frozen snapshot produces identical bytes. Timings distinguish separate
executions, even when their inputs and batch identities match. Neither identifier
is a cryptographic signature, authenticity proof or globally unique event ID.

| Field | Type and meaning |
|---|---|
| report_id, producer, batch_id, category | Strings; producer is FactoryVision AI, category is cashew or metalparts |
| schema_version | Integer 1 |
| model | model_id/category/name/score_label/scientific_status strings; registered_artifact_sha256 and catalog_sha256 lowercase 64-character hex; artifact_version `sha256:<registered digest>`; frozen_threshold number or null; observed_artifact_sha256 registered digest only when at least one successful row exists, otherwise null |
| execution_status | completed, completed_with_issues or stopped |
| summary | Integer selected, accepted, rejected, inspected, failed and skipped counts |
| timing | batch_duration_ms number; inference_sum_ms, inference_median_ms, inference_min_ms, inference_max_ms number or null when no success; scope string |
| rows | Ordered array of 1–20 scalar row objects described below |
| limitations | Array of fixed scientific limitation strings |

Each row has image_id (`Image 001` onwards in submitted order), status (inspected,
rejected, failed, skipped), score, threshold, research_anomaly_flag, inference_ms,
adapter_total_ms, artifact_sha256, error_code, interpretation and error_message.
Scores/timings are finite nonnegative numbers when inspected, otherwise null.
Threshold/flag are the existing frozen Cashew number and strict boolean
`score > threshold`; both are null for MetalParts and unsuccessful rows.
artifact_sha256 is the verified successful artifact digest or null. Success errors
are null. Other errors are fixed categories/messages: input_rejected,
inference_failed and execution_unavailable. Arbitrary original exception text is
discarded, including fatal exception text.

Accepted means upload validation passed, not product acceptance. Accepted equals
inspected + failed + skipped; selected equals accepted + rejected. A stopped batch
may contain successes before interruption. No success may follow a skipped image.
All failed/rejected/skipped predictions and durations stay null, never zero.
`inference_sum_ms` refers only to successful images; it is not total batch time.
Batch duration includes verification, decoding, inference and display preparation,
but excludes upload, initial preview, report/export creation and browser rendering.

Validation checks versions, exact supported metadata, hashes, row ordering,
1–20 bounds, count/status consistency, finite numeric types (booleans rejected as
numbers), nonnegative timings, adapter duration >= inference duration, batch duration
>= sum of successful adapter durations, frozen threshold and decision semantics,
artifact identity and nullable failure states. Export validates again. Python JSON
uses `allow_nan=False`; no arbitrary object serialization is used.

## CSV schema and spreadsheet security

UTF-8 without BOM; comma delimiter, every field quoted, CRLF records. One record
per submitted image, including rejected/failed/skipped images. Stable column order:

```text
schema_version,report_id,batch_id,category,model_id,model_name,artifact_version,registered_artifact_sha256,catalog_sha256,scientific_status,execution_status,selected,accepted,rejected,inspected,failed,skipped,batch_duration_ms,image_id,status,score_label,score,threshold,research_anomaly_flag,interpretation,inference_ms,adapter_total_ms,observed_artifact_sha256,error_code,error_message,limitations,timing_scope
```

Common provenance/counts/timing/limitations repeat on every row. Missing values
are literal `null`, booleans `true`/`false`. Numeric `repr` preserves round-trip
precision rather than rounding to UI precision. CSV observed_artifact_sha256 is
per-row. Full JSON additionally includes the aggregate inference statistics and
the registered frozen threshold even when every row failed.

No user filenames or free-form user/error text are exported. All strings derive
from approved metadata, fixed text, checked hashes and positional IDs. As defense
in depth, every data cell passes a guard: text whose NFKC-normalized, whitespace-
stripped prefix is `=`, `+`, `-` or `@`, or which contains any Unicode control/format
character, gets a literal `[text] ` prefix. This includes tabs, line breaks, NUL,
BOM and fullwidth formula characters. Quotes, commas and Unicode are escaped by
the standard CSV writer; risky text is deliberately visibly marked. No executable
formula is generated. This does not claim protection after arbitrary downstream
rewriting or removal of the prefix. Quoting alone is insufficient, as explained
by [OWASP CSV Injection](https://community.owasp.org/attacks/CSV_Injection).

## Privacy and scientific interpretation

Exports contain no original filenames, paths, input fingerprints, pixels, maps,
EXIF, model-weight contents, tracebacks or credentials. They contain opaque IDs,
model artifact/catalog digests and scalar research results. The UI uses ordinary
Streamlit text/table rendering, never unsafe HTML for report content. No new
network calls, telemetry, background jobs, services or persistence are introduced.
Installed metadata is reread only for validation; no dataset payload is opened.
Bytes are prepared eagerly and download buttons use `on_click="ignore"`, supported
by [Streamlit's download API](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button).
No callable deferred generation or thread-based export job is used.

Cashew remains historical research on an observed official test set. Frozen
threshold flags are not certified industrial PASS/FAIL. Feature-memory and
student-teacher independent defect accuracy is not established. MetalParts remains
exploratory source-domain diagnostics with continuous scores only; target-domain
independent evaluation is NOT CLEARED. PCB1 remains BLOCKED and PCB2 INCONCLUSIVE.
Scores are not probabilities and unrelated scales are not combined. Low scores
do not establish defect absence. No PDF export or later milestone is included.
