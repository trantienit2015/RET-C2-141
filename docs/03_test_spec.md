# RET-C2-141 — Test Specification

## Test Strategy

Deterministic-core: `llm` is optional (feedback synthesis enrichment only); the
full suite runs offline.

## Unit Tests (`tests/unit/test_nodes.py`)

| Node | Cases |
|---|---|
| InputValidateNode | success (schema valid, action_type valid); invalid JSON→ERROR; invalid action_type→ERROR; staff-identity pattern→ERROR; mynumber pattern→ERROR; malformed session (missing frames/keypoints)→ERROR |
| PoseMetricScoreNode | metrics scored above confidence floor; capture_quality=fail when >20% low-confidence frames; re-checked biometric-PII→ERROR |
| RubricQueryNode | query mapped from below-standard metric; missing pose_metrics→ERROR |
| TrainingRubricRetrieveNode | passages retrieved from KB; missing query→ERROR |
| FeedbackSynthesisNode | deterministic fallback remediation text assembled with capture_quality carried through |
| OutputValidateNode | success; upstream error short-circuit; non-suppressible hook blocks staff-identity/mynumber leak in the record |

## Integration Tests (`tests/integration/test_graph.py`)

| ID | Test | Expected |
|---|---|---|
| I-1 | greeting session, bow angle below standard, KB match | SUCCESS; ≥5 nodes; capture_quality=pass; remediation present |
| I-2 | staff-identity pattern in session payload | error (rejected, not processed) |
| I-3 | >20% low-confidence frames | error (insufficient capture quality) |
| I-4 | invalid action_type | error |

## Proof-of-Boundary Tests (`tests/proof_of_boundary/`)

| PB-ID | Boundary | Test | Expected Result |
|-------|----------|------|----------------|
| PB-1 | BaseNode → EventEmitter | `emit_trace_event()` fires on every invocation path | No silent failures |
| PB-2 | State serialization | Post-invoke State is primitives only | No Pydantic/dataclass |
| PB-4 | Import isolation | No Level 0 imports | AST scan: 0 violations |
| PB-5 | Checkpoint safety | No JWT/Pydantic in checkpoint | Inspection pass |
| PB-6 | Invoke execution order | S-1 → node_start → S-2 → execute → S-3 → node_complete | Order verified |

## Business Logic Tests

| TC-ID | Test | Input | Expected Result |
|-------|------|-------|----------------|
| BL-01 | bow angle below standard | greeting session, avg bow angle 10° (standard 15°) | rubric_query flags below-standard, gap=-5.0 |
| BL-02 | capture_quality fail-fast | 30% of frames confidence < 0.6 | capture_quality=fail, ERROR, no score emitted |
| BL-03 | staff-identity input reject | session payload containing "社員ID:12345" | ERROR, rejected before scoring |
| BL-04 | mynumber leak blocked at output | assembled record containing a 12-digit mynumber-shaped token | output gate blocks, ERROR |

## Non-suppressible biometric-PII tests (critical — PB-test contract)

- Staff-identity / 社員ID / 個人番号 / credential patterns in the input session payload
  are always rejected before any scoring occurs (unit + I-2 + BL-03).
- The post_process S-3 hook scans the assembled feedback record's own serialized
  text for the same patterns (it checks the record's own fields,
  never cross-referencing separate state fields) — verified the serialized output
  never contains a token matching the staff-identity/個人番号 regex set (BL-04).
- Audit events (S-4) never contain raw keypoints or staff identity — action_type +
  frame_count only.
- Low-confidence capture always fails fast; never silently produces a score from
  insufficient-quality data (BL-02).

## Test Execution Summary
- Execution date: 2026-07-07
- Total tests: see CI run
- Coverage: node-level unit + full-graph integration + PB-2/PB-4/PB-6
