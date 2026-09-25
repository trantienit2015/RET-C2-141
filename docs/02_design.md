# Template Design Specification

## Position in AgentCore Architecture

- **Agent Class**: `RetailStaffMotionTrainingQAAgent`
- **L1 Base**: AgentBaseGraph
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: L1 inheritance (Template Method: `execute(self, state: dict) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution)

## Architecture Overview

Cat 2: outer `AgentBaseGraph` (fixed 5-node backbone) + `GraphNode` in the `main`
slot wrapping an inner `BaseGraph` (4-step linear domain workflow), per the
node mapping below.

### Node Configuration (outer)

| Node | Responsibility | Input State | Output State | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------------|
| initialize | schema_version/session_id/trust_level | - | - | InitializeNode (default) |
| pre_process | `InputValidateNode` — S-1 trust gate + S-2 biometric-PII reject + freemocap schema validate | `user_input` | `action_type`, `validated_input` | FunctionNode |
| main | `MotionTrainingGraphNode` — wraps inner workflow | `validated_input` | `pose_metrics`, `capture_quality`, `rubric_query`, `matched_rubric`, `remediation_draft`, `final_feedback` | GraphNode |
| post_process | `OutputValidateNode` — S-3 non-suppressible biometric-PII leak re-check | `final_feedback` | `formatted_output`, `result` | FunctionNode |
| finalize | response_metadata, total_time_ms | - | - | FinalizeNode (default) |

### Node Configuration (inner)

| Node | Type (Agent-vs-Tool boundary) | Responsibility |
|------|------|-----------------|
| `PoseMetricScoreNode` | calls `PoseMetricScorer` Tool logic (deterministic geometry, in `src/services/service.py`) | Bow angle / greeting-stance scoring from keypoints; `min_confidence` fail-fast; re-checks biometric-PII (the pre_process→main edge is unconditional) |
| `RubricQueryNode` | deterministic mapping table, no LLM | Maps pose-metric scores to a rubric retrieval query |
| `TrainingRubricRetrieveNode` | calls `TrainingRubricRAG` Tool logic (dense vector retrieval, in `src/services/service.py`) | Retrieves matching rubric passages from the service-standard KB |
| `FeedbackSynthesisNode` | LLM node | Synthesizes language-neutral training feedback citing the rubric standard ID; temperature 0.0; deterministic fallback only when no LLM is configured — a configured LLM that fails or returns an empty response yields `status=error` |

> **Repo-layout note:** the design's "Tool" designation (`PoseMetricScorer`,
> `TrainingRubricRAG`) refers to the Agent-vs-Tool conceptual boundary (pure/
> deterministic function vs stateful orchestrating Agent) — both are implemented
> as pure functions in `src/services/service.py`, called by the corresponding
> LangGraph nodes, consistent with every other template's service-layer
> convention (no per-template `shared/tools/` package exists in this repo layout).

### Data Flow

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (retry)
                                          pre_process

Inner (main slot): START → pose_metric_score → rubric_query
                         → training_rubric_retrieve → feedback_synthesis → END
```

### State Definition

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| `action_type` | str | "greeting"/"hand_off"/"register_stance" | NotRequired |
| `capture_quality` | str | "pass"/"fail" | NotRequired |
| `pose_metrics` | str\|None (JSON) | dict{metric_name: {score, confidence, standard_id, gap}} | NotRequired |
| `rubric_query` | str | mapped rubric retrieval query | NotRequired |
| `matched_rubric` | str\|None (JSON) | list[{standard_id, passage}] | NotRequired |
| `remediation_draft` | str | draft synthesized feedback text | NotRequired |
| `final_feedback` | str\|None (JSON) | final structured training-feedback record | NotRequired |

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types)
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)
- No staff identity, 個人番号, or raw keypoint coordinate dump ever stored verbatim

## Freemocap Input Contract

- Schema version pinned: **v0.3.x** — `{session_id, frames: [{frame_number, pose_keypoints, confidence}]}`.
- `MIN_CONFIDENCE = 0.6` per frame.
- Fail-fast: if more than 20% of frames are below `MIN_CONFIDENCE`, return
  `capture_quality="fail"` ("insufficient capture quality") — never silently
  score low-confidence data.
- Keypoint→retail-posture-metric mapping: bow angle (nose–shoulder–hip angle,
  standard 15°), greeting-stance offset. Locked here; extend with additional
  metrics (hand-off, register-stance) as the rubric KB grows.

## Biometric-PII Handling

- **S-2 (input, `InputValidateNode` + re-checked in `PoseMetricScoreNode`):**
  deterministic regex reject of staff name / 社員ID / 個人番号 / credential
  patterns in the session payload. Keypoint arrays are pseudonymised at ingest
  (session_id only, no name) — deterministic, not an LLM check.
- **S-3 (output, `OutputValidateNode._extra_security_gate_output`):**
  deterministic regex block of the same patterns leaking into the assembled
  feedback record. On catch, the output is dropped entirely (`status=ERROR`,
  audit logged) — never suppressed.
- **PB-test contract:** the serialized final feedback record contains no token
  matching the staff-identity / 個人番号 / credential regex set.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, permissions, credential handle)
- [x] S-1: `required_trust_level: VERIFIED_EXTERNAL` on all nodes handling motion
      data (biometric) — refuses below-trust callers
- [x] S-4: `emit_trace_event()` — domain events at every node (session ingested,
      capture-quality checked, pose metrics scored, rubric query mapped, rubric
      retrieved, feedback synthesized/validated); audit metadata is action_type +
      frame_count only, never raw keypoints or staff identity

> **S-2/S-3 gate behaviour by node type (ADR-017):**
> - `FunctionNode` subclass → framework `@final` gate always runs automatically;
>   extend via `_extra_security_gate_input()` / `_extra_security_gate_output()` only
> - `GraphNode` / `RemoteAgentNode` → deliberate no-op (upstream or remote node's gate already applied)
> - Custom `BaseNode` subclass → must implement `_security_gate_input()` and
>   `_security_gate_output()` directly (`@abstractmethod` — omission raises `TypeError` at instantiation)

### Composition Pattern

- **Pattern**: GraphNode (subgraph)
- **Composition target**: `MotionTrainingWorkflowGraph` (inner `BaseGraph`)
- **Error propagation strategy**: propagate (fail fast; no HITL in this template)

## Import Isolation Confirmation
- [x] Template does not import agenticstar-platform SDK (Level 0)
- [x] Import targets: framework/ and shared/ only (no agents/base/ required)

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed multi-step scoring+retrieval workflow, no self-directed loop needed |
| Composition pattern | Standalone nodes | GraphNode (subgraph) | GraphNode | Cat 2 domain complexity (4-step inner workflow) encapsulated per scaffold convention |
| Level 2 base type | RAGAgent | ToolCallingAgent | ToolCallingAgent | Core capability is deterministic pose-metric scoring orchestrated with rubric retrieval, not primarily vector search |
| Low-confidence handling | Silent low-confidence score | Fail-fast reject | Fail-fast | Training feedback based on unreliable capture data would mislead staff and trainers |
