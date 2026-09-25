"""AgentCore Platform v1.0 - RET-C2-141 PoseMetricScoreNode (inner subgraph, step 1).

Calls the deterministic PoseMetricScorer Tool logic (src/services/service.py) -
geometry-based scoring of bow angle / greeting-stance offset from keypoint
coordinates. Applies min_confidence per frame; fail-fast returns
capture_quality="fail" ("insufficient capture quality") rather than silently
scoring low-confidence data, per the locked spec. Note: since AgentBaseGraph's
pre_process -> main edge is unconditional, an upstream pre_process ERROR still
dispatches into this inner subgraph. The biometric-PII scan and action_type
validity are therefore re-checked here.
"""

import json
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import VALID_ACTION_TYPES, check_capture_quality, detect_biometric_pii, score_pose_metrics


class PoseMetricScoreNode(FunctionNode):
    """Score deterministic pose metrics with min_confidence fail-fast."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        parsed = from_json(state.get("user_input"), None)
        if not isinstance(parsed, dict):
            return {"status": AgentStatus.ERROR.value, "error_log": ["PoseMetricScoreNode: missing validated session"]}

        pii_hits = detect_biometric_pii(json.dumps(parsed, ensure_ascii=False))
        if pii_hits:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PoseMetricScoreNode: biometric-PII detected - rejected, not processed"],
            }

        session = parsed.get("session", {})
        action_type = parsed.get("action_type", "")
        if action_type not in VALID_ACTION_TYPES:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PoseMetricScoreNode: invalid action_type '{action_type}'"],
            }
        frames = session.get("frames", [])

        capture_quality = check_capture_quality(frames)
        if capture_quality == "fail":
            emit_trace_event("motion_capture_quality_insufficient", {"action_type": action_type}, state)
            return {
                "capture_quality": "fail",
                "status": AgentStatus.ERROR.value,
                "error_log": ["PoseMetricScoreNode: insufficient capture quality - too many low-confidence frames"],
            }

        pose_metrics = score_pose_metrics(frames, action_type)

        emit_trace_event(
            "pose_metrics_scored",
            {
                "action_type": action_type,
                "metric_count": len(pose_metrics),
                "correlation_id": state.get("correlation_id", ""),
            },
            state,
        )

        return {
            "pose_metrics": to_json(pose_metrics),
            "capture_quality": "pass",
            "status": AgentStatus.SUCCESS.value,
        }
