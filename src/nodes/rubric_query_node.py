"""AgentCore Platform v1.0 - RET-C2-141 RubricQueryNode (inner subgraph, step 2).

Deterministic score-to-rubric-query mapping table (no LLM), locked in
docs/02_design.md.
"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.service import map_score_to_rubric_query


class RubricQueryNode(FunctionNode):
    """Map the pose-metric scores to a rubric retrieval query."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        pose_metrics = from_json(state.get("pose_metrics"), None)
        parsed = from_json(state.get("user_input"), {})
        action_type = parsed.get("action_type", "") if isinstance(parsed, dict) else ""

        if not isinstance(pose_metrics, dict) or not pose_metrics:
            return {"status": AgentStatus.ERROR.value, "error_log": ["RubricQueryNode: missing pose metrics"]}

        query = map_score_to_rubric_query(pose_metrics, action_type)

        emit_trace_event(
            "rubric_query_mapped",
            {"action_type": action_type, "correlation_id": state.get("correlation_id", "")},
            state,
        )

        return {
            "rubric_query": query,
            "status": AgentStatus.SUCCESS.value,
        }
