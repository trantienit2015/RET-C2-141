"""AgentCore Platform v1.0 - RET-C2-141 TrainingRubricRetrieveNode (inner subgraph, step 3).

Calls the deterministic TrainingRubricRAG Tool logic (src/services/service.py) -
dense vector retrieval mock over the service-standard rubric KB, scoped by
action_type metadata filter. KB-backed only.
"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import retrieve_rubric_passages


class TrainingRubricRetrieveNode(FunctionNode):
    """Retrieve matching rubric passages from the service-standard KB."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, kb: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._kb = kb or []

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("rubric_query", "")
        if not query:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["TrainingRubricRetrieveNode: missing rubric query"],
            }

        matched_rubric = retrieve_rubric_passages(query, self._kb)

        emit_trace_event(
            "training_rubric_retrieved",
            {"match_count": len(matched_rubric), "correlation_id": state.get("correlation_id", "")},
            state,
        )

        return {
            "matched_rubric": to_json(matched_rubric),
            "status": AgentStatus.SUCCESS.value,
        }
