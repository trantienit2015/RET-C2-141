"""AgentCore Platform v1.0 - RET-C2-141 outer graph (Cat 2).

Cat 2: outer AgentBaseGraph with the fixed 5-node backbone. Domain complexity
is encapsulated in MotionTrainingGraphNode (the `main` slot), which wraps
the inner MotionTrainingWorkflowGraph. Do NOT override add_edges().

Backbone: initialize -> pre_process(InputValidate) -> main(GraphNode)
          -> post_process(OutputValidate) -> finalize

MotionTrainingGraphNode lives here (not under src/nodes/) - the PB-6
invoke-order test only discovers BaseNode subclasses under src/nodes/, and a
GraphNode's __call__ intentionally skips the standard S-2/S-4/S-3 lifecycle
(gating is delegated to the inner subgraph).
"""

from typing import Any, ClassVar, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes.post_process_node import OutputValidateNode
from src.nodes.pre_process_node import InputValidateNode
from src.schemas.state import State


class MotionTrainingGraphNode(GraphNode):
    """Wraps the inner pose-scoring + rubric-Q&A workflow (Cat 2 composition)."""

    # S-1: outer main-slot wrapper - receives caller-supplied input directly
    # (matches agent.yaml required_trust_level + sibling outer nodes pre/post_process).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL
    # "propagate": re-raise inner errors as SubgraphError (fail fast - default).
    error_strategy: ClassVar[str] = "propagate"
    # No HITL in this template.
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, llm: Any = None, kb: Any = None) -> None:
        super().__init__()
        self._llm = llm
        self._kb = kb or []

    def get_subgraph(self) -> Any:
        from src.graph.domain_workflow_graph import MotionTrainingWorkflowGraph

        sg = MotionTrainingWorkflowGraph(config=self._parent_config())
        sg.compile()
        return sg

    def extract_input(self, state: AgentState) -> str:
        emit_trace_event(
            "motion_training_workflow_dispatched", {"correlation_id": state.get("correlation_id", "")}, state
        )
        return cast(str, state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        emit_trace_event(
            "motion_training_workflow_completed",
            {"correlation_id": state.get("correlation_id", ""), "status": str(sub_result.get("status"))},
            state,
        )
        return {
            "pose_metrics": sub_result.get("pose_metrics"),
            "capture_quality": sub_result.get("capture_quality"),
            "rubric_query": sub_result.get("rubric_query"),
            "matched_rubric": sub_result.get("matched_rubric"),
            "remediation_draft": sub_result.get("remediation_draft"),
            "final_feedback": sub_result.get("final_feedback"),
            "result": sub_result.get("output"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        return {"llm": self._llm, "kb": self._kb}


class RetailStaffMotionTrainingQAAgent(AgentBaseGraph):
    """RET-C2-141 - Retail Staff Motion Capture Performance & Training Q&A Agent (Cat 2)."""

    @property
    def name(self) -> str:
        return "ret-c2-141"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize + finalize

        llm = self.config.get("llm")
        kb = self.config.get("kb")

        self._nodes["pre_process"] = InputValidateNode()
        self._nodes["main"] = MotionTrainingGraphNode(llm=llm, kb=kb)
        self._nodes["post_process"] = OutputValidateNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.


# Alias for agent.yaml module:"src.graph" resolution (AgentRegistry / api/server.py).
Graph = RetailStaffMotionTrainingQAAgent
