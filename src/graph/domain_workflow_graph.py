"""AgentCore Platform v1.0 - RET-C2-141 inner domain workflow graph.

Cat 2 inner graph: pose-metric scoring -> rubric query mapping -> rubric
retrieval -> feedback synthesis. Instantiated by
MotionTrainingGraphNode.get_subgraph() in graph.py.

Pipeline (linear, fail-fast on ERROR):
    START -> pose_metric_score -> rubric_query -> training_rubric_retrieve
          -> feedback_synthesis -> END
"""

from typing import Any
from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.nodes.feedback_synthesis_node import FeedbackSynthesisNode
from src.nodes.pose_metric_score_node import PoseMetricScoreNode
from src.nodes.rubric_query_node import RubricQueryNode
from src.nodes.training_rubric_retrieve_node import TrainingRubricRetrieveNode
from src.schemas.state import State


class MotionTrainingWorkflowGraph(BaseGraph):
    """Inner graph for the RET-C2-141 motion-training Q&A workflow."""

    @property
    def name(self) -> str:
        return "motion-training-workflow"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        # No mandatory config: llm and kb are optional (deterministic fallback exists).
        pass

    def register_nodes(self) -> None:
        # No super() - BaseGraph.register_nodes() is abstract.
        llm = self.config.get("llm")
        kb = self.config.get("kb")

        self._nodes["pose_metric_score"] = PoseMetricScoreNode()
        self._nodes["rubric_query"] = RubricQueryNode()
        self._nodes["training_rubric_retrieve"] = TrainingRubricRetrieveNode(kb=kb)
        self._nodes["feedback_synthesis"] = FeedbackSynthesisNode(llm=llm)

    def add_edges(self) -> None:
        self._sg.add_edge(START, "pose_metric_score")
        self._sg.add_conditional_edges(
            "pose_metric_score",
            lambda s: END if self._is_error(s) else "rubric_query",
            {"rubric_query": "rubric_query", END: END},
        )
        self._sg.add_conditional_edges(
            "rubric_query",
            lambda s: END if self._is_error(s) else "training_rubric_retrieve",
            {"training_rubric_retrieve": "training_rubric_retrieve", END: END},
        )
        self._sg.add_conditional_edges(
            "training_rubric_retrieve",
            lambda s: END if self._is_error(s) else "feedback_synthesis",
            {"feedback_synthesis": "feedback_synthesis", END: END},
        )
        self._sg.add_edge("feedback_synthesis", END)

    @staticmethod
    def _is_error(state: AgentState) -> bool:
        return state.get("status") in (AgentStatus.ERROR.value, AgentStatus.ERROR.value)

    def route(self, state: AgentState) -> str:
        return END if self._is_error(state) else "feedback_synthesis"

    def get_output(self, state: AgentState) -> dict[str, Any]:
        return {
            "pose_metrics": state.get("pose_metrics"),
            "capture_quality": state.get("capture_quality"),
            "rubric_query": state.get("rubric_query"),
            "matched_rubric": state.get("matched_rubric"),
            "remediation_draft": state.get("remediation_draft"),
            "final_feedback": state.get("final_feedback"),
            "output": state.get("final_feedback"),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
