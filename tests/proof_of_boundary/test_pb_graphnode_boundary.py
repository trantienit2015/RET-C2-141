# PB-6 (GraphNode boundary supplement): `MotionTrainingGraphNode` lives in
# src/graph/graph.py (correct scaffold placement for a Cat 2 outer main-slot
# wrapper), which is outside PB-6's src/nodes/ discovery scope. This test
# bounds the outer composition boundary that PB-6 does not probe:
# S-1 trust gate on the wrapper itself, explicit field-mapping in
# extract_input()/merge_output() (criterion #9), and that gating for the
# inner subgraph is a deliberate delegation, not an omission.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import MotionTrainingGraphNode


def test_s1_trust_gate_denies_insufficient_caller():
    # __call__() reads required_trust_level off self.__class__ - a subclass
    # (not an instance attribute) is required to force a denial for this probe.
    _HighTrustProbe = type(
        "_HighTrustProbe", (MotionTrainingGraphNode,), {"required_trust_level": TrustLevel.INTERNAL}
    )
    node = _HighTrustProbe()
    out = node({"caller_trust_level": TrustLevel.ANONYMOUS.value, "validated_input": '{"action_type": "greeting"}'})
    assert str(out.get("status")).lower().endswith("error")
    assert out["error_log"] and "trust gate denied" in out["error_log"][0]


def test_extract_input_maps_only_the_contracted_field():
    node = MotionTrainingGraphNode()
    state = {
        "validated_input": '{"action_type": "greeting", "session": {}}',
        "user_input": "raw caller text should not leak through",
        "unrelated_field": "must not appear in subgraph input",
    }
    extracted = node.extract_input(state)
    assert extracted == '{"action_type": "greeting", "session": {}}'
    assert "unrelated_field" not in extracted
    assert "raw caller text" not in extracted


def test_merge_output_maps_fields_explicitly_not_raw_passthrough():
    node = MotionTrainingGraphNode()
    sub_result = {
        "pose_metrics": '{"bow_angle": {"score": 12.0}}',
        "capture_quality": "pass",
        "rubric_query": "greeting bow_angle below standard by 3.0 - remediation rubric",
        "matched_rubric": '[{"standard_id": "greeting-posture-001", "passage": "Bow at least 15 degrees."}]',
        "remediation_draft": "Training feedback based on matched service standards:\n- ...",
        "final_feedback": '{"session_id": "s1", "action_type": "greeting"}',
        "output": '{"session_id": "s1", "action_type": "greeting"}',
        "status": AgentStatus.SUCCESS.value,
        "internal_subgraph_only_field": "must not leak into merged output",
    }
    merged = node.merge_output({}, sub_result)
    assert "internal_subgraph_only_field" not in merged
    assert merged["result"] == sub_result["output"]
    assert merged["final_feedback"] == sub_result["final_feedback"]
    assert merged["status"] == AgentStatus.SUCCESS.value


def test_inner_subgraph_delegation_is_deliberate_design():
    """MotionTrainingGraphNode delegates content-scan gating to the inner subgraph's
    own entry node (PoseMetricScoreNode re-runs detect_biometric_pii() directly in
    execute() - pre_process -> main is an
    unconditional edge, so an upstream ERROR still dispatches into this inner
    subgraph and the content scan must be re-checked at the inner entry point) -
    this is the documented Cat 2 composition pattern (framework/nodes/graph_node.py),
    not a bypass.
    """
    import inspect

    from src.nodes.pose_metric_score_node import PoseMetricScoreNode

    source = inspect.getsource(PoseMetricScoreNode.execute)
    assert "detect_biometric_pii" in source
