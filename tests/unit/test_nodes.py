# RET-C2-141 - Unit tests: per-node success + error/edge paths.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.feedback_synthesis_node import FeedbackSynthesisNode
from src.nodes.pose_metric_score_node import PoseMetricScoreNode
from src.nodes.post_process_node import OutputValidateNode
from src.nodes.pre_process_node import InputValidateNode
from src.nodes.rubric_query_node import RubricQueryNode
from src.nodes.training_rubric_retrieve_node import TrainingRubricRetrieveNode
from src.schemas.state import from_json, to_json

KB = [{"standard_id": "greeting-posture-001", "title": "Greeting posture standard", "content": "Bow angle should be at least 15 degrees for a formal greeting."}]


def _good_frame(confidence=0.9):
    return {
        "frame_number": 1,
        "pose_keypoints": {"nose": [0, 0, 1], "left_shoulder": [0, -1, 0], "left_hip": [0, -2, -1]},
        "confidence": confidence,
    }


def _session(frames=None, session_id="sess-1"):
    return {"session_id": session_id, "frames": frames if frames is not None else [_good_frame(), _good_frame(), _good_frame()]}


class TestInputValidateNode:
    def test_success(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "greeting"})}
        r = InputValidateNode().execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        assert r["action_type"] == "greeting"

    def test_invalid_json_error(self):
        assert InputValidateNode().execute({"user_input": "not json"})["status"] == AgentStatus.ERROR

    def test_invalid_action_type_error(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "dance"})}
        assert InputValidateNode().execute(state)["status"] == AgentStatus.ERROR

    def test_staff_identity_rejected(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "greeting", "note": "社員ID:12345"})}
        assert InputValidateNode().execute(state)["status"] == AgentStatus.ERROR

    def test_mynumber_rejected(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "greeting", "note": "1234 5678 9012"})}
        assert InputValidateNode().execute(state)["status"] == AgentStatus.ERROR

    def test_malformed_session_error(self):
        state = {"user_input": to_json({"session": {"session_id": "x", "frames": []}, "action_type": "greeting"})}
        assert InputValidateNode().execute(state)["status"] == AgentStatus.ERROR


def _inner_state(session=None, action_type="greeting"):
    return {"user_input": to_json({"session": session or _session(), "action_type": action_type})}


class TestPoseMetricScoreNode:
    def test_success(self):
        r = PoseMetricScoreNode().execute(_inner_state())
        assert r["status"] == AgentStatus.SUCCESS
        assert r["capture_quality"] == "pass"

    def test_low_confidence_fail_fast(self):
        low_conf_frames = [_good_frame(confidence=0.2) for _ in range(4)] + [_good_frame(confidence=0.9)]
        state = _inner_state(session=_session(frames=low_conf_frames))
        r = PoseMetricScoreNode().execute(state)
        assert r["status"] == AgentStatus.ERROR
        assert r["capture_quality"] == "fail"

    def test_missing_session_error(self):
        assert PoseMetricScoreNode().execute({"user_input": None})["status"] == AgentStatus.ERROR


class TestRubricQueryNode:
    def test_success(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        state = {**_inner_state(), "pose_metrics": to_json(metrics)}
        r = RubricQueryNode().execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        assert "below standard" in r["rubric_query"]

    def test_missing_metrics_error(self):
        assert RubricQueryNode().execute({**_inner_state(), "pose_metrics": None})["status"] == AgentStatus.ERROR


class TestTrainingRubricRetrieveNode:
    def test_success(self):
        state = {"rubric_query": "greeting bow_angle below standard - remediation rubric"}
        r = TrainingRubricRetrieveNode(kb=KB).execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        matches = from_json(r["matched_rubric"], [])
        assert len(matches) >= 1

    def test_missing_query_error(self):
        assert TrainingRubricRetrieveNode(kb=KB).execute({"rubric_query": ""})["status"] == AgentStatus.ERROR


class _DictLLM:
    """Canonical BaseLLM.complete() fake - returns {"content": str, ...}."""

    def complete(self, messages, **kwargs):
        return {"content": "Canonical dict feedback citing greeting-posture-001.", "model": "fake"}


class _RaisingLLM:
    """Provider transport failure - must surface as ERROR, never a silent degrade."""

    def complete(self, messages, **kwargs):
        raise ConnectionError("provider unavailable")


class _EmptyLLM:
    """Canonical shape but empty content - must surface as ERROR, not fallback-as-SUCCESS."""

    def complete(self, messages, **kwargs):
        return {"content": "", "model": "fake"}


class TestFeedbackSynthesisNode:
    def test_fallback_remediation(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass"}
        r = FeedbackSynthesisNode().execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        record = from_json(r["final_feedback"], {})
        assert record["capture_quality"] == "pass"

    def test_canonical_dict_llm_response_is_normalized(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass"}
        r = FeedbackSynthesisNode(llm=_DictLLM()).execute(state)
        assert r["status"] == AgentStatus.SUCCESS
        assert isinstance(r["remediation_draft"], str)
        assert "Canonical dict feedback" in r["remediation_draft"]

    def test_configured_llm_failure_is_an_error_not_a_silent_degrade(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass"}
        r = FeedbackSynthesisNode(llm=_RaisingLLM()).execute(state)
        # A configured-but-failing LLM must surface as ERROR: returning the
        # deterministic fallback under SUCCESS makes "the coach spoke" and "the
        # coach was unreachable" indistinguishable to the caller.
        assert r["status"] == AgentStatus.ERROR.value
        assert any("LLM call failed" in e for e in r["error_log"])

    def test_configured_llm_empty_response_is_an_error(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass"}
        r = FeedbackSynthesisNode(llm=_EmptyLLM()).execute(state)
        assert r["status"] == AgentStatus.ERROR.value
        assert any("empty or malformed" in e for e in r["error_log"])

    def test_no_llm_configured_uses_deterministic_fallback(self):
        # The only valid deterministic-fallback mode: no LLM configured at all.
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass"}
        r = FeedbackSynthesisNode(llm=None).execute(state)
        assert r["status"] == AgentStatus.SUCCESS.value
        assert "matched service standards" in r["remediation_draft"]


class TestOutputValidateNode:
    def test_success(self):
        record = {"session_id": "sess-1", "action_type": "greeting", "metrics": [], "matched_rubric": [], "remediation": "ok", "capture_quality": "pass"}
        r = OutputValidateNode().execute({"final_feedback": to_json(record)})
        assert r["status"] == AgentStatus.SUCCESS

    def test_upstream_error_short_circuits(self):
        assert OutputValidateNode().execute({"status": AgentStatus.ERROR})["status"] == AgentStatus.ERROR

    def test_extra_gate_blocks_staff_identity_leak(self):
        bad = to_json({"remediation": "Refer to 社員ID:98765 for follow-up."})
        out = OutputValidateNode()._extra_security_gate_output({"result": bad})
        assert out["status"] == AgentStatus.ERROR

    def test_extra_gate_blocks_mynumber_leak(self):
        bad = to_json({"remediation": "Case ref 1234 5678 9012."})
        out = OutputValidateNode()._extra_security_gate_output({"result": bad})
        assert out["status"] == AgentStatus.ERROR

    def test_extra_gate_passthrough_clean_record(self):
        good = to_json({"remediation": "Improve bow angle by 5 degrees."})
        state = {"result": good}
        assert OutputValidateNode()._extra_security_gate_output(state) is state


# S-1 trust-gate reject coverage.
# InputValidateNode, PoseMetricScoreNode, RubricQueryNode, TrainingRubricRetrieveNode,
# FeedbackSynthesisNode all declare required_trust_level=VERIFIED_EXTERNAL, but existing tests above
# call .execute(state) directly, bypassing BaseNode.__call__()'s S-1 trust gate.
class TestTrustGateReject:
    def test_input_validate_rejects_insufficient_trust(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "greeting"}), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = InputValidateNode()(state)
        assert str(result["status"]).lower().endswith("error")
        assert any("trust gate denied" in msg.lower() for msg in result.get("error_log", []))

    def test_input_validate_succeeds_with_sufficient_trust(self):
        state = {"user_input": to_json({"session": _session(), "action_type": "greeting"}), "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = InputValidateNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_pose_metric_score_rejects_insufficient_trust(self):
        state = {**_inner_state(), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = PoseMetricScoreNode()(state)
        assert str(result["status"]).lower().endswith("error")
        assert any("trust gate denied" in msg.lower() for msg in result.get("error_log", []))

    def test_pose_metric_score_succeeds_with_sufficient_trust(self):
        state = {**_inner_state(), "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = PoseMetricScoreNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_rubric_query_rejects_insufficient_trust(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = RubricQueryNode()(state)
        assert str(result["status"]).lower().endswith("error")
        assert any("trust gate denied" in msg.lower() for msg in result.get("error_log", []))

    def test_rubric_query_succeeds_with_sufficient_trust(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = RubricQueryNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_training_rubric_retrieve_rejects_insufficient_trust(self):
        state = {"rubric_query": "greeting bow_angle below standard - remediation rubric", "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = TrainingRubricRetrieveNode(kb=KB)(state)
        assert str(result["status"]).lower().endswith("error")
        assert any("trust gate denied" in msg.lower() for msg in result.get("error_log", []))

    def test_training_rubric_retrieve_succeeds_with_sufficient_trust(self):
        state = {"rubric_query": "greeting bow_angle below standard - remediation rubric", "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = TrainingRubricRetrieveNode(kb=KB)(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_feedback_synthesis_rejects_insufficient_trust(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass", "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = FeedbackSynthesisNode()(state)
        assert str(result["status"]).lower().endswith("error")
        assert any("trust gate denied" in msg.lower() for msg in result.get("error_log", []))

    def test_feedback_synthesis_succeeds_with_sufficient_trust(self):
        metrics = {"bow_angle": {"score": 10.0, "confidence": 0.9, "standard_id": "greeting-posture-001", "gap": -5.0}}
        rubric = [{"standard_id": "greeting-posture-001", "passage": "Bow angle should be at least 15 degrees."}]
        state = {**_inner_state(), "pose_metrics": to_json(metrics), "matched_rubric": to_json(rubric), "capture_quality": "pass", "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = FeedbackSynthesisNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
