# RET-C2-141 - Integration test: full graph compile + invoke (Cat 2 outer + inner).

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.schemas.state import to_json


def _good_frame(confidence=0.9):
    return {
        "frame_number": 1,
        "pose_keypoints": {"nose": [0, 0, 1], "left_shoulder": [0, -1, 0], "left_hip": [0, -2, -1]},
        "confidence": confidence,
    }


GOOD_SESSION = to_json({"session": {"session_id": "sess-1", "frames": [_good_frame(), _good_frame(), _good_frame()]}, "action_type": "greeting"})
LOW_CONFIDENCE_SESSION = to_json({"session": {"session_id": "sess-2", "frames": [_good_frame(confidence=0.1) for _ in range(5)] + [_good_frame()]}, "action_type": "greeting"})
STAFF_IDENTITY_SESSION = to_json({"session": {"session_id": "sess-3", "frames": [_good_frame()]}, "action_type": "greeting", "note": "社員ID:12345"})
INVALID_ACTION_SESSION = to_json({"session": {"session_id": "sess-4", "frames": [_good_frame()]}, "action_type": "dance"})

KB = [{"standard_id": "greeting-posture-001", "title": "Greeting posture", "content": "Bow angle should be at least 15 degrees for a formal greeting."}]


class TestAgentIntegration:
    def test_greeting_session_scored(self):
        agent = Graph(config={"max_retry": 1, "kb": KB})
        agent.compile()
        ctx = InvocationContext(session_id="it-1", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="trainer-1")
        result = agent.invoke(GOOD_SESSION, ctx=ctx)

        assert result["status"] == "success"
        assert len(result.get("node_history", [])) >= 5

    def test_low_confidence_session_fails(self):
        agent = Graph(config={"max_retry": 1, "kb": KB})
        agent.compile()
        ctx = InvocationContext(session_id="it-2", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="trainer-1")
        result = agent.invoke(LOW_CONFIDENCE_SESSION, ctx=ctx)
        assert result["status"] in ("error", "cancelled")

    def test_staff_identity_rejected(self):
        agent = Graph(config={"max_retry": 1, "kb": KB})
        agent.compile()
        ctx = InvocationContext(session_id="it-3", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="trainer-1")
        result = agent.invoke(STAFF_IDENTITY_SESSION, ctx=ctx)
        assert result["status"] in ("error", "cancelled")

    def test_invalid_action_type_error(self):
        agent = Graph(config={"max_retry": 1, "kb": KB})
        agent.compile()
        ctx = InvocationContext(session_id="it-4", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="trainer-1")
        result = agent.invoke(INVALID_ACTION_SESSION, ctx=ctx)
        assert result["status"] in ("error", "cancelled")
