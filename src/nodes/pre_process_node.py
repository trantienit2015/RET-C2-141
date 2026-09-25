"""AgentCore Platform v1.0 - RET-C2-141 InputValidateNode (outer pre_process slot).

S-1 trust gate (required_trust_level VERIFIED_EXTERNAL - motion data is
biometric). S-2 non-suppressible gate: regex/pattern scan rejects staff name /
社員ID / 個人番号 / credential in the session payload -> SecurityViolationError-
equivalent hard reject. Keypoint arrays are pseudonymised at ingest (session_id
only, no name) - not an LLM check, deterministic by design.
"""

import json
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import (
    VALID_ACTION_TYPES,
    detect_biometric_pii,
    extract_audit_safe_topic,
    validate_freemocap_session,
)


class InputValidateNode(FunctionNode):
    """Validate the freemocap session as biometric-PII-free and schema-valid."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        user_input = state.get("user_input", "")

        try:
            raw = json.loads(user_input) if isinstance(user_input, str) else user_input
        except (TypeError, ValueError):
            return {"status": AgentStatus.ERROR.value, "error_log": ["InputValidateNode: user_input is not valid JSON"]}

        if not isinstance(raw, dict):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: payload is not a valid session record"],
            }

        pii_hits = detect_biometric_pii(json.dumps(raw, ensure_ascii=False))
        if pii_hits:
            emit_trace_event("biometric_pii_input_blocked", {"violation_count": len(pii_hits)}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "InputValidateNode: staff-identity/mynumber/credential pattern detected - rejected, not processed"
                ],
            }

        session = raw.get("session")
        action_type = raw.get("action_type", "")
        if action_type not in VALID_ACTION_TYPES:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: invalid action_type '{action_type}'"],
            }

        validated_session, error = validate_freemocap_session(cast(dict[str, Any], session))
        if error:
            return {"status": AgentStatus.ERROR.value, "error_log": [f"InputValidateNode: {error}"]}

        emit_trace_event(
            "motion_session_ingested",
            extract_audit_safe_topic(action_type, len(cast(dict[str, Any], validated_session)["frames"])),
            state,
        )

        return {
            "action_type": action_type,
            "validated_input": to_json({"session": validated_session, "action_type": action_type}),
            "status": AgentStatus.SUCCESS.value,
        }
