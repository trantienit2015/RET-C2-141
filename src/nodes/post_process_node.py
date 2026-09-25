"""AgentCore Platform v1.0 - RET-C2-141 OutputValidateNode (outer post_process slot).

S-3 non-suppressible deterministic output gate (PB-test contract): regex scan blocks any staff name / 社員ID / 個人番号 /
raw keypoint coordinate dump from leaking into the feedback record. The hook
checks the assembled record's OWN serialized fields only, never cross-
referencing separately-read sibling `state` fields. On
catch, the output is dropped entirely (status=ERROR, audit logged) - never
suppressed.
"""

import json
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import detect_biometric_pii


class OutputValidateNode(FunctionNode):
    """Assemble the final output; non-suppressible biometric-PII leak re-check."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("status") in (AgentStatus.ERROR.value, AgentStatus.ERROR.value):
            return {"status": AgentStatus.ERROR.value}

        record = from_json(state.get("final_feedback"), None)
        if not isinstance(record, dict):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputValidateNode: missing assembled feedback record"],
            }

        emit_trace_event(
            "training_feedback_validated",
            {
                "action_type": record.get("action_type"),
                "capture_quality": record.get("capture_quality"),
                "correlation_id": state.get("correlation_id", ""),
            },
            state,
        )

        return {
            "formatted_output": record,
            "result": to_json(record),
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Non-suppressible re-check: no staff-identity/mynumber/credential leak in the record (own-dict fields only)."""
        record = from_json(state.get("result"), None)
        if not isinstance(record, dict):
            return state

        record_text = json.dumps(record, ensure_ascii=False)
        pii_hits = detect_biometric_pii(record_text)
        if pii_hits:
            emit_trace_event("output_gate_biometric_pii_leak_blocked", {"violation_count": len(pii_hits)}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    "Output gate: staff-identity/mynumber/credential pattern detected in feedback record - blocked (biometric-PII leak guard)"
                ],
            }

        return state
