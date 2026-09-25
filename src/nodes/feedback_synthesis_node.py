"""AgentCore Platform v1.0 - RET-C2-141 FeedbackSynthesisNode (inner subgraph, step 4).

LLM-essential node by design: synthesizes language-
neutral training feedback from (metric scores + matched rubric), citing the
rubric standard ID; temperature 0.0 for consistent training guidance.
Deterministic fallback when no LLM is configured, or when a configured LLM
degrades (narrow provider/transport errors only - not a blanket swallow;
an advisory-degrade pattern for non-eligibility-gating LLM enrichment text).
"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import assemble_feedback_record, build_remediation_fallback


class FeedbackSynthesisNode(FunctionNode):
    """Synthesize training feedback and assemble the final structured record."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm: Any = None) -> None:
        super().__init__()
        self._llm = llm

    @staticmethod
    def _extract_text(raw: Any) -> str:
        """Normalize an LLM response (canonical dict or bare string fake) into text."""
        if isinstance(raw, dict):
            content = raw.get("content", "")
            return content if isinstance(content, str) else ""
        if isinstance(raw, str):
            return raw
        return ""

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        pose_metrics = from_json(state.get("pose_metrics"), {})
        matched_rubric = from_json(state.get("matched_rubric"), [])
        parsed = from_json(state.get("user_input"), {})
        action_type = parsed.get("action_type", "") if isinstance(parsed, dict) else ""
        session = parsed.get("session", {}) if isinstance(parsed, dict) else {}
        session_id = session.get("session_id", "")
        capture_quality = state.get("capture_quality", "pass")

        if self._llm is None:
            # Deterministic-core mode: no LLM configured at all, so the
            # deterministic remediation text IS the intended answer.
            remediation = build_remediation_fallback(pose_metrics, matched_rubric)
        else:
            # Canonical BaseLLM.complete(messages: list) -> dict {"content": str, ...}.
            messages = [
                {
                    "role": "system",
                    "content": "You are a retail training coach. Synthesize language-neutral "
                    "training feedback citing the rubric standard ID.",
                },
                {
                    "role": "user",
                    "content": f"Metrics: {pose_metrics}. Matched rubric standards: {matched_rubric}.",
                },
            ]
            # A configured LLM that fails (or returns an empty/malformed response)
            # is an error, never a silent degrade: the deterministic remediation
            # fallback is only a valid answer when NO llm is configured at all
            # (deterministic-core mode). Otherwise the caller cannot tell
            # "the coach spoke" from "the coach was unreachable".
            try:
                text = self._extract_text(self._llm.complete(messages, temperature=0.0))
            except (ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
                reason = f"LLM call failed: {type(exc).__name__}"
                emit_trace_event(
                    "training_feedback_llm_failed",
                    {"correlation_id": state.get("correlation_id", ""), "reason": type(exc).__name__},
                    state,
                )
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [*(state.get("error_log") or []), f"FeedbackSynthesisNode: {reason}"],
                }
            if not text:
                emit_trace_event(
                    "training_feedback_llm_failed",
                    {"correlation_id": state.get("correlation_id", ""), "reason": "empty_response"},
                    state,
                )
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [
                        *(state.get("error_log") or []),
                        "FeedbackSynthesisNode: LLM returned an empty or malformed response",
                    ],
                }
            remediation = text

        record = assemble_feedback_record(
            session_id, action_type, pose_metrics, matched_rubric, remediation, capture_quality
        )

        emit_trace_event(
            "training_feedback_synthesized",
            {
                "action_type": action_type,
                "llm_used": self._llm is not None,
                "correlation_id": state.get("correlation_id", ""),
            },
            state,
        )

        return {
            "remediation_draft": remediation,
            "final_feedback": to_json(record),
            "status": AgentStatus.SUCCESS.value,
        }
