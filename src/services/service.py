"""AgentCore Platform v1.0 - RET-C2-141 domain services.

Deterministic helpers only (PoseMetricScorer + TrainingRubricRAG "Tool" logic): freemocap v0.3.x session validation,
biometric-PII detect (staff identity / mynumber reject, not suppressible),
deterministic pose-metric geometry scoring with min_confidence fail-fast,
score-to-rubric-query mapping, rubric KB retrieval mock (real interface), and
deterministic feedback fallback. No agenticstar imports.
"""

from __future__ import annotations

from typing import Any
import math
import re

# freemocap output schema v0.3.x is pinned - see docs/02_design.md Input Contract.
FREEMOCAP_SCHEMA_VERSION = "0.3.x"
MIN_CONFIDENCE = 0.6
MAX_LOW_CONFIDENCE_FRAME_RATIO = 0.2  # fail-fast if >20% of frames are below MIN_CONFIDENCE

# Staff-identity / biometric-PII markers that must never appear in the input
# session payload or the output feedback. Non-suppressible per S-2/S-3 spec.
STAFF_IDENTITY_PATTERN = re.compile(r"(社員ID|staff\s*id|employee\s*id)[:\s]*\w+", re.IGNORECASE)
MYNUMBER_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
CREDENTIAL_PATTERN = re.compile(r"(bearer\s+[a-zA-Z0-9._-]{10,}|eyJ[a-zA-Z0-9._-]{10,})", re.IGNORECASE)

VALID_ACTION_TYPES = ("greeting", "hand_off", "register_stance")

# Deterministic keypoint -> retail-posture-metric mapping (locked in docs/02_design.md).
BOW_ANGLE_STANDARD = 15.0  # degrees, minimum acceptable bow angle for greeting
GREETING_STANCE_OFFSET_STANDARD = 0.1  # normalized shoulder-offset tolerance


def detect_biometric_pii(payload_text: str) -> list[str]:
    """S-2/S-3 non-suppressible: detect staff-identity/mynumber/credential markers."""
    hits = []
    if STAFF_IDENTITY_PATTERN.search(payload_text):
        hits.append("staff_identity_pattern")
    if MYNUMBER_PATTERN.search(payload_text):
        hits.append("mynumber_pattern")
    if CREDENTIAL_PATTERN.search(payload_text):
        hits.append("credential_pattern")
    return hits


def validate_freemocap_session(session: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Deterministic freemocap v0.3.x session-shape validation."""
    if not isinstance(session, dict):
        return None, "session is not a valid freemocap record"
    frames = session.get("frames")
    if not isinstance(frames, list) or not frames:
        return None, "session has no frames"
    for f in frames:
        if "pose_keypoints" not in f or "confidence" not in f:
            return None, "frame missing pose_keypoints or confidence"
    return session, None


def check_capture_quality(frames: list[dict[str, Any]]) -> str:
    """Fail-fast per the locked min_confidence spec: if too many frames are
    below MIN_CONFIDENCE, return 'fail' rather than silently scoring low-confidence data."""
    if not frames:
        return "fail"
    low_confidence_count = sum(1 for f in frames if f.get("confidence", 0.0) < MIN_CONFIDENCE)
    return "fail" if (low_confidence_count / len(frames)) > MAX_LOW_CONFIDENCE_FRAME_RATIO else "pass"


def _angle_between(p1: list[float], p2: list[float], p3: list[float]) -> float:
    """Deterministic geometry: angle at p2 formed by p1-p2-p3, in degrees."""
    v1 = [p1[i] - p2[i] for i in range(3)]
    v2 = [p3[i] - p2[i] for i in range(3)]
    dot = sum(v1[i] * v2[i] for i in range(3))
    mag1 = math.sqrt(sum(v1[i] ** 2 for i in range(3)))
    mag2 = math.sqrt(sum(v2[i] ** 2 for i in range(3)))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def score_pose_metrics(frames: list[dict[str, Any]], action_type: str) -> dict[str, Any]:
    """Deterministic pose-metric geometry scoring (PoseMetricScorer Tool logic).
    Computes bow angle / greeting-stance offset from keypoint coordinates,
    only using frames at or above MIN_CONFIDENCE."""
    valid_frames = [f for f in frames if f.get("confidence", 0.0) >= MIN_CONFIDENCE]
    metrics = {}

    bow_angles = []
    for f in valid_frames:
        kp = f.get("pose_keypoints", {})
        nose = kp.get("nose")
        shoulder = kp.get("left_shoulder")
        hip = kp.get("left_hip", kp.get("left_shoulder"))
        if nose and shoulder and hip:
            bow_angles.append(180.0 - _angle_between(nose, shoulder, hip))

    if bow_angles:
        avg_bow_angle = sum(bow_angles) / len(bow_angles)
        metrics["bow_angle"] = {
            "score": round(avg_bow_angle, 1),
            "confidence": round(sum(f.get("confidence", 0.0) for f in valid_frames) / max(1, len(valid_frames)), 2),
            "standard_id": "greeting-posture-001",
            "gap": round(avg_bow_angle - BOW_ANGLE_STANDARD, 1),
        }

    return metrics


def map_score_to_rubric_query(pose_metrics: dict[str, Any], action_type: str) -> str:
    """Deterministic score -> rubric-query mapping table (no LLM), locked in docs/02_design.md."""
    for metric_name, m in pose_metrics.items():
        if m["gap"] < 0:
            return f"{action_type} {metric_name} below standard by {abs(m['gap'])} - remediation rubric"
    return f"{action_type} within standard - reinforcement rubric"


def retrieve_rubric_passages(query: str, kb: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Deterministic mock vector search over the service-standard rubric KB (TrainingRubricRAG Tool logic)."""
    kb = kb or []
    q_lower = query.lower()
    matches = []
    for doc in kb:
        text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
        if any(tok in text for tok in q_lower.split() if len(tok) > 1):
            matches.append({"standard_id": doc.get("standard_id", ""), "passage": doc.get("content", "")[:200]})
    return matches[:5]


def build_remediation_fallback(pose_metrics: dict[str, Any], matched_rubric: list[dict[str, Any]]) -> str:
    """Deterministic fallback remediation text when no LLM is configured."""
    if not matched_rubric:
        return "No matching training rubric found - manual trainer review recommended."
    lines = [f"- [{r['standard_id']}] {r['passage']}" for r in matched_rubric]
    return "Training feedback based on matched service standards:\n" + "\n".join(lines)


def assemble_feedback_record(
    session_id: str,
    action_type: str,
    pose_metrics: dict[str, Any],
    matched_rubric: list[dict[str, Any]],
    remediation: str,
    capture_quality: str,
) -> dict[str, Any]:
    """Assemble the final structured training-feedback record per the locked output schema."""
    return {
        "session_id": session_id,
        "action_type": action_type,
        "metrics": [{"metric_name": k, **v} for k, v in pose_metrics.items()],
        "matched_rubric": matched_rubric,
        "remediation": remediation,
        "capture_quality": capture_quality,
    }


def extract_audit_safe_topic(action_type: str, frame_count: int) -> dict[str, Any]:
    """S-4: audit-safe metadata - action type + frame count only, never raw keypoints or staff identity."""
    return {"action_type": action_type, "frame_count": frame_count}
