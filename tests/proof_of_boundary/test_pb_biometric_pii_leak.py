# RET-C2-141 - PB test: biometric-PII leak contract.
#
# Contract: the serialized final feedback record must never contain a token
# matching the staff-identity / 個人番号 / credential regex set, regardless of
# what upstream data (KB content, LLM output) feeds into assembly.

import json

from src.services.service import detect_biometric_pii

STAFF_IDENTITY_LEAK = "Refer to 社員ID:98765 for the follow-up session."
MYNUMBER_LEAK = "Reference case 1234 5678 9012 for audit."
CREDENTIAL_LEAK = "Auth: Bearer abcdef1234567890ghijklmno"
CLEAN_TEXT = "Improve bow angle by 5 degrees for a stronger greeting posture."


class TestPBBiometricPIILeak:
    def test_staff_identity_detected(self):
        assert "staff_identity_pattern" in detect_biometric_pii(STAFF_IDENTITY_LEAK)

    def test_mynumber_detected(self):
        assert "mynumber_pattern" in detect_biometric_pii(MYNUMBER_LEAK)

    def test_credential_detected(self):
        assert "credential_pattern" in detect_biometric_pii(CREDENTIAL_LEAK)

    def test_clean_text_passes(self):
        assert detect_biometric_pii(CLEAN_TEXT) == []

    def test_serialized_record_with_leak_is_detected(self):
        record = {"session_id": "sess-1", "action_type": "greeting", "remediation": STAFF_IDENTITY_LEAK}
        assert detect_biometric_pii(json.dumps(record, ensure_ascii=False)) != []

    def test_serialized_clean_record_passes(self):
        record = {"session_id": "sess-1", "action_type": "greeting", "remediation": CLEAN_TEXT}
        assert detect_biometric_pii(json.dumps(record, ensure_ascii=False)) == []
