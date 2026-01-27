from __future__ import annotations

from typing import Optional, Tuple

from .models import ApptType, ClinicianRole, UrgencyBand


ESCALATION_KEYWORDS = [
    "chest pain",
    "breathing difficulty",
    "shortness of breath",
    "severe bleeding",
    "stroke",
    "unconscious",
    "heart attack",
]


def triage_request(free_text_reason: str) -> Tuple[UrgencyBand, ApptType, Optional[ClinicianRole]]:
    """Rule-based triage for NHS demo.

    Returns urgency band, required appointment type, and clinician role suggestion.
    If escalation keywords are detected, raises ValueError to indicate no scheduling.
    """
    text = free_text_reason.strip().lower()

    for keyword in ESCALATION_KEYWORDS:
        if keyword in text:
            raise ValueError("Emergency escalation required; do not schedule")

    if "repeat prescription" in text or "medication refill" in text:
        return "ROUTINE", "ROUTINE", "PHARMACIST"

    if "asthma" in text and ("worsening" in text or "flare" in text):
        return "SOON", "URGENT", "GP"

    if "rash" in text and ("long-standing" in text or "longstanding" in text):
        return "ROUTINE", "ROUTINE", "GP"

    if "test" in text or "blood" in text:
        return "ROUTINE", "TEST", "NURSE"

    # Default fallback for general cases
    return "ROUTINE", "ROUTINE", "GP"


__all__ = ["triage_request"]
