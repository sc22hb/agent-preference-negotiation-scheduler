from __future__ import annotations

from typing import List, Tuple

from .models import PatientRequest, Slot
from .nhs_engine import EligibilityResult, is_eligible, score_slot


def explain_eligibility(slot: Slot, request: PatientRequest) -> str:
    result: EligibilityResult = is_eligible(slot, request)
    if result.eligible:
        return "Eligible based on hard constraints."

    reason_map = {
        "appointment_type_mismatch": "Slot appointment type does not match the required type.",
        "clinician_role_mismatch": "Slot clinician role does not match the required role.",
        "mode_mismatch": "Slot mode does not match the required mode.",
        "unavailable_time_window": "Slot falls within an unavailable time window.",
        "urgency_violation": "Slot does not satisfy urgency constraints.",
    }
    return reason_map.get(result.reason, "Slot is not eligible.")


def explain_score(slot: Slot, request: PatientRequest) -> Tuple[int, List[str]]:
    score, factors = score_slot(slot, request)
    factor_map = {
        "preferred_time_window": "Matches a preferred time window",
        "preferred_day": "Matches a preferred day",
        "preferred_mode": "Matches a preferred mode",
        "preferred_slot": "Matches a preferred slot option",
        "soonest_bias": "Sooner appointment preference applied",
    }
    explanations = [factor_map.get(f, f) for f in factors]
    return score, explanations


def explain_assignment(slot: Slot, request: PatientRequest) -> str:
    eligibility_text = explain_eligibility(slot, request)
    score, factors = explain_score(slot, request)
    if not factors:
        factors_text = "No positive preference factors applied."
    else:
        factors_text = "; ".join(factors)
    return f"{eligibility_text} Utility score: {score}. Factors: {factors_text}."


__all__ = ["explain_eligibility", "explain_score", "explain_assignment"]
