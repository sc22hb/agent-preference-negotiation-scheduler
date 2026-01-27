from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import ApptMode, ApptType, ClinicianRole, PatientRequest, Slot, UrgencyBand


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str


def is_eligible(slot: Slot, request: PatientRequest) -> EligibilityResult:
    if slot.appt_type != request.required_appt_type:
        return EligibilityResult(False, "appointment_type_mismatch")

    if request.must_be_role and slot.clinician_role != request.must_be_role:
        return EligibilityResult(False, "clinician_role_mismatch")

    if request.must_be_mode and slot.mode != request.must_be_mode:
        return EligibilityResult(False, "mode_mismatch")

    for window in request.unavailable_windows:
        if window.start_time <= slot.start_time < window.end_time:
            return EligibilityResult(False, "unavailable_time_window")

    if request.urgency_band == "SAME_DAY":
        # In this demo, SAME_DAY requires a same-day slot or tagged urgent slot
        if "same_day" not in slot.tags and slot.appt_type != "URGENT":
            return EligibilityResult(False, "urgency_violation")

    return EligibilityResult(True, "eligible")


def _day_name(dt: datetime) -> str:
    return dt.strftime("%a")


def _score_time_windows(slot: Slot, request: PatientRequest) -> Tuple[int, List[str]]:
    score = 0
    factors: List[str] = []
    for pref in request.preferred_time_windows:
        if pref.window.start_time <= slot.start_time < pref.window.end_time:
            score += pref.weight
            factors.append("preferred_time_window")
    return score, factors


def score_slot(slot: Slot, request: PatientRequest) -> Tuple[int, List[str]]:
    score = 0
    factors: List[str] = []

    time_score, time_factors = _score_time_windows(slot, request)
    if time_score:
        score += time_score
        factors.extend(time_factors)

    day = _day_name(slot.start_time)
    if day in request.preferred_days:
        score += request.preferred_days[day]
        factors.append("preferred_day")

    if slot.mode in request.preferred_modes:
        score += request.preferred_modes[slot.mode]
        factors.append("preferred_mode")

    if slot.slot_id in request.preferred_slot_ids:
        score += 30
        factors.append("preferred_slot")

    # Sooner bias: earlier appointments get a score bump
    score += request.soonest_weight
    factors.append("soonest_bias")

    return score, factors


def rank_slots(slots: List[Slot], request: PatientRequest) -> List[Tuple[Slot, int, List[str]]]:
    ranked: List[Tuple[Slot, int, List[str]]] = []
    for slot in slots:
        eligibility = is_eligible(slot, request)
        if not eligibility.eligible:
            continue
        score, factors = score_slot(slot, request)
        ranked.append((slot, score, factors))

    ranked.sort(key=lambda x: (-x[1], x[0].start_time))
    return ranked


def negotiate(
    requests: Dict[str, PatientRequest],
    slots: List[Slot],
    max_rounds: int = 5,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Deterministic multi-round negotiation (no LLM)."""
    if seed is not None:
        import random
        random.seed(seed)

    slot_map = {s.slot_id: s for s in slots}
    available = set(slot_map.keys())
    assigned: Dict[str, str] = {}
    log: List[str] = []

    for round_num in range(1, max_rounds + 1):
        if len(assigned) == len(requests):
            log.append("All requests assigned.")
            break

        bids: Dict[str, List[Tuple[int, str, List[str]]]] = {}
        for req_id, req in requests.items():
            if req_id in assigned:
                continue
            ranked = rank_slots([slot_map[s] for s in available], req)
            if not ranked:
                log.append(f"{req_id} has no eligible slots.")
                continue
            best_slot, best_score, best_factors = ranked[0]
            bids.setdefault(best_slot.slot_id, []).append((best_score, req_id, best_factors))

        if not bids:
            log.append("No bids placed; ending negotiation.")
            break

        for slot_id, bid_list in bids.items():
            bid_list.sort(key=lambda x: (-x[0], x[1]))
            winner_score, winner_id, winner_factors = bid_list[0]
            assigned[winner_id] = slot_id
            available.discard(slot_id)
            log.append(f"{slot_id} assigned to {winner_id} (score {winner_score}).")

    return assigned, log


__all__ = [
    "EligibilityResult",
    "is_eligible",
    "score_slot",
    "rank_slots",
    "negotiate",
]
