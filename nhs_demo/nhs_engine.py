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


class ProviderAgent:
    def __init__(self, role: ClinicianRole, slots: List[Slot]) -> None:
        self.role = role
        self.slot_map = {s.slot_id: s for s in slots}
        self.available = set(self.slot_map.keys())

    def available_slots(self) -> List[Slot]:
        return [self.slot_map[sid] for sid in self.available]

    def reserve(self, slot_id: str) -> None:
        if slot_id in self.available:
            self.available.remove(slot_id)

    def get_slot(self, slot_id: str) -> Slot:
        return self.slot_map[slot_id]


class PatientAgent:
    def __init__(self, request_id: str, request: PatientRequest) -> None:
        self.request_id = request_id
        self.request = request

    def score_all(self, slots: List[Slot]) -> List[Tuple[Slot, int, List[str]]]:
        return rank_slots(slots, self.request)

    def bid(self, slots: List[Slot]) -> Optional[Tuple[Slot, int, List[str]]]:
        ranked = self.score_all(slots)
        if not ranked:
            return None
        return ranked[0]

    @property
    def urgency_priority(self) -> int:
        return _urgency_priority(self.request.urgency_band)


class SchedulerAgent:
    def __init__(self, providers: List[ProviderAgent], patients: List[PatientAgent]) -> None:
        self.providers = providers
        self.patients = patients

    def negotiate(self, max_rounds: int = 5, seed: Optional[int] = None) -> Tuple[Dict[str, str], List[str]]:
        if seed is not None:
            import random
            random.seed(seed)

        assigned: Dict[str, str] = {}
        log: List[str] = []

        for round_num in range(1, max_rounds + 1):
            log.append(f"[Scheduler] Round {round_num} begins.")
            if len(assigned) == len(self.patients):
                log.append("All requests assigned.")
                break

            bids: Dict[str, List[Tuple[int, int, str, List[str]]]] = {}
            for patient in self.patients:
                if patient.request_id in assigned:
                    continue
                available_slots: List[Slot] = []
                for provider in self.providers:
                    available_slots.extend(provider.available_slots())
                ranked = patient.score_all(available_slots)
                if ranked:
                    score_list = ", ".join(
                        f"{slot.slot_id}:{score}" for slot, score, _ in ranked
                    )
                    log.append(f"[PatientAgent:{patient.request_id}] Scores => {score_list}")
                bid = ranked[0] if ranked else None
                if not bid:
                    log.append(f"[PatientAgent:{patient.request_id}] No eligible slots. Requesting alternatives.")
                    continue
                best_slot, best_score, best_factors = bid
                log.append(f"[PatientAgent:{patient.request_id}] Bids for {best_slot.slot_id} (score {best_score}).")
                bids.setdefault(best_slot.slot_id, []).append(
                    (patient.urgency_priority, best_score, patient.request_id, best_factors)
                )

            if not bids:
                log.append("[Scheduler] No bids placed; ending negotiation.")
                break

            for slot_id, bid_list in bids.items():
                bid_list.sort(key=lambda x: (-x[0], -x[1], x[2]))
                _, winner_score, winner_id, _ = bid_list[0]
                assigned[winner_id] = slot_id
                for provider in self.providers:
                    if slot_id in provider.available:
                        provider.reserve(slot_id)
                        log.append(f"[ProviderAgent:{provider.role}] Reserved {slot_id}.")
                        break
                log.append(f"[Scheduler] Assigned {slot_id} to {winner_id} (score {winner_score}).")

        return assigned, log


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


def _urgency_priority(urgency: UrgencyBand) -> int:
    if urgency == "SAME_DAY":
        return 3
    if urgency == "SOON":
        return 2
    return 1


def _soonest_bonus(slot: Slot, min_start: datetime, max_start: datetime, weight: int) -> int:
    if max_start == min_start:
        return weight
    span = (max_start - min_start).total_seconds()
    if span <= 0:
        return weight
    position = (slot.start_time - min_start).total_seconds() / span
    return int(round(weight * (1 - position)))


def score_slot(
    slot: Slot,
    request: PatientRequest,
    min_start: Optional[datetime] = None,
    max_start: Optional[datetime] = None,
) -> Tuple[int, List[str]]:
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
    if min_start and max_start:
        bonus = _soonest_bonus(slot, min_start, max_start, request.soonest_weight)
        score += bonus
        if bonus > 0:
            factors.append("soonest_bias")
    else:
        score += request.soonest_weight
        factors.append("soonest_bias")

    return score, factors


def rank_slots(slots: List[Slot], request: PatientRequest) -> List[Tuple[Slot, int, List[str]]]:
    ranked: List[Tuple[Slot, int, List[str]]] = []
    if slots:
        min_start = min(s.start_time for s in slots)
        max_start = max(s.start_time for s in slots)
    else:
        min_start = None
        max_start = None
    for slot in slots:
        eligibility = is_eligible(slot, request)
        if not eligibility.eligible:
            continue
        score, factors = score_slot(slot, request, min_start=min_start, max_start=max_start)
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
    slots_by_role: Dict[ClinicianRole, List[Slot]] = {"GP": [], "NURSE": [], "PHARMACIST": []}
    for slot in slots:
        slots_by_role[slot.clinician_role].append(slot)
    providers = [ProviderAgent(role, role_slots) for role, role_slots in slots_by_role.items()]
    patients = [PatientAgent(req_id, req) for req_id, req in requests.items()]
    scheduler = SchedulerAgent(providers, patients)
    return scheduler.negotiate(max_rounds=max_rounds, seed=seed)


__all__ = [
    "EligibilityResult",
    "ProviderAgent",
    "PatientAgent",
    "SchedulerAgent",
    "is_eligible",
    "score_slot",
    "rank_slots",
    "negotiate",
]
