from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import PatientRequest, Slot
from .nhs_engine import is_eligible, score_slot


@dataclass
class MetricsResult:
    assignment_rate: float
    average_utility: float
    fairness_variance: float
    negotiation_rounds: int
    escalation_rate: float


def compute_assignment_rate(assignments: Dict[str, str], total_requests: int) -> float:
    if total_requests == 0:
        return 0.0
    return len(assignments) / total_requests


def compute_average_utility(
    assignments: Dict[str, str],
    requests: Dict[str, PatientRequest],
    slot_map: Dict[str, Slot],
) -> float:
    if not assignments:
        return 0.0
    scores: List[int] = []
    for req_id, slot_id in assignments.items():
        slot = slot_map[slot_id]
        score, _ = score_slot(slot, requests[req_id])
        scores.append(score)
    return sum(scores) / len(scores)


def compute_fairness_variance(
    assignments: Dict[str, str],
    requests: Dict[str, PatientRequest],
    slot_map: Dict[str, Slot],
) -> float:
    if not assignments:
        return 0.0
    scores: List[int] = []
    for req_id, slot_id in assignments.items():
        slot = slot_map[slot_id]
        score, _ = score_slot(slot, requests[req_id])
        scores.append(score)
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def compute_escalation_rate(escalated_count: int, total_requests: int) -> float:
    if total_requests == 0:
        return 0.0
    return escalated_count / total_requests


def evaluate_run(
    assignments: Dict[str, str],
    requests: Dict[str, PatientRequest],
    slots: List[Slot],
    negotiation_rounds: int,
    escalated_count: int = 0,
) -> MetricsResult:
    slot_map = {s.slot_id: s for s in slots}
    assignment_rate = compute_assignment_rate(assignments, len(requests))
    average_utility = compute_average_utility(assignments, requests, slot_map)
    fairness_variance = compute_fairness_variance(assignments, requests, slot_map)
    escalation_rate = compute_escalation_rate(escalated_count, len(requests))
    return MetricsResult(
        assignment_rate=assignment_rate,
        average_utility=average_utility,
        fairness_variance=fairness_variance,
        negotiation_rounds=negotiation_rounds,
        escalation_rate=escalation_rate,
    )


__all__ = [
    "MetricsResult",
    "compute_assignment_rate",
    "compute_average_utility",
    "compute_fairness_variance",
    "compute_escalation_rate",
    "evaluate_run",
]
