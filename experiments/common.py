from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from nhs_demo.config import DATE_RANGE_EXTENSION_DAYS, DEFAULT_DATE_HORIZON_DAYS
from nhs_demo.orchestrator import DemoOrchestrator
from nhs_demo.schemas import (
    AuditLog,
    BookingOffer,
    IntakeSummary,
    PatientPreferences,
    RouteRequest,
    RoutingDecision,
    ScheduleOfferRequest,
    ScheduleOfferResponse,
    ScheduleRelaxRequest,
    Slot,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIXED_RANDOM_SEED = 20260303

RelaxationDecision = Tuple[Dict[str, bool], Dict[str, List[str]]]
RelaxationPolicy = Callable[[ScheduleOfferResponse, Mapping[str, Any]], RelaxationDecision]


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_results_dir()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_preferences(payload: Mapping[str, Any]) -> PatientPreferences:
    return PatientPreferences.model_validate(payload)


def build_intake_summary(scenario: Mapping[str, Any], run_id: str = "structured-run") -> IntakeSummary:
    intake = scenario["intake"]
    extracted_constraints = intake.get("extracted_constraints_text", [])
    return IntakeSummary(
        run_id=run_id,
        raw_text=intake["raw_text"],
        complaint_category=intake["complaint_category"],
        duration_text=intake.get("duration_text"),
        extracted_constraints_text=list(extracted_constraints),
        preferences=build_preferences(scenario["preferences"]),
        missing_fields=[],
    )


def build_routing_decision(
    scenario: Mapping[str, Any],
    run_id: str = "structured-run",
) -> RoutingDecision:
    return RoutingDecision(run_id=run_id, **scenario["routing"])


def accept_all_relaxations(
    offer: ScheduleOfferResponse,
    scenario: Mapping[str, Any],
) -> RelaxationDecision:
    answers = {question.key: True for question in offer.relaxation_questions}
    selections = dict(scenario.get("relaxation_selections", {}))
    return answers, selections


def reject_all_relaxations(
    offer: ScheduleOfferResponse,
    scenario: Mapping[str, Any],
) -> RelaxationDecision:
    answers = {question.key: False for question in offer.relaxation_questions}
    return answers, {}


def accept_first_relaxation_only(
    offer: ScheduleOfferResponse,
    scenario: Mapping[str, Any],
) -> RelaxationDecision:
    answers = {question.key: False for question in offer.relaxation_questions}
    if offer.relaxation_questions:
        answers[offer.relaxation_questions[0].key] = True
    selections = dict(scenario.get("relaxation_selections", {}))
    return answers, selections


def run_structured_workflow(
    orchestrator: DemoOrchestrator,
    scenario: Mapping[str, Any],
    *,
    use_routing_hint: bool = False,
    relaxation_policy: RelaxationPolicy | None = None,
    candidate_slots: Sequence[Slot] | None = None,
) -> Dict[str, Any]:
    intake_summary = build_intake_summary(scenario)
    routing_hint = build_routing_decision(scenario) if use_routing_hint and "routing" in scenario else None
    run_id = orchestrator.create_structured_run(
        intake_summary=intake_summary,
        routing_decision=routing_hint,
    )
    intake_for_run = intake_summary.model_copy(update={"run_id": run_id})

    if routing_hint is not None:
        routing_decision = routing_hint.model_copy(update={"run_id": run_id})
    else:
        route_response = orchestrator.route(
            RouteRequest(run_id=run_id, intake_summary=intake_for_run)
        )
        routing_decision = route_response.routing_decision

    current_preferences = intake_for_run.preferences
    negotiation_rounds = 0
    applied_relaxations: List[str] = []
    rejected_relaxations: List[str] = []

    while True:
        request = ScheduleOfferRequest(
            run_id=run_id,
            routing_decision=routing_decision,
            preferences=current_preferences,
        )
        if candidate_slots is None:
            offer = orchestrator.offer(request)
        else:
            offer = orchestrator.offer_with_candidate_slots(request, list(candidate_slots))

        if offer.status != "needs_relaxation":
            audit = orchestrator.audit(run_id)
            return {
                "run_id": run_id,
                "intake_summary": intake_for_run,
                "routing_decision": routing_decision,
                "offer": offer,
                "audit": audit,
                "final_preferences": current_preferences,
                "negotiation_rounds": negotiation_rounds,
                "applied_relaxations": applied_relaxations,
                "rejected_relaxations": rejected_relaxations,
            }

        if relaxation_policy is None:
            audit = orchestrator.audit(run_id)
            return {
                "run_id": run_id,
                "intake_summary": intake_for_run,
                "routing_decision": routing_decision,
                "offer": offer,
                "audit": audit,
                "final_preferences": current_preferences,
                "negotiation_rounds": negotiation_rounds,
                "applied_relaxations": applied_relaxations,
                "rejected_relaxations": rejected_relaxations,
            }

        answers, relaxation_selections = relaxation_policy(offer, scenario)
        relax_response = orchestrator.relax(
            ScheduleRelaxRequest(
                run_id=run_id,
                preferences=current_preferences,
                answers=answers,
                relaxation_selections=relaxation_selections,
            )
        )
        current_preferences = relax_response.updated_preferences
        negotiation_rounds += 1
        applied_relaxations.extend(relax_response.applied_relaxations)
        rejected_relaxations.extend(relax_response.rejected_relaxations)


def booking_to_slot(booking: BookingOffer) -> Slot:
    return Slot(
        slot_id=booking.slot_id,
        service_type=booking.service_type,
        clinician_id=booking.clinician_id,
        site=booking.site,
        modality=booking.modality,
        start_time=booking.start_time,
        duration_minutes=booking.duration_minutes,
    )


def candidate_horizon_for_preferences(preferences: PatientPreferences) -> int:
    return max(
        preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS,
        DEFAULT_DATE_HORIZON_DAYS + DATE_RANGE_EXTENSION_DAYS,
    )


def utility_for_booking(
    orchestrator: DemoOrchestrator,
    routing_decision: RoutingDecision,
    preferences: PatientPreferences,
    booking: BookingOffer | None,
    candidate_slots: Sequence[Slot] | None = None,
) -> float | None:
    if booking is None:
        return None

    slots = list(candidate_slots) if candidate_slots is not None else orchestrator.rota_agent.generate_slots(
        routing_decision,
        candidate_horizon_for_preferences(preferences),
    )
    target_slot = next((slot for slot in slots if slot.slot_id == booking.slot_id), None)
    if target_slot is None:
        target_slot = booking_to_slot(booking)
        slots = list(slots) + [target_slot]

    min_start = min(slot.start_time for slot in slots)
    max_start = max(slot.start_time for slot in slots)
    score, _ = orchestrator.patient_agent.score_slot(target_slot, preferences, min_start, max_start)
    return score


def stable_audit_payload(audit: AuditLog) -> Dict[str, Any]:
    payload = audit.model_dump(mode="json")
    payload.pop("run_id", None)
    payload.pop("created_at", None)
    payload.pop("updated_at", None)
    if payload.get("intake_summary"):
        payload["intake_summary"].pop("run_id", None)
    if payload.get("routing_decision"):
        payload["routing_decision"].pop("run_id", None)
    return payload


def audit_hash(audit: AuditLog) -> str:
    canonical_json = json.dumps(
        stable_audit_payload(audit),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def most_common_frequency(values: Iterable[str]) -> int:
    counter = Counter(values)
    return counter.most_common(1)[0][1] if counter else 0


def basic_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(float(values[0]), 3)

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return round(lower_value, 3)
    weight = position - lower_index
    interpolated = lower_value + (upper_value - lower_value) * weight
    return round(interpolated, 3)


def extended_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    stats = basic_stats(values)
    return {
        **stats,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
    }


def list_to_pipe_string(items: Sequence[str]) -> str:
    return "|".join(items)


# ---------------------------------------------------------------------------
# Statistical utilities for academic-grade evaluation
# ---------------------------------------------------------------------------

import random as _random


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = FIXED_RANDOM_SEED,
) -> Dict[str, float]:
    """Non-parametric bootstrap confidence interval.

    Returns a dict with 'lower', 'upper', and 'point' keys.
    Supports statistic='mean' or statistic='proportion' (for 0/1 data).
    """
    if not values:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0}

    rng = _random.Random(seed)
    n = len(values)

    if statistic == "proportion":
        point = sum(values) / n
    else:
        point = statistics.mean(values)

    replicates: List[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        if statistic == "proportion":
            replicates.append(sum(sample) / n)
        else:
            replicates.append(statistics.mean(sample))

    replicates.sort()
    lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
    hi_idx = int(math.floor((1 - alpha / 2) * n_bootstrap)) - 1
    return {
        "point": round(point, 4),
        "lower": round(replicates[lo_idx], 4),
        "upper": round(replicates[hi_idx], 4),
    }


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Cohen's d effect size between two groups.  Returns 0.0 if either
    group has fewer than 2 observations."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)
    pooled_sd = math.sqrt((var_a + var_b) / 2)
    if pooled_sd == 0:
        return 0.0
    return round((mean_a - mean_b) / pooled_sd, 4)


def ols_fit(x_values: Sequence[float], y_values: Sequence[float]) -> Dict[str, float]:
    """Simple ordinary-least-squares regression.  Returns slope, intercept,
    and R-squared.  Useful for characterising runtime growth."""
    n = len(x_values)
    if n < 2:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    ss_xx = sum((x - x_mean) ** 2 for x in x_values)
    ss_yy = sum((y - y_mean) ** 2 for y in y_values)

    if ss_xx == 0:
        return {"slope": 0.0, "intercept": y_mean, "r_squared": 0.0}

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0

    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
    }
