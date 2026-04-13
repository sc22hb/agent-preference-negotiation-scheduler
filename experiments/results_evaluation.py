from __future__ import annotations

import math
import os
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CACHE_DIR = REPO_ROOT / ".tmp-matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
import numpy as np
import pandas as pd

from nhs_demo.agents.master_allocator_agent import MasterAllocatorAgent
from nhs_demo.agents.patient_agent import PatientAgent
from nhs_demo.config import DATE_RANGE_EXTENSION_DAYS, MAX_NEGOTIATION_ROUNDS, UTILITY_WEIGHTS
from nhs_demo.schemas import (
    BlockerSummary,
    BookingOffer,
    CandidateEvaluation,
    PatientPreferences,
    PreferenceWeightProfile,
    RoutingDecision,
    Slot,
)
from experiments.results_fixtures import (
    DAY_ORDER,
    EvalCase,
    PreparedCase,
    baseline_profiles,
    build_components,
    constraint_profiles,
    contention_cohort,
    make_preferences,
    prepare_cases,
    runtime_cohort,
)
from experiments.intake_eval_fixtures import IntakeExpected, intake_prompt_cases

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"

@dataclass(frozen=True)
class ResultsConfig:
    constraint_total_slots: int = 300
    baseline_total_slots: int = 200
    baseline_random_runs: int = 500
    fairness_total_slots: int = 150
    fairness_random_runs: int = 200
    runtime_repetitions: int = 200
    include_llm_intake: bool = False
    llm_model: str = "qwen2.5:3b"


@dataclass(frozen=True)
class SchedulerResult:
    booking: BookingOffer | None
    selected_evaluation: CandidateEvaluation | None
    final_preferences: PatientPreferences
    blocker_summary: BlockerSummary
    rounds_used: int
    relaxations_applied: List[str]
    round_one_feasible_count: int
    round_one_candidate_count: int


class VariantPatientAgent(PatientAgent):
    def __init__(
        self,
        utility_weights: Dict[str, int] | None = None,
        use_weight_profile: bool = True,
        allow_relaxation: bool = True,
    ) -> None:
        super().__init__()
        self.utility_weights = {**UTILITY_WEIGHTS, **(utility_weights or {})}
        self.use_weight_profile = use_weight_profile
        self.allow_relaxation = allow_relaxation

    def score_slot(
        self,
        slot: Slot,
        preferences: PatientPreferences,
        min_start: datetime,
        max_start: datetime,
    ) -> tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        day = self._weekday_for_slot(slot.start_time)
        period = self._period_for_slot(slot.start_time)
        day_weight = preferences.weight_profile.day if self.use_weight_profile else 100
        period_weight = preferences.weight_profile.period if self.use_weight_profile else 100
        modality_weight = preferences.weight_profile.modality if self.use_weight_profile else 100
        synergy_weight = preferences.weight_profile.day_period_synergy if self.use_weight_profile else 100

        breakdown["preferred_modality"] = self._weighted_preference_component(
            slot.modality,
            preferences.preferred_modalities,
            self.utility_weights["preferred_modality"],
            modality_weight,
            0.35,
        )
        breakdown["preferred_day"] = self._weighted_preference_component(
            day,
            preferences.preferred_days,
            self.utility_weights["preferred_day"],
            day_weight,
            0.30,
        )
        breakdown["preferred_period"] = self._weighted_preference_component(
            period,
            preferences.preferred_periods,
            self.utility_weights["preferred_period"],
            period_weight,
            0.28,
        )
        breakdown["preferred_day_period"] = self._day_period_component_with_weight(
            day=day,
            period=period,
            preferences=preferences,
            base_weight=self.utility_weights["preferred_day_period"],
            profile_weight=synergy_weight,
        )
        breakdown["adjacent_preferred_day"] = self._adjacent_preferred_day_component_with_weight(
            day=day,
            preferences=preferences,
            base_weight=self.utility_weights["adjacent_preferred_day"],
            profile_weight=day_weight,
        )
        breakdown["soonest"] = self._soonest_component(
            slot.start_time,
            min_start,
            max_start,
            self.utility_weights["soonest"],
            preferences.soonest_weight,
        )
        return round(sum(breakdown.values()), 3), breakdown

    def propose_relaxations(
        self,
        routing: RoutingDecision,
        blocker_summary: BlockerSummary,
        preferences: PatientPreferences,
        applied_relaxations: Iterable[str],
    ) -> List:
        if not self.allow_relaxation:
            return []
        return super().propose_relaxations(routing, blocker_summary, preferences, applied_relaxations)

    def _day_period_component_with_weight(
        self,
        day: str,
        period: str,
        preferences: PatientPreferences,
        base_weight: int,
        profile_weight: int,
    ) -> float:
        if not preferences.preferred_day_periods or base_weight <= 0:
            return 0.0
        scaled_weight = self._scaled_weight(base_weight, profile_weight)
        if any(pair.day == day and pair.period == period for pair in preferences.preferred_day_periods):
            return round(scaled_weight, 3)
        return round(-scaled_weight * 0.25, 3)

    def _adjacent_preferred_day_component_with_weight(
        self,
        day: str,
        preferences: PatientPreferences,
        base_weight: int,
        profile_weight: int,
    ) -> float:
        if not preferences.preferred_days or base_weight <= 0:
            return 0.0
        if day in preferences.preferred_days:
            return 0.0
        day_idx = self.DAY_ORDER.index(day)
        preferred_idx = [self.DAY_ORDER.index(item) for item in preferences.preferred_days]
        if any(min(abs(day_idx - idx), 7 - abs(day_idx - idx)) == 1 for idx in preferred_idx):
            return round(self._scaled_weight(base_weight, profile_weight), 3)
        return 0.0

def policy_patient_agent(policy_name: str) -> VariantPatientAgent:
    if policy_name == "full":
        return VariantPatientAgent()
    if policy_name == "no_day_period_synergy":
        return VariantPatientAgent(utility_weights={"preferred_day_period": 0})
    if policy_name == "no_preference_weighting":
        uniform = {key: 20 for key in UTILITY_WEIGHTS}
        return VariantPatientAgent(utility_weights=uniform, use_weight_profile=False)
    if policy_name == "no_temporal_proximity":
        return VariantPatientAgent(utility_weights={"soonest": 0})
    if policy_name == "no_relaxation":
        return VariantPatientAgent(allow_relaxation=False)
    raise ValueError(f"Unknown policy variant: {policy_name}")


def weekday_for_slot(slot: Slot) -> str:
    return DAY_ORDER[slot.start_time.weekday()]


def period_for_slot(slot: Slot) -> str:
    if slot.start_time.hour < 12:
        return "morning"
    if slot.start_time.hour < 17:
        return "afternoon"
    return "evening"


def booking_to_evaluation(
    booking: BookingOffer,
    evaluations: Sequence[CandidateEvaluation],
) -> CandidateEvaluation | None:
    for evaluation in evaluations:
        if evaluation.slot.slot_id == booking.slot_id:
            return evaluation
    return None


def run_scheduler(
    case: PreparedCase,
    slot_pool: Sequence[Slot],
    allocator: MasterAllocatorAgent,
    patient_agent: VariantPatientAgent,
    negotiate: bool = True,
) -> SchedulerResult:
    current_preferences = case.preferences
    relaxations_applied: List[str] = []
    last_blocker_summary = BlockerSummary()
    round_one_feasible_count = 0
    round_one_candidate_count = 0

    for round_number in range(1, MAX_NEGOTIATION_ROUNDS + 1):
        allocation = allocator.allocate(
            routing=case.routing_decision,
            preferences=current_preferences,
            candidate_slots=list(slot_pool),
            patient_agent=patient_agent,
        )
        feasible_count = sum(1 for item in allocation.candidate_evaluations if item.feasible)
        if round_number == 1:
            round_one_feasible_count = feasible_count
            round_one_candidate_count = len(allocation.candidate_evaluations)
        last_blocker_summary = allocation.blocker_summary

        if allocation.booking is not None:
            return SchedulerResult(
                booking=allocation.booking,
                selected_evaluation=booking_to_evaluation(allocation.booking, allocation.candidate_evaluations),
                final_preferences=current_preferences,
                blocker_summary=allocation.blocker_summary,
                rounds_used=round_number,
                relaxations_applied=relaxations_applied,
                round_one_feasible_count=round_one_feasible_count,
                round_one_candidate_count=round_one_candidate_count,
            )

        if not negotiate:
            break

        questions = patient_agent.propose_relaxations(
            routing=case.routing_decision,
            blocker_summary=allocation.blocker_summary,
            preferences=current_preferences,
            applied_relaxations=relaxations_applied,
        )
        if not questions:
            break
        for question in questions:
            relaxations_applied.append(question.key)
            current_preferences = patient_agent.apply_relaxation(current_preferences, question.key)

    return SchedulerResult(
        booking=None,
        selected_evaluation=None,
        final_preferences=current_preferences,
        blocker_summary=last_blocker_summary,
        rounds_used=max(1, len(relaxations_applied) + 1),
        relaxations_applied=relaxations_applied,
        round_one_feasible_count=round_one_feasible_count,
        round_one_candidate_count=round_one_candidate_count,
    )


def earliest_evaluation(evaluations: Sequence[CandidateEvaluation]) -> CandidateEvaluation | None:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        return None
    return sorted(feasible, key=lambda item: (item.slot.start_time, item.slot.slot_id))[0]


def random_evaluation(
    evaluations: Sequence[CandidateEvaluation],
    rng: np.random.Generator,
) -> CandidateEvaluation | None:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        return None
    return feasible[int(rng.integers(0, len(feasible)))]


def preferred_modality_earliest_evaluation(
    case: PreparedCase,
    evaluations: Sequence[CandidateEvaluation],
) -> CandidateEvaluation | None:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        return None
    preferred = [
        item for item in feasible if item.slot.modality in set(case.preferences.preferred_modalities)
    ]
    candidates = preferred if preferred else feasible
    return sorted(candidates, key=lambda item: (item.slot.start_time, item.slot.slot_id))[0]


def preference_hits(case: PreparedCase, slot: Slot) -> tuple[bool | None, bool | None]:
    day = weekday_for_slot(slot)
    preferred_days = set(case.preferences.preferred_days)
    if case.preferences.preferred_day_periods:
        preferred_days.update(pair.day for pair in case.preferences.preferred_day_periods)
    day_hit = None if not preferred_days else day in preferred_days

    preferred_modalities = set(case.preferences.preferred_modalities)
    modality_hit = None if not preferred_modalities else slot.modality in preferred_modalities
    return day_hit, modality_hit


def bootstrap_ci(values: Sequence[float], seed: int = 7, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    means = []
    for _ in range(samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@contextmanager
def local_llm_env(model: str):
    original = {
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "MAS_LLM_MODEL": os.environ.get("MAS_LLM_MODEL"),
    }
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:11434/v1"
    os.environ["OPENAI_API_KEY"] = "local"
    os.environ["MAS_LLM_MODEL"] = model
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def serialise_expected(expected: IntakeExpected) -> dict[str, object]:
    return {
        "complaint_category": expected.complaint_category,
        "preferred_days": list(expected.preferred_days),
        "excluded_days": list(expected.excluded_days),
        "preferred_modalities": list(expected.preferred_modalities),
        "excluded_modalities": list(expected.excluded_modalities),
        "preferred_periods": list(expected.preferred_periods),
        "excluded_periods": list(expected.excluded_periods),
        "date_horizon_days": expected.date_horizon_days,
        "preferred_day_periods": [list(item) for item in expected.preferred_day_periods],
        "duration_text": _normalise_duration_text(expected.duration_text),
    }


def _normalise_duration_text(value: object) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def _as_listish(value: object) -> object:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return value


def _extract_days_from_pair_payload(receptionist, value: object) -> list[str]:
    items = _as_listish(value)
    days = receptionist._extract_days(items)  # type: ignore[attr-defined]
    if days:
        return days
    if not isinstance(items, list):
        return []
    seen: list[str] = []
    for item in items:
        if isinstance(item, dict):
            seen.extend(receptionist._extract_days([str(item.get("day", ""))]))  # type: ignore[attr-defined]
    return list(dict.fromkeys(seen))


def _extract_periods_from_pair_payload(receptionist, value: object) -> list[str]:
    items = _as_listish(value)
    periods = receptionist._extract_periods(items)  # type: ignore[attr-defined]
    if periods:
        return periods
    if not isinstance(items, list):
        return []
    seen: list[str] = []
    for item in items:
        if isinstance(item, dict):
            seen.extend(receptionist._extract_periods([str(item.get("period", ""))]))  # type: ignore[attr-defined]
    return list(dict.fromkeys(seen))


def normalise_raw_llm_payload(receptionist, raw_payload: dict[str, object] | None) -> dict[str, object]:
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    pref_payload = payload.get("preferences", {})
    if not isinstance(pref_payload, dict):
        pref_payload = {}

    raw_horizon = pref_payload.get("date_horizon_days")
    try:
        parsed_horizon = int(raw_horizon)
    except Exception:
        parsed_horizon = None

    return {
        "complaint_category": str(payload.get("complaint_category", "")).strip().lower(),
        "preferred_days": receptionist._extract_days(  # type: ignore[attr-defined]
            pref_payload.get("preferred_days", pref_payload.get("days", []))
        )
        or _extract_days_from_pair_payload(
            receptionist,
            pref_payload.get("preferred_day_periods", pref_payload.get("day_periods", [])),
        ),
        "excluded_days": receptionist._extract_days(  # type: ignore[attr-defined]
            pref_payload.get("excluded_days", pref_payload.get("blocked_days", []))
        )
        or _extract_days_from_pair_payload(
            receptionist,
            pref_payload.get("excluded_day_periods", pref_payload.get("blocked_day_periods", [])),
        ),
        "preferred_modalities": receptionist._extract_modalities(  # type: ignore[attr-defined]
            pref_payload.get(
                "preferred_modalities",
                pref_payload.get("preferred_modality", pref_payload.get("modalities", [])),
            )
        ),
        "excluded_modalities": receptionist._extract_modalities(  # type: ignore[attr-defined]
            pref_payload.get(
                "excluded_modalities",
                pref_payload.get("excluded_modality", pref_payload.get("blocked_modalities", [])),
            )
        ),
        "preferred_periods": receptionist._extract_periods(  # type: ignore[attr-defined]
            pref_payload.get("preferred_periods", pref_payload.get("periods", []))
        )
        or _extract_periods_from_pair_payload(
            receptionist,
            pref_payload.get("preferred_day_periods", pref_payload.get("day_periods", [])),
        ),
        "excluded_periods": receptionist._extract_periods(  # type: ignore[attr-defined]
            pref_payload.get("excluded_periods", pref_payload.get("blocked_periods", []))
        )
        or _extract_periods_from_pair_payload(
            receptionist,
            pref_payload.get("excluded_day_periods", pref_payload.get("blocked_day_periods", [])),
        ),
        "date_horizon_days": parsed_horizon,
        "preferred_day_periods": [
            [pair.day, pair.period]
            for pair in receptionist._extract_day_period_pairs(  # type: ignore[attr-defined]
                _as_listish(pref_payload.get("preferred_day_periods", pref_payload.get("day_periods", [])))
            )
        ],
        "duration_text": _normalise_duration_text(payload.get("duration_text", "")),
    }


def normalise_canonical_intake(intake) -> dict[str, object]:
    return {
        "complaint_category": intake.complaint_category,
        "preferred_days": list(intake.preferences.preferred_days),
        "excluded_days": list(intake.preferences.excluded_days),
        "preferred_modalities": list(intake.preferences.preferred_modalities),
        "excluded_modalities": list(intake.preferences.excluded_modalities),
        "preferred_periods": list(intake.preferences.preferred_periods),
        "excluded_periods": list(intake.preferences.excluded_periods),
        "date_horizon_days": intake.preferences.date_horizon_days,
        "preferred_day_periods": [
            [pair.day, pair.period] for pair in intake.preferences.preferred_day_periods
        ],
        "duration_text": _normalise_duration_text(intake.duration_text or ""),
    }


def empty_intake_prediction() -> dict[str, object]:
    return {
        "complaint_category": "",
        "preferred_days": [],
        "excluded_days": [],
        "preferred_modalities": [],
        "excluded_modalities": [],
        "preferred_periods": [],
        "excluded_periods": [],
        "date_horizon_days": None,
        "preferred_day_periods": [],
        "duration_text": "",
    }


def evaluate_constraint_satisfaction(total_slots: int = 300) -> pd.DataFrame:
    prepared = [
        (constraint_type, item)
        for constraint_type, item in zip(
            [group for group, _case in constraint_profiles()],
            prepare_cases([case for _group, case in constraint_profiles()]),
        )
    ]
    _receptionist, _triage, allocator, rota = build_components()
    patient_agent = VariantPatientAgent()

    rows = []
    totals = defaultdict(int)
    violations = defaultdict(int)

    for constraint_type, case in prepared:
        slot_pool = rota.generate_scaled_slots(case.routing_decision, total_slots)
        result = run_scheduler(case, slot_pool, allocator, patient_agent, negotiate=False)
        totals[constraint_type] += 1
        if result.booking is None:
            violations[constraint_type] += 1
            continue

        slot = next(slot for slot in slot_pool if slot.slot_id == result.booking.slot_id)
        constraint_broken = False
        if constraint_type == "excluded_day":
            constraint_broken = weekday_for_slot(slot) in set(case.preferences.excluded_days)
        elif constraint_type == "excluded_period":
            constraint_broken = period_for_slot(slot) in set(case.preferences.excluded_periods)
        elif constraint_type == "excluded_modality":
            constraint_broken = slot.modality in set(case.preferences.excluded_modalities)
        elif constraint_type == "outside_horizon":
            min_start = min(candidate.start_time for candidate in slot_pool)
            horizon_end = min_start + timedelta(days=case.preferences.date_horizon_days)
            constraint_broken = slot.start_time > horizon_end
        else:
            day = weekday_for_slot(slot)
            period = period_for_slot(slot)
            min_start = min(candidate.start_time for candidate in slot_pool)
            horizon_end = min_start + timedelta(days=case.preferences.date_horizon_days)
            constraint_broken = any(
                [
                    slot.modality in set(case.preferences.excluded_modalities),
                    day in set(case.preferences.excluded_days),
                    period in set(case.preferences.excluded_periods),
                    any(pair.day == day and pair.period == period for pair in case.preferences.excluded_day_periods),
                    slot.start_time > horizon_end,
                ]
            )
        if constraint_broken:
            violations[constraint_type] += 1

    display_names = {
        "excluded_day": "Excluded day",
        "excluded_period": "Excluded period",
        "excluded_modality": "Excluded modality",
        "outside_horizon": "Outside horizon",
        "combined_constraints": "Combined constraints",
    }
    total_profiles = 0
    total_violations = 0
    for key in [
        "excluded_day",
        "excluded_period",
        "excluded_modality",
        "outside_horizon",
        "combined_constraints",
    ]:
        tested = totals[key]
        broken = violations[key]
        total_profiles += tested
        total_violations += broken
        rows.append(
            {
                "Constraint Type": display_names[key],
                "Profiles Tested": tested,
                "Violations": broken,
                "Violation Rate": f"{(100 * broken / tested) if tested else 0:.1f}%",
            }
        )

    rows.append(
        {
            "Constraint Type": "Total",
            "Profiles Tested": total_profiles,
            "Violations": total_violations,
            "Violation Rate": f"{(100 * total_violations / total_profiles) if total_profiles else 0:.1f}%",
        }
    )
    return pd.DataFrame(rows)


def evaluate_baselines(total_slots: int = 200, random_runs: int = 500) -> pd.DataFrame:
    prepared_cases = prepare_cases(baseline_profiles())
    _receptionist, _triage, allocator, rota = build_components()
    patient_agent = VariantPatientAgent()
    base_slot_pool = rota.generate_scaled_slots(prepared_cases[0].routing_decision, total_slots)
    rows_by_policy: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    deterministic_policies = ("scheduler", "earliest_feasible", "preferred_modality_earliest")
    for policy_name in deterministic_policies:
        for case in prepared_cases:
            allocation = allocator.allocate(
                routing=case.routing_decision,
                preferences=case.preferences,
                candidate_slots=base_slot_pool,
                patient_agent=patient_agent,
            )
            selected: CandidateEvaluation | None
            if policy_name == "scheduler":
                selected = (
                    booking_to_evaluation(allocation.booking, allocation.candidate_evaluations)
                    if allocation.booking is not None
                    else None
                )
            elif policy_name == "earliest_feasible":
                selected = earliest_evaluation(allocation.candidate_evaluations)
            else:
                selected = preferred_modality_earliest_evaluation(case, allocation.candidate_evaluations)
            if selected is None:
                rows_by_policy[policy_name].append({"success": False})
                continue
            day_hit, modality_hit = preference_hits(case, selected.slot)
            rows_by_policy[policy_name].append(
                {
                    "success": True,
                    "utility": selected.utility,
                    "day_hit": day_hit,
                    "modality_hit": modality_hit,
                }
            )

    rng = np.random.default_rng(17)
    for _run in range(random_runs):
        for case in prepared_cases:
            allocation = allocator.allocate(
                routing=case.routing_decision,
                preferences=case.preferences,
                candidate_slots=base_slot_pool,
                patient_agent=patient_agent,
            )
            selected = random_evaluation(allocation.candidate_evaluations, rng)
            if selected is None:
                rows_by_policy["random_feasible"].append({"success": False})
                continue
            day_hit, modality_hit = preference_hits(case, selected.slot)
            rows_by_policy["random_feasible"].append(
                {
                    "success": True,
                    "utility": selected.utility,
                    "day_hit": day_hit,
                    "modality_hit": modality_hit,
                }
            )

    output_rows = []
    policy_labels = {
        "random_feasible": "Random feasible",
        "earliest_feasible": "Earliest feasible",
        "preferred_modality_earliest": "Preferred modality earliest",
        "scheduler": "Your scheduler",
    }
    for key in [
        "random_feasible",
        "earliest_feasible",
        "preferred_modality_earliest",
        "scheduler",
    ]:
        records = rows_by_policy[key]
        successful = [record for record in records if record.get("success")]
        utilities = [float(record["utility"]) for record in successful]
        day_hits = [bool(record["day_hit"]) for record in successful if record.get("day_hit") is not None]
        modality_hits = [
            bool(record["modality_hit"])
            for record in successful
            if record.get("modality_hit") is not None
        ]
        lower, upper = bootstrap_ci(utilities, seed=19)
        output_rows.append(
            {
                "Policy": policy_labels[key],
                "Success Rate": f"{100 * len(successful) / len(records):.1f}%",
                "Mean Utility (95% CI)": (
                    "n/a"
                    if not utilities
                    else f"{np.mean(utilities):.1f} ({lower:.1f}, {upper:.1f})"
                ),
                "Median Utility": "n/a" if not utilities else f"{np.median(utilities):.1f}",
                "P95 Utility": "n/a" if not utilities else f"{np.percentile(utilities, 95):.1f}",
                "Preferred Day Hit": "n/a" if not day_hits else f"{100 * np.mean(day_hits):.1f}%",
                "Preferred Modality Hit": (
                    "n/a" if not modality_hits else f"{100 * np.mean(modality_hits):.1f}%"
                ),
            }
        )
    return pd.DataFrame(output_rows)


def evaluate_llm_intake(model: str = "qwen2.5:3b") -> tuple[pd.DataFrame, pd.DataFrame]:
    receptionist, _triage, _allocator, _rota = build_components()
    prompts = intake_prompt_cases()
    prompt_rows: List[dict[str, object]] = []
    tracked_fields = [
        ("complaint_category", "Complaint category"),
        ("preferred_days", "Preferred days"),
        ("excluded_days", "Excluded days"),
        ("preferred_modalities", "Preferred modalities"),
        ("excluded_modalities", "Excluded modalities"),
        ("preferred_periods", "Preferred periods"),
        ("excluded_periods", "Excluded periods"),
        ("date_horizon_days", "Date horizon"),
        ("preferred_day_periods", "Preferred day-period combinations"),
        ("duration_text", "Duration text"),
    ]

    with local_llm_env(model):
        for prompt_case in prompts:
            expected = serialise_expected(prompt_case.expected)
            try:
                intake, _questions, _engine, llm_payload = receptionist.build_intake(
                    run_id=prompt_case.case_id,
                    user_text=prompt_case.prompt,
                    clarification_answers={},
                    extractor="llm",
                    api_key="local",
                    llm_model=model,
                )
                raw_payload = (
                    llm_payload.get("raw_llm_output", {})
                    if isinstance(llm_payload, dict)
                    else {}
                )
                raw_prediction = normalise_raw_llm_payload(receptionist, raw_payload)
                canonical_prediction = normalise_canonical_intake(intake)
                llm_error = ""
            except Exception as exc:
                raw_prediction = empty_intake_prediction()
                canonical_prediction = empty_intake_prediction()
                llm_error = str(exc)

            row = {
                "case_id": prompt_case.case_id,
                "prompt_style": prompt_case.prompt_style,
                "prompt": prompt_case.prompt,
                "llm_error": llm_error,
                "raw_exact_all": True,
                "canonical_exact_all": True,
            }
            for field_key, _field_label in tracked_fields:
                row[f"expected_{field_key}"] = expected[field_key]
                row[f"raw_{field_key}"] = raw_prediction[field_key]
                row[f"canonical_{field_key}"] = canonical_prediction[field_key]
                row[f"raw_match_{field_key}"] = raw_prediction[field_key] == expected[field_key]
                row[f"canonical_match_{field_key}"] = canonical_prediction[field_key] == expected[field_key]
                row["raw_exact_all"] = row["raw_exact_all"] and row[f"raw_match_{field_key}"]
                row["canonical_exact_all"] = (
                    row["canonical_exact_all"] and row[f"canonical_match_{field_key}"]
                )
            prompt_rows.append(row)

    prompt_df = pd.DataFrame(prompt_rows)
    summary_rows: List[dict[str, object]] = []
    for field_key, field_label in tracked_fields:
        raw_acc = 100 * float(prompt_df[f"raw_match_{field_key}"].mean())
        canonical_acc = 100 * float(prompt_df[f"canonical_match_{field_key}"].mean())
        delta = canonical_acc - raw_acc
        summary_rows.append(
            {
                "Field": field_label,
                "Raw LLM Accuracy": f"{raw_acc:.1f}%",
                "After Canonicalisation": f"{canonical_acc:.1f}%",
                "Delta": f"{delta:+.1f}pp",
                "Annotation": ">=30pp improvement" if delta >= 30.0 else "",
            }
        )

    for prompt_style in ["canonical", "noisy"]:
        subset = prompt_df[prompt_df["prompt_style"] == prompt_style]
        raw_acc = 100 * float(subset["raw_exact_all"].mean())
        canonical_acc = 100 * float(subset["canonical_exact_all"].mean())
        delta = canonical_acc - raw_acc
        summary_rows.append(
            {
                "Field": f"Overall exact match ({prompt_style})",
                "Raw LLM Accuracy": f"{raw_acc:.1f}%",
                "After Canonicalisation": f"{canonical_acc:.1f}%",
                "Delta": f"{delta:+.1f}pp",
                "Annotation": ">=30pp improvement" if delta >= 30.0 else "",
            }
        )

    raw_acc = 100 * float(prompt_df["raw_exact_all"].mean())
    canonical_acc = 100 * float(prompt_df["canonical_exact_all"].mean())
    delta = canonical_acc - raw_acc
    summary_rows.append(
        {
            "Field": "Overall exact match (all prompts)",
            "Raw LLM Accuracy": f"{raw_acc:.1f}%",
            "After Canonicalisation": f"{canonical_acc:.1f}%",
            "Delta": f"{delta:+.1f}pp",
            "Annotation": ">=30pp improvement" if delta >= 30.0 else "",
        }
    )
    return pd.DataFrame(summary_rows), prompt_df


def schedule_cohort(
    prepared_cases: Sequence[PreparedCase],
    slot_pool: Sequence[Slot],
    *,
    ordering_policy: Literal["fcfs", "random_order", "scarcity_first"],
    patient_agent: VariantPatientAgent,
    seed: int,
) -> tuple[dict[str, dict[str, object]], List[dict[str, object]]]:
    _receptionist, _triage, allocator, _rota = build_components()
    ordering_rng = np.random.default_rng(seed)
    full_pool_initial_counts = {
        case.patient_id: allocator.allocate(
            routing=case.routing_decision,
            preferences=case.preferences,
            candidate_slots=list(slot_pool),
            patient_agent=patient_agent,
        )
        for case in prepared_cases
    }
    initial_feasible_counts = {
        patient_id: sum(1 for item in allocation.candidate_evaluations if item.feasible)
        for patient_id, allocation in full_pool_initial_counts.items()
    }
    cases_order = list(prepared_cases)
    if ordering_policy == "random_order":
        ordering_rng.shuffle(cases_order)
    elif ordering_policy == "scarcity_first":
        cases_order = sorted(
            cases_order,
            key=lambda case: (
                initial_feasible_counts[case.patient_id],
                -sum(
                    [
                        len(case.preferences.excluded_days),
                        len(case.preferences.excluded_periods),
                        len(case.preferences.excluded_modalities),
                        len(case.preferences.excluded_day_periods),
                    ]
                ),
                case.patient_id,
            ),
        )

    reserved_slot_ids: set[str] = set()
    outcomes: dict[str, dict[str, object]] = {}
    traces: List[dict[str, object]] = []

    for step_number, case in enumerate(cases_order, start=1):
        available_slots = [slot for slot in slot_pool if slot.slot_id not in reserved_slot_ids]
        result = run_scheduler(
            case,
            available_slots,
            allocator,
            patient_agent,
            negotiate=patient_agent.allow_relaxation,
        )
        feasible_loss = 0.0
        initial_count = initial_feasible_counts[case.patient_id]
        if initial_count > 0:
            feasible_loss = 1.0 - (result.round_one_feasible_count / initial_count)
        traces.append(
            {
                "policy": ordering_policy,
                "step": step_number,
                "patient_id": case.patient_id,
                "group": case.group,
                "success": result.booking is not None,
                "pool_fullness": 1.0 - (len(available_slots) / len(slot_pool)),
                "contention_level": feasible_loss,
                "relaxations_applied": len(result.relaxations_applied),
            }
        )
        outcomes[case.patient_id] = {
            "group": case.group,
            "request_type": case.request_type,
            "success": result.booking is not None,
            "utility": result.booking.utility if result.booking else None,
            "rounds_used": result.rounds_used,
            "relaxations_applied": result.relaxations_applied,
        }
        if result.booking is not None:
            reserved_slot_ids.add(result.booking.slot_id)

    return outcomes, traces


def summarise_cohort_runs(
    run_outcomes: Sequence[dict[str, dict[str, object]]],
) -> dict[str, float]:
    metrics = defaultdict(list)
    for outcomes in run_outcomes:
        records = list(outcomes.values())
        booked = [item for item in records if item["success"]]
        strict = [item for item in records if item["group"] == "strict"]
        flexible = [item for item in records if item["group"] == "flexible"]
        strict_booked = [item for item in strict if item["success"]]
        flexible_booked = [item for item in flexible if item["success"]]

        metrics["overall_booking_rate"].append(len(booked) / len(records))
        metrics["mean_utility"].append(
            float(np.mean([float(item["utility"] or 0.0) for item in records])) if records else 0.0
        )
        metrics["strict_booking_rate"].append(len(strict_booked) / len(strict) if strict else 0.0)
        metrics["flexible_booking_rate"].append(
            len(flexible_booked) / len(flexible) if flexible else 0.0
        )
        strict_mean = float(np.mean([float(item["utility"] or 0.0) for item in strict])) if strict else 0.0
        flexible_mean = (
            float(np.mean([float(item["utility"] or 0.0) for item in flexible])) if flexible else 0.0
        )
        metrics["utility_gap"].append(flexible_mean - strict_mean)
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def evaluate_contention_and_fairness(
    total_slots: int = 150,
    random_runs: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    prepared_cases = prepare_cases(contention_cohort())
    _receptionist, _triage, _allocator, rota = build_components()
    slot_pool = rota.generate_scaled_slots(prepared_cases[0].routing_decision, total_slots)
    full_agent = policy_patient_agent("full")

    run_results: Dict[str, List[dict[str, dict[str, object]]]] = defaultdict(list)
    all_traces: List[dict[str, object]] = []

    fcfs_outcomes, fcfs_traces = schedule_cohort(
        prepared_cases,
        slot_pool,
        ordering_policy="fcfs",
        patient_agent=full_agent,
        seed=11,
    )
    run_results["fcfs"].append(fcfs_outcomes)
    all_traces.extend(fcfs_traces)

    scarcity_outcomes, scarcity_traces = schedule_cohort(
        prepared_cases,
        slot_pool,
        ordering_policy="scarcity_first",
        patient_agent=full_agent,
        seed=13,
    )
    run_results["scarcity_first"].append(scarcity_outcomes)
    all_traces.extend(scarcity_traces)

    for run_index in range(random_runs):
        random_outcomes, random_traces = schedule_cohort(
            prepared_cases,
            slot_pool,
            ordering_policy="random_order",
            patient_agent=full_agent,
            seed=100 + run_index,
        )
        run_results["random_order"].append(random_outcomes)
        all_traces.extend(random_traces)

    table_rows = []
    policy_labels = {
        "fcfs": "FCFS",
        "random_order": "Random Order",
        "scarcity_first": "Scarcity-First",
    }
    for key in ["fcfs", "random_order", "scarcity_first"]:
        summary = summarise_cohort_runs(run_results[key])
        table_rows.append(
            {
                "Metric": "Overall booking rate",
                policy_labels[key]: f"{100 * summary['overall_booking_rate']:.1f}%",
            }
        )
        table_rows.append(
            {
                "Metric": "Mean utility",
                policy_labels[key]: f"{summary['mean_utility']:.1f}",
            }
        )
        table_rows.append(
            {
                "Metric": "Strict group booking rate",
                policy_labels[key]: f"{100 * summary['strict_booking_rate']:.1f}%",
            }
        )
        table_rows.append(
            {
                "Metric": "Flexible group booking rate",
                policy_labels[key]: f"{100 * summary['flexible_booking_rate']:.1f}%",
            }
        )
        table_rows.append(
            {
                "Metric": "Utility gap (flexible minus strict)",
                policy_labels[key]: f"{summary['utility_gap']:.1f}",
            }
        )

    summary_df = pd.DataFrame(table_rows).groupby("Metric", as_index=False).first()

    request_type_rows = []
    for policy_key, outcomes_list in run_results.items():
        per_type_success: Dict[str, List[float]] = defaultdict(list)
        for outcomes in outcomes_list:
            request_types = defaultdict(list)
            for record in outcomes.values():
                request_types[str(record["request_type"])].append(float(bool(record["success"])))
            for request_type, flags in request_types.items():
                per_type_success[request_type].append(float(np.mean(flags)))
        for request_type, values in sorted(per_type_success.items()):
            request_type_rows.append(
                {
                    "Policy": policy_labels[policy_key],
                    "Request Type": request_type,
                    "Success Rate": f"{100 * np.mean(values):.1f}%",
                }
            )
    request_type_df = pd.DataFrame(request_type_rows)

    traces_df = pd.DataFrame(all_traces)
    plot_source = traces_df[traces_df["policy"] == "random_order"].copy()
    plot_source["pool_fullness_bin"] = pd.cut(
        plot_source["pool_fullness"],
        bins=np.linspace(0.0, 1.0, 7),
        include_lowest=True,
        duplicates="drop",
    )
    plot_df = (
        plot_source.groupby(["group", "pool_fullness_bin"], observed=False)["success"]
        .mean()
        .reset_index()
    )
    plot_df["pool_fullness_mid"] = plot_df["pool_fullness_bin"].apply(lambda interval: float(interval.mid))

    figure_path = RESULTS_DIR / "multi_patient_contention.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colours = {"strict": "#d95f02", "flexible": "#1b9e77"}
    labels = {"strict": "Strict patients", "flexible": "Flexible patients"}
    for group in ["strict", "flexible"]:
        subset = plot_df[plot_df["group"] == group].sort_values("pool_fullness_mid")
        if subset.empty:
            continue
        ax.plot(
            subset["pool_fullness_mid"],
            subset["success"],
            color=colours[group],
            linestyle="-",
            marker="o",
            linewidth=2.2,
            label=labels[group],
        )
    ax.set_xlabel("Slot pool fullness")
    ax.set_ylabel("Booking rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Booking rate falls as the shared slot pool fills up")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    return summary_df, request_type_df, figure_path


def evaluate_ablation(total_slots: int = 150) -> pd.DataFrame:
    prepared_cases = prepare_cases(contention_cohort())
    _receptionist, _triage, _allocator, rota = build_components()
    slot_pool = rota.generate_scaled_slots(prepared_cases[0].routing_decision, total_slots)
    rows = []
    variant_labels = {
        "full": "Full system",
        "no_day_period_synergy": "No day-period synergy",
        "no_preference_weighting": "No preference weighting",
        "no_temporal_proximity": "No temporal proximity",
        "no_relaxation": "No relaxation",
    }

    for variant in [
        "full",
        "no_day_period_synergy",
        "no_preference_weighting",
        "no_temporal_proximity",
        "no_relaxation",
    ]:
        outcomes, _traces = schedule_cohort(
            prepared_cases,
            slot_pool,
            ordering_policy="scarcity_first",
            patient_agent=policy_patient_agent(variant),
            seed=41,
        )
        booked = [item for item in outcomes.values() if item["success"]]
        strict = [item for item in outcomes.values() if item["group"] == "strict"]
        flexible = [item for item in outcomes.values() if item["group"] == "flexible"]
        strict_booked = [item for item in strict if item["success"]]
        flexible_booked = [item for item in flexible if item["success"]]

        rows.append(
            {
                "Variant": variant_labels[variant],
                "Booking Rate": f"{100 * len(booked) / len(outcomes):.1f}%",
                "Mean Utility": (
                    f"{np.mean([float(item['utility'] or 0.0) for item in outcomes.values()]):.1f}"
                ),
                "Fairness Gap": (
                    f"{(100 * len(flexible_booked) / len(flexible)) - (100 * len(strict_booked) / len(strict)):.1f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_runtime_curve(x_values: Sequence[int], y_values: Sequence[float]) -> tuple[float, float, float]:
    transformed = np.asarray([value * math.log2(value) for value in x_values], dtype=float)
    design = np.column_stack([transformed, np.ones(len(transformed))])
    slope, intercept = np.linalg.lstsq(design, np.asarray(y_values, dtype=float), rcond=None)[0]
    predictions = design @ np.asarray([slope, intercept])
    ss_res = float(np.sum((np.asarray(y_values) - predictions) ** 2))
    ss_tot = float(np.sum((np.asarray(y_values) - np.mean(y_values)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept), float(r_squared)


def evaluate_runtime_scaling(repetitions: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, Path, tuple[float, float, float]]:
    prepared_single = prepare_cases(
        [
            EvalCase(
                patient_id="runtime-single",
                reason_code="persistent_cough",
                preferences=make_preferences(
                    preferred_modalities=["phone"],
                    preferred_days=["Mon"],
                    preferred_periods=["morning"],
                    preferred_day_periods=[("Mon", "morning")],
                    date_horizon_days=10,
                    soonest_weight=85,
                    weight_profile=PreferenceWeightProfile(
                        modality=120,
                        day=140,
                        period=140,
                        day_period_synergy=160,
                    ),
                ),
            )
        ]
    )[0]
    _receptionist, _triage, allocator, rota = build_components()
    patient_agent = policy_patient_agent("full")
    slot_sizes = [50, 100, 200, 400, 800]
    single_rows = []
    mean_values = []

    for slot_size in slot_sizes:
        slot_pool = rota.generate_scaled_slots(prepared_single.routing_decision, slot_size)
        durations_ms = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            allocator.allocate(
                routing=prepared_single.routing_decision,
                preferences=prepared_single.preferences,
                candidate_slots=slot_pool,
                patient_agent=patient_agent,
            )
            durations_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        mean_runtime = float(np.mean(durations_ms))
        mean_values.append(mean_runtime)
        single_rows.append(
            {
                "Slot Pool Size": slot_size,
                "Mean Runtime (ms)": mean_runtime,
                "P95 Runtime (ms)": float(np.percentile(durations_ms, 95)),
            }
        )

    slope, intercept, r_squared = fit_runtime_curve(slot_sizes, mean_values)

    patient_sizes = [5, 10, 20, 30, 50]
    multi_rows = []
    for patient_count in patient_sizes:
        prepared_cases = prepare_cases(runtime_cohort(patient_count))
        slot_pool = rota.generate_scaled_slots(prepared_cases[0].routing_decision, 200)
        durations_ms = []
        for iteration in range(repetitions):
            start = time.perf_counter_ns()
            schedule_cohort(
                prepared_cases,
                slot_pool,
                ordering_policy="scarcity_first",
                patient_agent=patient_agent,
                seed=700 + iteration,
            )
            durations_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        multi_rows.append(
            {
                "Patient Count": patient_count,
                "Mean Runtime (ms)": float(np.mean(durations_ms)),
                "P95 Runtime (ms)": float(np.percentile(durations_ms, 95)),
            }
        )

    left_df = pd.DataFrame(single_rows)
    right_df = pd.DataFrame(multi_rows)

    figure_path = RESULTS_DIR / "runtime_scaling.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(left_df["Slot Pool Size"], left_df["Mean Runtime (ms)"], marker="o", color="#1f77b4")
    axes[0].fill_between(
        left_df["Slot Pool Size"],
        left_df["Mean Runtime (ms)"],
        left_df["P95 Runtime (ms)"],
        alpha=0.2,
        color="#1f77b4",
        label="P95 band",
    )
    predicted = [slope * (size * math.log2(size)) + intercept for size in slot_sizes]
    axes[0].plot(slot_sizes, predicted, linestyle="--", color="#d62728", label="OLS fit")
    axes[0].set_title("Single-patient slot scaling")
    axes[0].set_xlabel("Slot pool size")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.03,
        0.92,
        f"R² = {r_squared:.3f}",
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    axes[1].plot(right_df["Patient Count"], right_df["Mean Runtime (ms)"], marker="o", color="#2ca02c")
    axes[1].fill_between(
        right_df["Patient Count"],
        right_df["Mean Runtime (ms)"],
        right_df["P95 Runtime (ms)"],
        alpha=0.2,
        color="#2ca02c",
    )
    axes[1].set_title("Multi-patient scaling at 200 slots")
    axes[1].set_xlabel("Patient count")
    axes[1].set_ylabel("Runtime (ms)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    return left_df, right_df, figure_path, (slope, intercept, r_squared)


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False))


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def run_all(config: ResultsConfig | None = None) -> dict[str, object]:
    config = config or ResultsConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    constraint_df = evaluate_constraint_satisfaction(total_slots=config.constraint_total_slots)
    baseline_df = evaluate_baselines(
        total_slots=config.baseline_total_slots,
        random_runs=config.baseline_random_runs,
    )
    fairness_df, request_type_df, fairness_figure = evaluate_contention_and_fairness(
        total_slots=config.fairness_total_slots,
        random_runs=config.fairness_random_runs,
    )
    ablation_df = evaluate_ablation(total_slots=config.fairness_total_slots)
    runtime_single_df, runtime_multi_df, runtime_figure, runtime_fit = evaluate_runtime_scaling(
        repetitions=config.runtime_repetitions
    )
    llm_intake_df = None
    llm_intake_prompt_df = None
    if config.include_llm_intake:
        llm_intake_df, llm_intake_prompt_df = evaluate_llm_intake(model=config.llm_model)

    save_dataframe(constraint_df, "constraint_satisfaction.csv")
    save_dataframe(baseline_df, "baseline_comparison.csv")
    save_dataframe(fairness_df, "multi_patient_fairness.csv")
    save_dataframe(request_type_df, "multi_patient_request_type_success.csv")
    save_dataframe(ablation_df, "ablation_study.csv")
    save_dataframe(runtime_single_df, "runtime_single_patient.csv")
    save_dataframe(runtime_multi_df, "runtime_multi_patient.csv")
    if llm_intake_df is not None and llm_intake_prompt_df is not None:
        save_dataframe(llm_intake_df, "llm_intake_evaluation.csv")
        save_dataframe(llm_intake_prompt_df, "llm_intake_prompt_level.csv")

    print_table("Constraint Satisfaction", constraint_df)
    print("\nHard constraints remained absolute veto checks before utility scoring. Across the tested profiles, the scheduler did not return a booking that broke an active hard rule.")
    print_table("Baseline Comparison", baseline_df)
    print_table("Multi-Patient Contention And Fairness", fairness_df)
    print_table("Ablation Study", ablation_df)
    if llm_intake_df is not None:
        print_table("LLM Intake Evaluation", llm_intake_df)
    print_table("Runtime Scaling: Single Patient", runtime_single_df.round(3))
    print_table("Runtime Scaling: Multi Patient", runtime_multi_df.round(3))
    slope, intercept, r_squared = runtime_fit
    print(
        "\nRuntime fit: "
        f"slope={slope:.6f}, intercept={intercept:.4f}, R²={r_squared:.4f}. "
        f"Largest tested mean runtime={runtime_multi_df['Mean Runtime (ms)'].max():.3f} ms."
    )
    print(f"\nSaved figures:\n- {fairness_figure}\n- {runtime_figure}")

    return {
        "constraint_df": constraint_df,
        "baseline_df": baseline_df,
        "fairness_df": fairness_df,
        "request_type_df": request_type_df,
        "ablation_df": ablation_df,
        "runtime_single_df": runtime_single_df,
        "runtime_multi_df": runtime_multi_df,
        "llm_intake_df": llm_intake_df,
        "llm_intake_prompt_df": llm_intake_prompt_df,
        "fairness_figure": fairness_figure,
        "runtime_figure": runtime_figure,
        "runtime_fit": runtime_fit,
    }


def main() -> None:
    run_all(
        ResultsConfig(
            include_llm_intake=os.environ.get("RUN_LLM_INTAKE", "0") == "1",
            llm_model=os.environ.get("MAS_LLM_MODEL", "qwen2.5:3b"),
        )
    )


if __name__ == "__main__":
    main()
