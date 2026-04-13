from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from nhs_demo.agents.master_allocator_agent import MasterAllocatorAgent
from nhs_demo.agents.receptionist import ReceptionistAgent
from nhs_demo.agents.rota_agent import RotaAgent
from nhs_demo.agents.triage_routing import TriageRoutingAgent
from nhs_demo.schemas import (
    DayPeriodPreference,
    FlexibilityOptions,
    FormReason,
    PatientPreferences,
    PreferenceWeightProfile,
    RoutingDecision,
)


FIXED_NOW = datetime(2026, 4, 6, 8, 0, 0)
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PERIODS = ["morning", "afternoon", "evening"]
MODALITIES = ["in_person", "phone", "video"]


@dataclass(frozen=True)
class EvalCase:
    patient_id: str
    reason_code: FormReason
    preferences: PatientPreferences
    group: str = "single"
    label: str = ""
    request_type: str = "gp_review"


@dataclass(frozen=True)
class PreparedCase:
    patient_id: str
    group: str
    label: str
    request_type: str
    preferences: PatientPreferences
    routing_decision: RoutingDecision


def fixed_now() -> datetime:
    return FIXED_NOW


def make_preferences(
    *,
    preferred_modalities: Sequence[str] | None = None,
    excluded_modalities: Sequence[str] | None = None,
    preferred_days: Sequence[str] | None = None,
    excluded_days: Sequence[str] | None = None,
    preferred_periods: Sequence[str] | None = None,
    excluded_periods: Sequence[str] | None = None,
    preferred_day_periods: Sequence[tuple[str, str]] | None = None,
    excluded_day_periods: Sequence[tuple[str, str]] | None = None,
    date_horizon_days: int = 10,
    soonest_weight: int = 60,
    weight_profile: PreferenceWeightProfile | None = None,
    flexibility: FlexibilityOptions | None = None,
) -> PatientPreferences:
    return PatientPreferences(
        preferred_modalities=list(preferred_modalities or []),
        excluded_modalities=list(excluded_modalities or []),
        preferred_days=list(preferred_days or []),
        excluded_days=list(excluded_days or []),
        preferred_periods=list(preferred_periods or []),
        excluded_periods=list(excluded_periods or []),
        preferred_day_periods=[
            DayPeriodPreference(day=day, period=period)
            for day, period in (preferred_day_periods or [])
        ],
        excluded_day_periods=[
            DayPeriodPreference(day=day, period=period)
            for day, period in (excluded_day_periods or [])
        ],
        date_horizon_days=date_horizon_days,
        soonest_weight=soonest_weight,
        weight_profile=weight_profile or PreferenceWeightProfile(),
        flexibility=flexibility or FlexibilityOptions(),
    )


def build_components() -> tuple[ReceptionistAgent, TriageRoutingAgent, MasterAllocatorAgent, RotaAgent]:
    return (
        ReceptionistAgent(now_provider=fixed_now),
        TriageRoutingAgent(),
        MasterAllocatorAgent(),
        RotaAgent(base_datetime=FIXED_NOW, now_provider=fixed_now),
    )


def prepare_cases(cases: Sequence[EvalCase]) -> List[PreparedCase]:
    receptionist, triage, _allocator, _rota = build_components()
    prepared: List[PreparedCase] = []
    for case in cases:
        intake, _questions, _engine, _llm = receptionist.build_form_intake(
            run_id=case.patient_id,
            reason_code=case.reason_code,
            preferences=case.preferences,
            extractor="rule",
        )
        prepared.append(
            PreparedCase(
                patient_id=case.patient_id,
                group=case.group,
                label=case.label,
                request_type=case.request_type,
                preferences=intake.preferences,
                routing_decision=triage.route(intake),
            )
        )
    return prepared


def constraint_profiles() -> list[tuple[str, EvalCase]]:
    profiles: list[tuple[str, EvalCase]] = []
    weekday_cycle = ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"]
    modality_cycle = ["phone", "video", "in_person", "phone", "video", "in_person", "phone", "video", "in_person", "phone", "video", "in_person"]
    period_cycle = ["morning", "afternoon", "morning", "afternoon", "morning", "afternoon", "morning", "afternoon", "morning", "afternoon", "morning", "afternoon"]

    for index in range(12):
        excluded_day = weekday_cycle[index]
        preferred_day = DAY_ORDER[(DAY_ORDER.index(excluded_day) + 2) % 5]
        profiles.append(
            (
                "excluded_day",
                EvalCase(
                    patient_id=f"constraint-day-{index:02d}",
                    reason_code="persistent_cough",
                    preferences=make_preferences(
                        preferred_modalities=[modality_cycle[index]],
                        preferred_days=[preferred_day],
                        excluded_days=[excluded_day],
                        preferred_periods=[period_cycle[index]],
                        preferred_day_periods=[(preferred_day, period_cycle[index])],
                        date_horizon_days=10,
                        soonest_weight=70,
                    ),
                    label="Excluded day",
                ),
            )
        )

    for index in range(12):
        excluded_period = PERIODS[index % 3]
        preferred_period = PERIODS[(index + 1) % 2]
        preferred_day = weekday_cycle[index]
        profiles.append(
            (
                "excluded_period",
                EvalCase(
                    patient_id=f"constraint-period-{index:02d}",
                    reason_code="persistent_cough",
                    preferences=make_preferences(
                        preferred_modalities=[modality_cycle[index]],
                        preferred_days=[preferred_day],
                        preferred_periods=[preferred_period],
                        excluded_periods=[excluded_period],
                        preferred_day_periods=[(preferred_day, preferred_period)],
                        date_horizon_days=10,
                        soonest_weight=60,
                    ),
                    label="Excluded period",
                ),
            )
        )

    for index in range(12):
        excluded_modality = MODALITIES[index % 3]
        preferred_modality = MODALITIES[(index + 1) % 3]
        profiles.append(
            (
                "excluded_modality",
                EvalCase(
                    patient_id=f"constraint-modality-{index:02d}",
                    reason_code="persistent_cough",
                    preferences=make_preferences(
                        preferred_modalities=[preferred_modality],
                        excluded_modalities=[excluded_modality],
                        preferred_days=[weekday_cycle[index]],
                        preferred_periods=[period_cycle[index]],
                        preferred_day_periods=[(weekday_cycle[index], period_cycle[index])],
                        date_horizon_days=10,
                        soonest_weight=65,
                    ),
                    label="Excluded modality",
                ),
            )
        )

    for index in range(12):
        profiles.append(
            (
                "outside_horizon",
                EvalCase(
                    patient_id=f"constraint-horizon-{index:02d}",
                    reason_code="persistent_cough",
                    preferences=make_preferences(
                        preferred_modalities=[modality_cycle[index]],
                        preferred_days=[weekday_cycle[index]],
                        preferred_periods=[period_cycle[index]],
                        preferred_day_periods=[(weekday_cycle[index], period_cycle[index])],
                        date_horizon_days=3,
                        soonest_weight=85,
                    ),
                    label="Outside horizon",
                ),
            )
        )

    combined_patterns = [
        ("Mon", "morning", "phone"),
        ("Tue", "morning", "video"),
        ("Wed", "afternoon", "in_person"),
        ("Thu", "morning", "phone"),
        ("Fri", "afternoon", "video"),
        ("Mon", "afternoon", "in_person"),
        ("Tue", "morning", "phone"),
        ("Wed", "afternoon", "video"),
        ("Thu", "afternoon", "in_person"),
        ("Fri", "morning", "phone"),
        ("Mon", "morning", "video"),
        ("Tue", "afternoon", "in_person"),
    ]
    for index, (preferred_day, preferred_period, preferred_modality) in enumerate(combined_patterns):
        excluded_day = DAY_ORDER[(DAY_ORDER.index(preferred_day) + 1) % 5]
        excluded_period = PERIODS[(PERIODS.index(preferred_period) + 1) % 3]
        excluded_modality = MODALITIES[(MODALITIES.index(preferred_modality) + 1) % 3]
        profiles.append(
            (
                "combined_constraints",
                EvalCase(
                    patient_id=f"constraint-combined-{index:02d}",
                    reason_code="persistent_cough",
                    preferences=make_preferences(
                        preferred_modalities=[preferred_modality],
                        excluded_modalities=[excluded_modality],
                        preferred_days=[preferred_day],
                        excluded_days=[excluded_day],
                        preferred_periods=[preferred_period],
                        excluded_periods=[excluded_period],
                        preferred_day_periods=[(preferred_day, preferred_period)],
                        excluded_day_periods=[(excluded_day, excluded_period)],
                        date_horizon_days=6,
                        soonest_weight=85,
                    ),
                    label="Combined constraints",
                ),
            )
        )

    return profiles


def baseline_profiles() -> List[EvalCase]:
    day_options = [["Mon"], ["Tue"], ["Wed"], ["Thu"], ["Fri"], ["Mon", "Wed"]]
    period_options = [["morning"], ["afternoon"], ["morning", "afternoon"], ["morning"]]
    cases: List[EvalCase] = []
    case_index = 0

    for modality in ["phone", "video", "in_person"]:
        for day_pref in day_options:
            for period_pref in period_options:
                excluded_day = DAY_ORDER[(DAY_ORDER.index(day_pref[0]) + 2) % 5] if case_index % 3 == 0 else None
                excluded_period = "evening" if case_index % 4 == 0 else None
                excluded_modality = MODALITIES[(MODALITIES.index(modality) + 1) % 3] if case_index % 5 == 0 else None
                day_period_pref = [(day_pref[0], period_pref[0])] if len(period_pref) == 1 else []
                cases.append(
                    EvalCase(
                        patient_id=f"baseline-{case_index:02d}",
                        reason_code="persistent_cough",
                        preferences=make_preferences(
                            preferred_modalities=[modality],
                            excluded_modalities=[excluded_modality] if excluded_modality else [],
                            preferred_days=day_pref,
                            excluded_days=[excluded_day] if excluded_day else [],
                            preferred_periods=period_pref,
                            excluded_periods=[excluded_period] if excluded_period else [],
                            preferred_day_periods=day_period_pref,
                            date_horizon_days=[7, 10, 14][case_index % 3],
                            soonest_weight=[35, 60, 90][case_index % 3],
                            weight_profile=PreferenceWeightProfile(
                                modality=130 if case_index % 2 == 0 else 100,
                                day=145 if len(day_pref) == 1 else 110,
                                period=130 if len(period_pref) == 1 else 100,
                                day_period_synergy=160 if day_period_pref else 100,
                            ),
                        ),
                        label="Baseline profile",
                    )
                )
                case_index += 1
    return cases


def contention_cohort() -> List[EvalCase]:
    strict_cases: List[EvalCase] = []
    flexible_cases: List[EvalCase] = []

    strict_templates = [
        ("Mon", "morning", "phone"),
        ("Tue", "morning", "phone"),
        ("Wed", "morning", "phone"),
        ("Mon", "morning", "phone"),
        ("Tue", "morning", "video"),
        ("Wed", "morning", "video"),
        ("Mon", "morning", "video"),
        ("Tue", "morning", "video"),
        ("Wed", "morning", "in_person"),
        ("Mon", "morning", "in_person"),
    ]
    relax_needed_templates = [
        ("Mon", "evening", "phone"),
        ("Tue", "evening", "video"),
        ("Wed", "evening", "phone"),
        ("Thu", "evening", "in_person"),
        ("Fri", "evening", "video"),
        ("Mon", "evening", "in_person"),
    ]
    flexible_templates = [
        ("Mon", "morning", "phone"),
        ("Tue", "morning", "phone"),
        ("Wed", "morning", "phone"),
        ("Mon", "morning", "phone"),
        ("Tue", "morning", "video"),
        ("Wed", "morning", "video"),
        ("Mon", "morning", "video"),
        ("Tue", "morning", "video"),
        ("Mon", "morning", "video"),
        ("Tue", "afternoon", "in_person"),
        ("Wed", "afternoon", "in_person"),
        ("Thu", "morning", "phone"),
        ("Fri", "morning", "video"),
        ("Thu", "afternoon", "phone"),
        ("Fri", "afternoon", "in_person"),
        ("Mon", "afternoon", "phone"),
    ]

    for index, (preferred_day, preferred_period, preferred_modality) in enumerate(strict_templates):
        strict_cases.append(
            EvalCase(
                patient_id=f"strict-{index:02d}",
                reason_code="persistent_cough",
                preferences=make_preferences(
                    preferred_modalities=[preferred_modality],
                    excluded_modalities=[item for item in MODALITIES if item != preferred_modality],
                    preferred_days=["Mon", "Tue", "Wed"],
                    excluded_days=["Thu", "Fri"],
                    preferred_periods=[preferred_period],
                    excluded_periods=["afternoon", "evening"],
                    preferred_day_periods=[(preferred_day, preferred_period)],
                    date_horizon_days=3,
                    soonest_weight=95,
                    weight_profile=PreferenceWeightProfile(
                        modality=150,
                        day=160,
                        period=155,
                        day_period_synergy=180,
                    ),
                    flexibility=FlexibilityOptions(
                        allow_time_relax=False,
                        allow_modality_relax=False,
                        allow_date_horizon_relax=False,
                    ),
                ),
                group="strict",
                label="strict_feasible",
                request_type="strict_gp",
            )
        )

    for index, (preferred_day, preferred_period, preferred_modality) in enumerate(
        relax_needed_templates,
        start=len(strict_cases),
    ):
        strict_cases.append(
            EvalCase(
                patient_id=f"strict-{index:02d}",
                reason_code="persistent_cough",
                preferences=make_preferences(
                    preferred_modalities=[preferred_modality],
                    excluded_modalities=[item for item in MODALITIES if item != preferred_modality],
                    preferred_days=[preferred_day],
                    excluded_days=[day for day in DAY_ORDER[:5] if day != preferred_day],
                    preferred_periods=[preferred_period],
                    excluded_periods=["morning", "afternoon"],
                    preferred_day_periods=[(preferred_day, preferred_period)],
                    date_horizon_days=4,
                    soonest_weight=90,
                    weight_profile=PreferenceWeightProfile(
                        modality=150,
                        day=160,
                        period=170,
                        day_period_synergy=190,
                    ),
                ),
                group="strict",
                label="strict_relax_needed",
                request_type="strict_gp_relax",
            )
        )

    for index, (preferred_day, preferred_period, preferred_modality) in enumerate(flexible_templates):
        flexible_cases.append(
            EvalCase(
                patient_id=f"flex-{index:02d}",
                reason_code="persistent_cough",
                preferences=make_preferences(
                    preferred_modalities=[preferred_modality],
                    preferred_days=[preferred_day, DAY_ORDER[(DAY_ORDER.index(preferred_day) + 1) % 5]],
                    preferred_periods=[preferred_period],
                    preferred_day_periods=[(preferred_day, preferred_period)],
                    date_horizon_days=10,
                    soonest_weight=65,
                    weight_profile=PreferenceWeightProfile(
                        modality=115,
                        day=120,
                        period=120,
                        day_period_synergy=135,
                    ),
                ),
                group="flexible",
                label="flexible",
                request_type="flexible_gp",
            )
        )

    return flexible_cases + strict_cases


def runtime_cohort(patient_count: int) -> List[EvalCase]:
    seed_cases = contention_cohort()
    return [
        EvalCase(
            patient_id=f"runtime-{index:03d}",
            reason_code=seed_cases[index % len(seed_cases)].reason_code,
            preferences=seed_cases[index % len(seed_cases)].preferences,
            group=seed_cases[index % len(seed_cases)].group,
            label=seed_cases[index % len(seed_cases)].label,
            request_type=seed_cases[index % len(seed_cases)].request_type,
        )
        for index in range(patient_count)
    ]
