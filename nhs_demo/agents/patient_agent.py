from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple

from nhs_demo.config import DATE_RANGE_EXTENSION_DAYS, RELAXATION_ORDER, UTILITY_WEIGHTS
from nhs_demo.schemas import BlockerSummary, PatientPreferences, RelaxationQuestion, Slot


class PatientAgent:
    """Own patient utility scoring and minimal relaxation proposals."""

    DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def score_slot(
        self,
        slot: Slot,
        preferences: PatientPreferences,
        min_start: datetime,
        max_start: datetime,
    ) -> Tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}

        breakdown["preferred_modality"] = self._preference_component(
            slot.modality,
            preferences.preferred_modalities,
            UTILITY_WEIGHTS["preferred_modality"],
        )
        breakdown["preferred_day"] = self._preference_component(
            self._weekday_for_slot(slot.start_time),
            preferences.preferred_days,
            UTILITY_WEIGHTS["preferred_day"],
        )
        breakdown["preferred_period"] = self._preference_component(
            self._period_for_slot(slot.start_time),
            preferences.preferred_periods,
            UTILITY_WEIGHTS["preferred_period"],
        )
        breakdown["soonest"] = self._soonest_component(
            slot.start_time,
            min_start,
            max_start,
            UTILITY_WEIGHTS["soonest"],
            preferences.soonest_weight,
        )

        utility = round(sum(breakdown.values()), 3)
        return utility, breakdown

    def propose_relaxations(
        self,
        blocker_summary: BlockerSummary,
        preferences: PatientPreferences,
        applied_relaxations: Iterable[str],
    ) -> List[RelaxationQuestion]:
        applied: Set[str] = set(applied_relaxations)
        blockers = blocker_summary.reason_counts
        relaxations: List[RelaxationQuestion] = []

        for key in RELAXATION_ORDER:
            if key in applied:
                continue
            question = self._relaxation_question_if_applicable(key, blockers, preferences)
            if question:
                relaxations.append(question)
            if len(relaxations) == 2:
                break

        return relaxations

    def apply_relaxation(self, preferences: PatientPreferences, key: str) -> PatientPreferences:
        if key == "relax_excluded_periods":
            return preferences.model_copy(update={"excluded_periods": []})
        if key == "relax_excluded_days":
            return preferences.model_copy(update={"excluded_days": []})
        if key == "relax_excluded_modalities":
            return preferences.model_copy(update={"excluded_modalities": []})
        if key == "extend_date_horizon":
            updated_horizon = min(preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS, 30)
            return preferences.model_copy(update={"date_horizon_days": updated_horizon})
        return preferences

    def _preference_component(self, value: str, preferred: List[str], weight: int) -> float:
        if not preferred:
            return 0.0
        return float(weight if value in preferred else 0)

    def _soonest_component(
        self,
        slot_start: datetime,
        min_start: datetime,
        max_start: datetime,
        max_weight: int,
        soonest_weight: int,
    ) -> float:
        if max_start <= min_start:
            return float(max_weight)
        span = (max_start - min_start).total_seconds()
        position = (slot_start - min_start).total_seconds() / span
        score = max_weight * (1 - position) * (soonest_weight / 100.0)
        return round(score, 3)

    def _period_for_slot(self, dt: datetime) -> str:
        if dt.hour < 12:
            return "morning"
        if dt.hour < 17:
            return "afternoon"
        return "evening"

    def _weekday_for_slot(self, dt: datetime) -> str:
        return self.DAY_ORDER[dt.weekday()]

    def _relaxation_question_if_applicable(
        self,
        key: str,
        blockers: Dict[str, int],
        preferences: PatientPreferences,
    ) -> RelaxationQuestion | None:
        if key == "relax_excluded_periods":
            if blockers.get("excluded_period", 0) and preferences.excluded_periods and preferences.flexibility.allow_time_relax:
                return RelaxationQuestion(
                    key=key,
                    prompt="Can we open up the time-of-day restrictions for the next search round?",
                )
            return None

        if key == "relax_excluded_days":
            if blockers.get("excluded_day", 0) and preferences.excluded_days and preferences.flexibility.allow_time_relax:
                return RelaxationQuestion(
                    key=key,
                    prompt="Can we remove day exclusions and search all weekdays?",
                )
            return None

        if key == "relax_excluded_modalities":
            if blockers.get("excluded_modality", 0) and preferences.excluded_modalities and preferences.flexibility.allow_modality_relax:
                return RelaxationQuestion(
                    key=key,
                    prompt="Can we remove modality exclusions and use any allowed modality?",
                )
            return None

        if key == "extend_date_horizon":
            if blockers.get("outside_horizon", 0) and preferences.flexibility.allow_date_horizon_relax:
                return RelaxationQuestion(
                    key=key,
                    prompt="Can we extend the search window by a few extra days?",
                )
            return None

        return None
