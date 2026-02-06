from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple

from nhs_demo.config import (
    DATE_RANGE_EXTENSION_DAYS,
    MISMATCH_PENALTIES,
    RELAXATION_ORDER,
    UTILITY_WEIGHTS,
)
from nhs_demo.schemas import (
    BlockerSummary,
    PatientPreferences,
    RelaxationQuestion,
    Slot,
)


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
        day = self._weekday_for_slot(slot.start_time)
        period = self._period_for_slot(slot.start_time)

        breakdown["preferred_modality"] = self._weighted_preference_component(
            slot.modality,
            preferences.preferred_modalities,
            UTILITY_WEIGHTS["preferred_modality"],
            preferences.weight_profile.modality,
            MISMATCH_PENALTIES["modality"],
        )
        breakdown["preferred_day"] = self._weighted_preference_component(
            day,
            preferences.preferred_days,
            UTILITY_WEIGHTS["preferred_day"],
            preferences.weight_profile.day,
            MISMATCH_PENALTIES["day"],
        )
        breakdown["preferred_period"] = self._weighted_preference_component(
            period,
            preferences.preferred_periods,
            UTILITY_WEIGHTS["preferred_period"],
            preferences.weight_profile.period,
            MISMATCH_PENALTIES["period"],
        )
        breakdown["preferred_day_period"] = self._day_period_component(
            day=day,
            period=period,
            preferences=preferences,
        )
        breakdown["adjacent_preferred_day"] = self._adjacent_preferred_day_component(
            day=day,
            preferences=preferences,
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
            return preferences.model_copy(update={"excluded_periods": [], "excluded_day_periods": []})
        if key == "relax_excluded_days":
            return preferences.model_copy(update={"excluded_days": [], "excluded_day_periods": []})
        if key == "relax_excluded_modalities":
            return preferences.model_copy(update={"excluded_modalities": []})
        if key == "extend_date_horizon":
            updated_horizon = min(preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS, 30)
            return preferences.model_copy(update={"date_horizon_days": updated_horizon})
        return preferences

    def _weighted_preference_component(
        self,
        value: str,
        preferred: List[str],
        base_weight: int,
        weight_profile_pct: int,
        mismatch_penalty_ratio: float,
    ) -> float:
        if not preferred:
            return 0.0
        scaled_weight = self._scaled_weight(base_weight, weight_profile_pct)
        if value in preferred:
            return round(scaled_weight, 3)
        return round(-scaled_weight * mismatch_penalty_ratio, 3)

    def _day_period_component(
        self,
        day: str,
        period: str,
        preferences: PatientPreferences,
    ) -> float:
        if not preferences.preferred_day_periods:
            return 0.0

        scaled_weight = self._scaled_weight(
            UTILITY_WEIGHTS["preferred_day_period"],
            preferences.weight_profile.day_period_synergy,
        )
        if any(pair.day == day and pair.period == period for pair in preferences.preferred_day_periods):
            return round(scaled_weight, 3)
        return round(-scaled_weight * 0.25, 3)

    def _adjacent_preferred_day_component(
        self,
        day: str,
        preferences: PatientPreferences,
    ) -> float:
        if not preferences.preferred_days:
            return 0.0
        if day in preferences.preferred_days:
            return 0.0

        day_idx = self.DAY_ORDER.index(day)
        preferred_idx = [self.DAY_ORDER.index(item) for item in preferences.preferred_days]
        if any(min(abs(day_idx - idx), 7 - abs(day_idx - idx)) == 1 for idx in preferred_idx):
            return round(
                self._scaled_weight(UTILITY_WEIGHTS["adjacent_preferred_day"], preferences.weight_profile.day),
                3,
            )
        return 0.0

    def _scaled_weight(self, base_weight: int, weight_profile_pct: int) -> float:
        return base_weight * (weight_profile_pct / 100.0)

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
        # Non-linear decay gives stronger preference to earlier appointments.
        score = max_weight * ((1 - position) ** 1.35) * (soonest_weight / 100.0)
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
            period_blockers = blockers.get("excluded_period", 0) + blockers.get("excluded_day_period", 0)
            if period_blockers and (
                preferences.excluded_periods or preferences.excluded_day_periods
            ) and preferences.flexibility.allow_time_relax:
                return RelaxationQuestion(
                    key=key,
                    prompt="Can we open up the time-of-day restrictions for the next search round?",
                )
            return None

        if key == "relax_excluded_days":
            day_blockers = blockers.get("excluded_day", 0) + blockers.get("excluded_day_period", 0)
            if day_blockers and (
                preferences.excluded_days or preferences.excluded_day_periods
            ) and preferences.flexibility.allow_time_relax:
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
