from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

from nhs_demo.config import BLOCKER_SEVERITY
from nhs_demo.schemas import (
    BlockerSummary,
    BookingOffer,
    CandidateEvaluation,
    PatientPreferences,
    RoutingDecision,
    Slot,
)

from nhs_demo.agents.patient_agent import PatientAgent


@dataclass
class AllocationResult:
    booking: BookingOffer | None
    blocker_summary: BlockerSummary
    candidate_evaluations: List[CandidateEvaluation]
    tie_break_reason: str | None


class MasterAllocatorAgent:
    """Apply hard feasibility constraints and deterministically pick best feasible slot."""

    DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def allocate(
        self,
        routing: RoutingDecision,
        preferences: PatientPreferences,
        candidate_slots: List[Slot],
        patient_agent: PatientAgent,
    ) -> AllocationResult:
        if not candidate_slots:
            return AllocationResult(
                booking=None,
                blocker_summary=BlockerSummary(
                    reason_counts={"no_candidates": 1},
                    ranked_reasons=["no_candidates"],
                ),
                candidate_evaluations=[],
                tie_break_reason=None,
            )

        min_start = min(slot.start_time for slot in candidate_slots)
        max_start = max(slot.start_time for slot in candidate_slots)

        blocker_counts: Dict[str, int] = {}
        evaluations: List[CandidateEvaluation] = []
        feasible: List[Tuple[Slot, float, Dict[str, float]]] = []

        for slot in candidate_slots:
            veto_reasons = self._veto_reasons(slot, routing, preferences, min_start)
            if veto_reasons:
                for reason in veto_reasons:
                    blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
                evaluations.append(
                    CandidateEvaluation(slot=slot, feasible=False, veto_reasons=veto_reasons)
                )
                continue

            utility, breakdown = patient_agent.score_slot(slot, preferences, min_start, max_start)
            feasible.append((slot, utility, breakdown))
            evaluations.append(
                CandidateEvaluation(
                    slot=slot,
                    feasible=True,
                    utility=utility,
                    breakdown=breakdown,
                )
            )

        blocker_summary = self._build_blocker_summary(blocker_counts)

        if not feasible:
            return AllocationResult(
                booking=None,
                blocker_summary=blocker_summary,
                candidate_evaluations=evaluations,
                tie_break_reason=None,
            )

        sorted_feasible = sorted(
            feasible,
            key=lambda item: (
                -item[1],
                item[0].start_time,
                self._modality_rank(item[0].modality, routing.allowed_modalities),
                item[0].slot_id,
            ),
        )

        best_slot, best_utility, _ = sorted_feasible[0]
        tie_break_reason = (
            "max utility, then earliest start, then modality priority, then slot_id"
        )

        booking = BookingOffer(
            slot_id=best_slot.slot_id,
            service_type=best_slot.service_type,
            site=best_slot.site,
            modality=best_slot.modality,
            clinician_id=best_slot.clinician_id,
            start_time=best_slot.start_time,
            duration_minutes=best_slot.duration_minutes,
            utility=best_utility,
            tie_break_reason=tie_break_reason,
        )

        return AllocationResult(
            booking=booking,
            blocker_summary=blocker_summary,
            candidate_evaluations=evaluations,
            tie_break_reason=tie_break_reason,
        )

    def _veto_reasons(
        self,
        slot: Slot,
        routing: RoutingDecision,
        preferences: PatientPreferences,
        min_start,
    ) -> List[str]:
        reasons: List[str] = []

        if slot.modality not in routing.allowed_modalities:
            reasons.append("modality_not_allowed")

        if slot.modality in preferences.excluded_modalities:
            reasons.append("excluded_modality")

        day = self.DAY_ORDER[slot.start_time.weekday()]
        if day in preferences.excluded_days:
            reasons.append("excluded_day")

        period = self._period_for_slot(slot.start_time)
        if period in preferences.excluded_periods:
            reasons.append("excluded_period")

        horizon_end = min_start + timedelta(days=preferences.date_horizon_days)
        if slot.start_time > horizon_end:
            reasons.append("outside_horizon")

        return reasons

    def _build_blocker_summary(self, blocker_counts: Dict[str, int]) -> BlockerSummary:
        if not blocker_counts:
            return BlockerSummary(reason_counts={}, ranked_reasons=[])

        ranked = sorted(
            blocker_counts,
            key=lambda reason: (
                -blocker_counts[reason],
                -BLOCKER_SEVERITY.get(reason, 0),
                reason,
            ),
        )
        return BlockerSummary(reason_counts=blocker_counts, ranked_reasons=ranked)

    def _modality_rank(self, modality: str, allowed_modalities: List[str]) -> int:
        try:
            return allowed_modalities.index(modality)
        except ValueError:
            return len(allowed_modalities) + 1

    def _period_for_slot(self, dt) -> str:
        if dt.hour < 12:
            return "morning"
        if dt.hour < 17:
            return "afternoon"
        return "evening"
