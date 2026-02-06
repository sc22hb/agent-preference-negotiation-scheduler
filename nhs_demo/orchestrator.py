from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

from nhs_demo.agents.master_allocator_agent import MasterAllocatorAgent
from nhs_demo.agents.patient_agent import PatientAgent
from nhs_demo.agents.receptionist import ReceptionistAgent
from nhs_demo.agents.rota_agent import RotaAgent
from nhs_demo.agents.safety_gate import SafetyGateAgent
from nhs_demo.agents.triage_routing import TriageRoutingAgent
from nhs_demo.config import DATE_RANGE_EXTENSION_DAYS, DEFAULT_DATE_HORIZON_DAYS, MAX_NEGOTIATION_ROUNDS
from nhs_demo.schemas import (
    AuditLog,
    AuditRound,
    CandidateEvaluation,
    FormIntakeRequest,
    IntakeRequest,
    IntakeRefineRequest,
    IntakeResponse,
    RouteRequest,
    RouteResponse,
    RunState,
    SlotScore,
    ScheduleOfferRequest,
    ScheduleOfferResponse,
    ScheduleRelaxRequest,
    ScheduleRelaxResponse,
)


class DemoOrchestrator:
    """Coordinate all agents for deterministic MAS scheduling runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self.safety_agent = SafetyGateAgent()
        self.receptionist_agent = ReceptionistAgent()
        self.triage_agent = TriageRoutingAgent()
        self.rota_agent = RotaAgent()
        self.patient_agent = PatientAgent()
        self.allocator_agent = MasterAllocatorAgent()

    def intake(self, payload: IntakeRequest) -> IntakeResponse:
        run = self._create_run()
        safety = self.safety_agent.assess(payload.user_text)
        run.safety = safety

        if safety.triggered:
            run.status = "safety_escalation"
            run.failure_reason = "Safety gate triggered"
            self._save_run(run)
            return IntakeResponse(
                run_id=run.run_id,
                capture_mode="conversation",
                extraction_engine="rule",
                safety=safety,
            )

        intake_summary, questions, engine, llm_response = self.receptionist_agent.build_intake(
            run_id=run.run_id,
            user_text=payload.user_text,
            clarification_answers=payload.clarification_answers,
            extractor=payload.extractor,
            api_key=payload.api_key,
            llm_model=payload.llm_model,
            preference_hint=None,
        )
        run.intake_summary = intake_summary
        run.current_preferences = intake_summary.preferences
        self._save_run(run)

        return IntakeResponse(
            run_id=run.run_id,
            capture_mode="conversation",
            extraction_engine=engine,
            safety=safety,
            intake_summary=intake_summary,
            clarification_questions=questions,
            llm_response=llm_response,
        )

    def refine_intake(self, payload: IntakeRefineRequest) -> IntakeResponse:
        run = self._get_run(payload.run_id)
        if run.intake_summary is None:
            raise ValueError("Cannot refine intake before initial intake exists")

        base_text = run.intake_summary.raw_text
        appended = " ".join(
            f"{key}: {value}" for key, value in sorted(payload.clarification_answers.items()) if value.strip()
        )
        safety_text = f"{base_text} {appended}".strip()
        safety = self.safety_agent.assess(safety_text)
        run.safety = safety

        if safety.triggered:
            run.status = "safety_escalation"
            run.failure_reason = "Safety gate triggered"
            self._save_run(run)
            return IntakeResponse(
                run_id=run.run_id,
                capture_mode="conversation",
                extraction_engine="rule",
                safety=safety,
            )

        intake_summary, questions, engine, llm_response = self.receptionist_agent.build_intake(
            run_id=run.run_id,
            user_text=base_text,
            clarification_answers=payload.clarification_answers,
            extractor=payload.extractor,
            api_key=payload.api_key,
            llm_model=payload.llm_model,
            preference_hint=run.current_preferences,
        )
        run.intake_summary = intake_summary
        run.current_preferences = intake_summary.preferences
        self._save_run(run)

        return IntakeResponse(
            run_id=run.run_id,
            capture_mode="conversation",
            extraction_engine=engine,
            safety=safety,
            intake_summary=intake_summary,
            clarification_questions=questions,
            llm_response=llm_response,
        )

    def intake_form(self, payload: FormIntakeRequest) -> IntakeResponse:
        run = self._create_run()
        safety = self.safety_agent.assess(payload.reason_code.replace("_", " "))
        run.safety = safety

        if safety.triggered:
            run.status = "safety_escalation"
            run.failure_reason = "Safety gate triggered"
            self._save_run(run)
            return IntakeResponse(
                run_id=run.run_id,
                capture_mode="form",
                extraction_engine="rule",
                safety=safety,
            )

        intake_summary, questions, engine, llm_response = self.receptionist_agent.build_form_intake(
            run_id=run.run_id,
            reason_code=payload.reason_code,
            preferences=payload.preferences,
            extractor=payload.extractor,
            api_key=payload.api_key,
            llm_model=payload.llm_model,
        )
        run.intake_summary = intake_summary
        run.current_preferences = intake_summary.preferences
        self._save_run(run)

        return IntakeResponse(
            run_id=run.run_id,
            capture_mode="form",
            extraction_engine=engine,
            safety=safety,
            intake_summary=intake_summary,
            clarification_questions=questions,
            llm_response=llm_response,
        )

    def route(self, payload: RouteRequest) -> RouteResponse:
        run = self._get_run(payload.run_id)
        if run.safety.triggered:
            raise ValueError("Cannot route a safety-escalated run")

        routing_decision = self.triage_agent.route(payload.intake_summary)
        run.intake_summary = payload.intake_summary
        run.routing_decision = routing_decision
        self._save_run(run)

        return RouteResponse(run_id=run.run_id, routing_decision=routing_decision)

    def offer(self, payload: ScheduleOfferRequest) -> ScheduleOfferResponse:
        run = self._get_run(payload.run_id)
        if run.safety.triggered:
            run.status = "safety_escalation"
            self._save_run(run)
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                slot_scores=[],
                message=run.safety.message or "Safety escalation",
            )

        run.routing_decision = payload.routing_decision
        run.current_preferences = payload.preferences
        run.round_number += 1

        candidate_horizon = max(
            payload.preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS,
            DEFAULT_DATE_HORIZON_DAYS + DATE_RANGE_EXTENSION_DAYS,
        )
        candidate_slots = self.rota_agent.generate_slots(payload.routing_decision, candidate_horizon)
        allocation = self.allocator_agent.allocate(
            routing=payload.routing_decision,
            preferences=payload.preferences,
            candidate_slots=candidate_slots,
            patient_agent=self.patient_agent,
        )

        feasible_count = sum(1 for eval_item in allocation.candidate_evaluations if eval_item.feasible)
        slot_scores = self._build_slot_scores(
            evaluations=allocation.candidate_evaluations,
            allowed_modalities=payload.routing_decision.allowed_modalities,
        )
        audit_round = AuditRound(
            round_number=run.round_number,
            candidate_count=len(allocation.candidate_evaluations),
            feasible_count=feasible_count,
            blocker_summary=allocation.blocker_summary,
            candidate_evaluations=allocation.candidate_evaluations,
            selected_slot_id=allocation.booking.slot_id if allocation.booking else None,
            tie_break_reason=allocation.tie_break_reason,
            relaxation_offered=[],
        )

        if allocation.booking:
            run.status = "booked"
            run.final_booking = allocation.booking
            run.pending_relaxations = []
            run.round_audits.append(audit_round)
            self._save_run(run)
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                booking=allocation.booking,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message="Booking found.",
            )

        if run.round_number >= MAX_NEGOTIATION_ROUNDS:
            run.status = "failed"
            run.failure_reason = "No feasible slot after maximum negotiation rounds"
            run.pending_relaxations = []
            run.round_audits.append(audit_round)
            self._save_run(run)
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message=run.failure_reason,
            )

        proposed_relaxations = self.patient_agent.propose_relaxations(
            blocker_summary=allocation.blocker_summary,
            preferences=payload.preferences,
            applied_relaxations=run.relaxation_history,
        )
        relaxation_questions = self.receptionist_agent.build_relaxation_questions(proposed_relaxations)
        audit_round.relaxation_offered = [question.key for question in relaxation_questions]
        run.round_audits.append(audit_round)

        if not relaxation_questions:
            run.status = "failed"
            run.failure_reason = "No additional safe relaxations available"
            run.pending_relaxations = []
            self._save_run(run)
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message=run.failure_reason,
            )

        run.status = "needs_relaxation"
        run.pending_relaxations = relaxation_questions
        self._save_run(run)

        return ScheduleOfferResponse(
            run_id=run.run_id,
            status=run.status,
            round_number=run.round_number,
            slot_scores=slot_scores,
            blocker_summary=allocation.blocker_summary,
            relaxation_questions=relaxation_questions,
            message="No slot yet. Relaxation approval required for next round.",
        )

    def relax(self, payload: ScheduleRelaxRequest) -> ScheduleRelaxResponse:
        run = self._get_run(payload.run_id)
        if run.status != "needs_relaxation":
            raise ValueError("Relaxation is only valid when run status is needs_relaxation")

        updated_preferences = payload.preferences
        applied: List[str] = []
        rejected: List[str] = []

        for question in run.pending_relaxations:
            approved = bool(payload.answers.get(question.key, False))
            if approved:
                updated_preferences = self.patient_agent.apply_relaxation(updated_preferences, question.key)
                applied.append(question.key)
                run.relaxation_history.append(question.key)
            else:
                rejected.append(question.key)

        run.current_preferences = updated_preferences
        run.pending_relaxations = []
        run.status = "open"
        self._save_run(run)

        message = "Relaxations applied. Retry /api/schedule/offer for the next round."
        if not applied:
            message = "No relaxations applied. Retry offer only if constraints remain acceptable."

        return ScheduleRelaxResponse(
            run_id=run.run_id,
            round_number=run.round_number,
            updated_preferences=updated_preferences,
            applied_relaxations=applied,
            rejected_relaxations=rejected,
            message=message,
        )

    def audit(self, run_id: str) -> AuditLog:
        run = self._get_run(run_id)
        return AuditLog(
            run_id=run.run_id,
            status=run.status,
            safety=run.safety,
            intake_summary=run.intake_summary,
            routing_decision=run.routing_decision,
            round_audits=run.round_audits,
            relaxation_history=run.relaxation_history,
            final_booking=run.final_booking,
            failure_reason=run.failure_reason,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )

    def get_run_state(self, run_id: str) -> RunState:
        return self._get_run(run_id)

    def _create_run(self) -> RunState:
        run_id = uuid4().hex[:12]
        run = RunState(run_id=run_id)
        self._runs[run_id] = run
        return run

    def _get_run(self, run_id: str) -> RunState:
        if run_id not in self._runs:
            raise KeyError(f"run_id not found: {run_id}")
        return self._runs[run_id]

    def _save_run(self, run: RunState) -> None:
        run.updated_at = datetime.utcnow()
        self._runs[run.run_id] = run

    def _build_slot_scores(
        self,
        evaluations: List[CandidateEvaluation],
        allowed_modalities: List[str],
    ) -> List[SlotScore]:
        def modality_rank(modality: str) -> int:
            try:
                return allowed_modalities.index(modality)
            except ValueError:
                return len(allowed_modalities) + 1

        def sort_key(item: CandidateEvaluation):
            utility = item.utility if item.utility is not None else float("-inf")
            return (
                0 if item.feasible else 1,
                -utility,
                item.slot.start_time,
                modality_rank(item.slot.modality),
                item.slot.slot_id,
            )

        ordered = sorted(evaluations, key=sort_key)
        scores: List[SlotScore] = []
        for idx, evaluation in enumerate(ordered, start=1):
            scores.append(
                SlotScore(
                    slot_number=idx,
                    slot_id=evaluation.slot.slot_id,
                    service_type=evaluation.slot.service_type,
                    clinician_id=evaluation.slot.clinician_id,
                    modality=evaluation.slot.modality,
                    site=evaluation.slot.site,
                    start_time=evaluation.slot.start_time,
                    feasible=evaluation.feasible,
                    utility=evaluation.utility,
                    breakdown=evaluation.breakdown,
                    veto_reasons=evaluation.veto_reasons,
                )
            )
        return scores
