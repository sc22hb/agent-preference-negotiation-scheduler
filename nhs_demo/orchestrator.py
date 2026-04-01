from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    IntakeSummary,
    MultiScheduleBid,
    MultiSchedulePatientRequest,
    MultiSchedulePatientResult,
    MultiScheduleRequest,
    MultiScheduleRoundResult,
    MultiScheduleResponse,
    PatientPreferences,
    RotaModeResponse,
    RouteRequest,
    RoutingDecision,
    RouteResponse,
    RunState,
    SafetyGateResult,
    Slot,
    SlotScore,
    ScheduleOfferRequest,
    ScheduleOfferResponse,
    ScheduleRelaxRequest,
    ScheduleRelaxResponse,
    SlotInventoryItem,
    SlotInventoryResponse,
)


@dataclass
class _BatchPatientContext:
    patient_id: str
    input_position: int
    run_id: str
    intake_summary: IntakeSummary
    routing_decision: RoutingDecision
    preferences: PatientPreferences
    initial_candidate_count: int
    initial_feasible_count: int


@dataclass
class _BatchBid:
    patient: _BatchPatientContext
    booking: object
    utility: float
    feasible_count: int
    candidate_count: int
    blocker_summary: object


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

    def _normalize_clarification_answers(self, answers: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in answers.items():
            cleaned_key = key.strip()
            cleaned_value = value.strip()
            if cleaned_key and cleaned_value:
                normalized[cleaned_key] = cleaned_value
        return normalized

    def intake(self, payload: IntakeRequest) -> IntakeResponse:
        run = self._create_run()
        safety = self.safety_agent.assess(payload.user_text)
        run.safety = safety
        run.clarification_answers = self._normalize_clarification_answers(payload.clarification_answers)

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
            clarification_answers=run.clarification_answers,
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

        merged_answers = dict(run.clarification_answers)
        merged_answers.update(self._normalize_clarification_answers(payload.clarification_answers))
        run.clarification_answers = merged_answers

        base_text = run.intake_summary.raw_text
        appended = " ".join(
            f"{key}: {value}" for key, value in sorted(merged_answers.items())
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
            clarification_answers=merged_answers,
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

    def create_structured_run(
        self,
        intake_summary: IntakeSummary,
        routing_decision: RoutingDecision | None = None,
        safety: SafetyGateResult | None = None,
    ) -> str:
        """Register a run from already-validated structured intake for experiments/tests."""
        run = self._create_run()
        run.safety = safety or SafetyGateResult(triggered=False)
        run.intake_summary = intake_summary.model_copy(update={"run_id": run.run_id})
        run.current_preferences = run.intake_summary.preferences
        if routing_decision is not None:
            run.routing_decision = routing_decision.model_copy(update={"run_id": run.run_id})
        self._save_run(run)
        return run.run_id

    def offer(self, payload: ScheduleOfferRequest) -> ScheduleOfferResponse:
        candidate_horizon = max(
            payload.preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS,
            DEFAULT_DATE_HORIZON_DAYS + DATE_RANGE_EXTENSION_DAYS,
        )
        candidate_slots = self.rota_agent.generate_slots(payload.routing_decision, candidate_horizon)
        return self.offer_with_candidate_slots(payload, candidate_slots)

    def offer_with_candidate_slots(
        self,
        payload: ScheduleOfferRequest,
        candidate_slots: List[Slot],
    ) -> ScheduleOfferResponse:
        """Use the normal offer logic against a caller-provided deterministic slot pool."""
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
            response = ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                booking=allocation.booking,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message="Booking found.",
            )
            self._rotate_rota_after_completed_run()
            return response

        if run.round_number >= MAX_NEGOTIATION_ROUNDS:
            run.status = "failed"
            run.failure_reason = "No feasible slot after maximum negotiation rounds"
            run.pending_relaxations = []
            run.round_audits.append(audit_round)
            self._save_run(run)
            response = ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message=run.failure_reason,
            )
            self._rotate_rota_after_completed_run()
            return response

        proposed_relaxations = self.patient_agent.propose_relaxations(
            routing=payload.routing_decision,
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
            response = ScheduleOfferResponse(
                run_id=run.run_id,
                status=run.status,
                round_number=run.round_number,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message=run.failure_reason,
            )
            self._rotate_rota_after_completed_run()
            return response

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

    def preview_offer(self, payload: ScheduleOfferRequest) -> ScheduleOfferResponse:
        run = self._get_run(payload.run_id)
        if run.safety.triggered:
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status="safety_escalation",
                round_number=run.round_number,
                slot_scores=[],
                message=run.safety.message or "Safety escalation",
            )

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
        slot_scores = self._build_slot_scores(
            evaluations=allocation.candidate_evaluations,
            allowed_modalities=payload.routing_decision.allowed_modalities,
        )

        if allocation.booking:
            return ScheduleOfferResponse(
                run_id=run.run_id,
                status="booked",
                round_number=run.round_number,
                booking=allocation.booking,
                slot_scores=slot_scores,
                blocker_summary=allocation.blocker_summary,
                message="Preview found a feasible booking.",
            )

        return ScheduleOfferResponse(
            run_id=run.run_id,
            status="open",
            round_number=run.round_number,
            slot_scores=slot_scores,
            blocker_summary=allocation.blocker_summary,
            message="Preview found no feasible booking with current constraints.",
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
                if question.key == "relax_excluded_days":
                    selected_days = payload.relaxation_selections.get(question.key, [])
                    if selected_days:
                        updated_preferences = self.patient_agent.apply_partial_day_relaxation(
                            updated_preferences,
                            selected_days,
                        )
                        normalized_days = [
                            day for day in self.patient_agent.DAY_ORDER if day in set(selected_days)
                        ]
                        if normalized_days:
                            run.relaxation_history.append(
                                f"relax_excluded_days_partial:{','.join(normalized_days)}"
                            )
                        else:
                            run.relaxation_history.append(question.key)
                    else:
                        updated_preferences = self.patient_agent.apply_relaxation(
                            updated_preferences,
                            question.key,
                        )
                        run.relaxation_history.append(question.key)
                else:
                    updated_preferences = self.patient_agent.apply_relaxation(
                        updated_preferences,
                        question.key,
                    )
                    run.relaxation_history.append(question.key)
                applied.append(question.key)
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

    def schedule_many(self, payload: MultiScheduleRequest) -> MultiScheduleResponse:
        batch = [
            self._prepare_batch_patient(item, input_position=index)
            for index, item in enumerate(payload.patients, start=1)
        ]
        reserved_slot_ids: set[str] = set()
        rounds = []
        pending = {patient.patient_id: patient for patient in batch}
        final_results: dict[str, MultiSchedulePatientResult] = {}
        round_number = 0

        while pending:
            round_number += 1
            round_bids: dict[str, List[_BatchBid]] = {}
            exhausted_patient_ids: List[str] = []

            for patient_id in list(pending.keys()):
                patient = pending[patient_id]
                candidate_slots = self._available_candidate_slots(
                    patient.routing_decision,
                    patient.preferences,
                    reserved_slot_ids,
                )
                allocation = self.allocator_agent.allocate(
                    routing=patient.routing_decision,
                    preferences=patient.preferences,
                    candidate_slots=candidate_slots,
                    patient_agent=self.patient_agent,
                )
                feasible_count = sum(1 for item in allocation.candidate_evaluations if item.feasible)

                if allocation.booking is None:
                    final_results[patient.patient_id] = MultiSchedulePatientResult(
                        patient_id=patient.patient_id,
                        run_id=patient.run_id,
                        input_position=patient.input_position,
                        assigned_round=None,
                        initial_candidate_count=patient.initial_candidate_count,
                        initial_feasible_count=patient.initial_feasible_count,
                        status="failed",
                        routing_decision=patient.routing_decision,
                        booking=None,
                        blocker_summary=allocation.blocker_summary,
                        candidate_count_last_round=len(candidate_slots),
                        feasible_count_last_round=feasible_count,
                    )
                    exhausted_patient_ids.append(patient.patient_id)
                    pending.pop(patient.patient_id, None)
                    continue

                bid = _BatchBid(
                    patient=patient,
                    booking=allocation.booking,
                    utility=allocation.booking.utility,
                    feasible_count=feasible_count,
                    candidate_count=len(candidate_slots),
                    blocker_summary=allocation.blocker_summary,
                )
                round_bids.setdefault(allocation.booking.slot_id, []).append(bid)

            if not round_bids:
                rounds.append(
                    MultiScheduleRoundResult(
                        round_number=round_number,
                        bids=[],
                        assigned_patient_ids=[],
                        assigned_slot_ids=[],
                        exhausted_patient_ids=sorted(exhausted_patient_ids),
                    )
                )
                break

            assigned_patient_ids: List[str] = []
            assigned_slot_ids: List[str] = []
            round_bid_records = []

            for slot_id in sorted(round_bids):
                bids = round_bids[slot_id]
                for bid in bids:
                    round_bid_records.append(
                        MultiScheduleBid(
                            patient_id=bid.patient.patient_id,
                            slot_id=slot_id,
                            utility=bid.utility,
                            feasible_count=bid.feasible_count,
                            input_position=bid.patient.input_position,
                        )
                    )

                winning_bid = self._select_batch_winner(
                    bids=bids,
                    conflict_policy=payload.conflict_policy,
                )
                reserved_slot_ids.add(slot_id)
                assigned_patient_ids.append(winning_bid.patient.patient_id)
                assigned_slot_ids.append(slot_id)
                final_results[winning_bid.patient.patient_id] = MultiSchedulePatientResult(
                    patient_id=winning_bid.patient.patient_id,
                    run_id=winning_bid.patient.run_id,
                    input_position=winning_bid.patient.input_position,
                    assigned_round=round_number,
                    initial_candidate_count=winning_bid.patient.initial_candidate_count,
                    initial_feasible_count=winning_bid.patient.initial_feasible_count,
                    status="booked",
                    routing_decision=winning_bid.patient.routing_decision,
                    booking=winning_bid.booking,
                    blocker_summary=winning_bid.blocker_summary,
                    candidate_count_last_round=winning_bid.candidate_count,
                    feasible_count_last_round=winning_bid.feasible_count,
                )
                pending.pop(winning_bid.patient.patient_id, None)

            rounds.append(
                MultiScheduleRoundResult(
                    round_number=round_number,
                    bids=sorted(
                        round_bid_records,
                        key=lambda bid: (bid.slot_id, bid.input_position, bid.patient_id),
                    ),
                    assigned_patient_ids=sorted(assigned_patient_ids),
                    assigned_slot_ids=sorted(assigned_slot_ids),
                    exhausted_patient_ids=sorted(exhausted_patient_ids),
                )
            )

        assignments = [
            final_results[patient.patient_id]
            for patient in sorted(batch, key=lambda item: item.input_position)
        ]
        booked_patients = sum(1 for item in assignments if item.booking is not None)
        response = MultiScheduleResponse(
            conflict_policy=payload.conflict_policy,
            rounds_run=round_number,
            total_patients=len(assignments),
            booked_patients=booked_patients,
            unbooked_patients=len(assignments) - booked_patients,
            assignments=assignments,
            reserved_slot_ids=sorted(reserved_slot_ids),
            rounds=rounds,
        )
        self._rotate_rota_after_completed_run()
        return response

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

    def slot_inventory(
        self,
        service_type: str | None = None,
        horizon_days: int | None = None,
    ) -> SlotInventoryResponse:
        inventory = self.rota_agent.list_inventory(
            service_type=service_type,
            horizon_days=horizon_days,
        )
        services: dict[str, List[SlotInventoryItem]] = {}
        for service, slots in inventory.items():
            services[service] = [
                SlotInventoryItem(
                    slot_id=slot.slot_id,
                    service_type=slot.service_type,  # type: ignore[arg-type]
                    clinician_id=slot.clinician_id,
                    site=slot.site,
                    modality=slot.modality,  # type: ignore[arg-type]
                    start_time=slot.start_time,
                    duration_minutes=slot.duration_minutes,
                )
                for slot in slots
            ]

        return SlotInventoryResponse(
            hospital=self.rota_agent.HOSPITAL_NAME,
            database_horizon_days=self.rota_agent.database_horizon_days,
            database_build_count=self.rota_agent.database_build_count,
            stochastic_mode=self.rota_agent.stochastic_mode,
            services=services,
        )

    def rota_mode(self) -> RotaModeResponse:
        return RotaModeResponse(
            stochastic_mode=self.rota_agent.stochastic_mode,
            database_build_count=self.rota_agent.database_build_count,
        )

    def set_rota_mode(self, stochastic_mode: bool) -> RotaModeResponse:
        self.rota_agent.set_stochastic_mode(stochastic_mode)
        return self.rota_mode()

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
        run.updated_at = datetime.now(timezone.utc)
        self._runs[run.run_id] = run

    def _rotate_rota_after_completed_run(self) -> None:
        self.rota_agent.rotate_inventory()

    def _prepare_batch_patient(
        self,
        payload: MultiSchedulePatientRequest,
        input_position: int,
    ) -> _BatchPatientContext:
        if payload.intake_summary is not None:
            run_id = self.create_structured_run(
                intake_summary=payload.intake_summary,
                routing_decision=payload.routing_decision,
            )
            intake_for_run = payload.intake_summary.model_copy(update={"run_id": run_id})
        else:
            if payload.reason_code is None or payload.preferences is None:
                raise ValueError(
                    "Each batch patient requires either intake_summary or reason_code plus preferences"
                )
            run = self._create_run()
            run.safety = self.safety_agent.assess(payload.reason_code.replace("_", " "))
            if run.safety.triggered:
                run.status = "safety_escalation"
                run.failure_reason = "Safety gate triggered"
                self._save_run(run)
                raise ValueError(f"Safety gate triggered for batch patient: {payload.patient_id}")

            intake_summary, _questions, _engine, _llm_response = self.receptionist_agent.build_form_intake(
                run_id=run.run_id,
                reason_code=payload.reason_code,
                preferences=payload.preferences,
                extractor="rule",
                api_key=None,
                llm_model=None,
            )
            run.intake_summary = intake_summary
            run.current_preferences = intake_summary.preferences
            self._save_run(run)
            run_id = run.run_id
            intake_for_run = intake_summary

        if payload.routing_decision is not None:
            routing_decision = payload.routing_decision.model_copy(update={"run_id": run_id})
        else:
            route_response = self.route(
                RouteRequest(run_id=run_id, intake_summary=intake_for_run)
            )
            routing_decision = route_response.routing_decision

        preferences = intake_for_run.preferences
        candidate_slots = self._available_candidate_slots(
            routing_decision,
            preferences,
            reserved_slot_ids=set(),
        )
        initial_feasible_count = self._feasible_candidate_count(
            routing_decision,
            preferences,
            candidate_slots,
        )
        return _BatchPatientContext(
            patient_id=payload.patient_id,
            input_position=input_position,
            run_id=run_id,
            intake_summary=intake_for_run,
            routing_decision=routing_decision,
            preferences=preferences,
            initial_candidate_count=len(candidate_slots),
            initial_feasible_count=initial_feasible_count,
        )

    def _select_batch_winner(
        self,
        bids: List[_BatchBid],
        conflict_policy: str,
    ) -> _BatchBid:
        if conflict_policy == "input_order":
            return sorted(
                bids,
                key=lambda bid: (
                    -bid.utility,
                    bid.patient.input_position,
                    bid.feasible_count,
                    bid.patient.patient_id,
                ),
            )[0]

        return sorted(
            bids,
            key=lambda bid: (
                -bid.utility,
                bid.feasible_count,
                bid.patient.input_position,
                bid.patient.patient_id,
            ),
        )[0]

    def _available_candidate_slots(
        self,
        routing_decision: RoutingDecision,
        preferences: PatientPreferences,
        reserved_slot_ids: set[str],
    ) -> List[Slot]:
        candidate_horizon = max(
            preferences.date_horizon_days + DATE_RANGE_EXTENSION_DAYS,
            DEFAULT_DATE_HORIZON_DAYS + DATE_RANGE_EXTENSION_DAYS,
        )
        candidate_slots = self.rota_agent.generate_slots(routing_decision, candidate_horizon)
        if not reserved_slot_ids:
            return candidate_slots
        return [slot for slot in candidate_slots if slot.slot_id not in reserved_slot_ids]

    def _feasible_candidate_count(
        self,
        routing_decision: RoutingDecision,
        preferences: PatientPreferences,
        candidate_slots: List[Slot],
    ) -> int:
        allocation = self.allocator_agent.allocate(
            routing=routing_decision,
            preferences=preferences,
            candidate_slots=candidate_slots,
            patient_agent=self.patient_agent,
        )
        return sum(1 for item in allocation.candidate_evaluations if item.feasible)

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
