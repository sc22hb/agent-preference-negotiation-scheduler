from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Period = Literal["morning", "afternoon", "evening"]
Modality = Literal["in_person", "phone", "video"]
ServiceType = Literal["GP", "Nurse", "Pharmacist", "Admin"]
UrgencyBand = Literal["routine", "soon", "urgent-ish"]
RunStatus = Literal["open", "booked", "needs_relaxation", "failed", "safety_escalation"]
ExtractorMode = Literal["rule", "llm"]
CaptureMode = Literal["conversation", "form"]
FormReason = Literal[
    "repeat_prescription",
    "blood_test",
    "persistent_cough",
    "uti_symptoms",
    "sore_throat",
    "admin_request",
    "general_review",
]


class SafetyGateResult(BaseModel):
    triggered: bool
    message: Optional[str] = None
    matched_keywords: List[str] = Field(default_factory=list)


class FlexibilityOptions(BaseModel):
    allow_time_relax: bool = True
    allow_modality_relax: bool = True
    allow_date_horizon_relax: bool = True


class DayPeriodPreference(BaseModel):
    day: Weekday
    period: Period


class PreferenceWeightProfile(BaseModel):
    modality: int = Field(default=100, ge=0, le=200)
    day: int = Field(default=100, ge=0, le=200)
    period: int = Field(default=100, ge=0, le=200)
    day_period_synergy: int = Field(default=100, ge=0, le=200)


class PatientPreferences(BaseModel):
    preferred_modalities: List[Modality] = Field(default_factory=list)
    excluded_modalities: List[Modality] = Field(default_factory=list)
    preferred_days: List[Weekday] = Field(default_factory=list)
    excluded_days: List[Weekday] = Field(default_factory=list)
    preferred_periods: List[Period] = Field(default_factory=list)
    excluded_periods: List[Period] = Field(default_factory=list)
    preferred_day_periods: List[DayPeriodPreference] = Field(default_factory=list)
    excluded_day_periods: List[DayPeriodPreference] = Field(default_factory=list)
    date_horizon_days: int = Field(default=10, ge=1, le=30)
    soonest_weight: int = Field(default=60, ge=0, le=100)
    weight_profile: PreferenceWeightProfile = Field(default_factory=PreferenceWeightProfile)
    flexibility: FlexibilityOptions = Field(default_factory=FlexibilityOptions)


class ClarificationQuestion(BaseModel):
    question_id: str
    prompt: str


class IntakeSummary(BaseModel):
    run_id: str
    raw_text: str
    complaint_category: str
    duration_text: Optional[str] = None
    extracted_constraints_text: List[str] = Field(default_factory=list)
    preferences: PatientPreferences
    missing_fields: List[str] = Field(default_factory=list)


class IntakeRequest(BaseModel):
    user_text: str = Field(min_length=2)
    clarification_answers: Dict[str, str] = Field(default_factory=dict)
    extractor: ExtractorMode = "rule"
    api_key: Optional[str] = None
    llm_model: Optional[str] = None


class IntakeRefineRequest(BaseModel):
    run_id: str
    clarification_answers: Dict[str, str] = Field(default_factory=dict)
    extractor: ExtractorMode = "rule"
    api_key: Optional[str] = None
    llm_model: Optional[str] = None


class FormIntakeRequest(BaseModel):
    reason_code: FormReason
    preferences: PatientPreferences
    extractor: ExtractorMode = "rule"
    api_key: Optional[str] = None
    llm_model: Optional[str] = None


class IntakeResponse(BaseModel):
    run_id: str
    capture_mode: CaptureMode
    extraction_engine: str
    safety: SafetyGateResult
    intake_summary: Optional[IntakeSummary] = None
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    llm_response: Optional[Dict[str, Any]] = None


class RoutingDecision(BaseModel):
    run_id: str
    service_type: ServiceType
    appointment_length_minutes: int
    allowed_modalities: List[Modality]
    urgency_band: UrgencyBand
    confidence: float = Field(ge=0.0, le=1.0)
    rule_hit: str
    explanation: str


class RouteRequest(BaseModel):
    run_id: str
    intake_summary: IntakeSummary


class RouteResponse(BaseModel):
    run_id: str
    routing_decision: RoutingDecision


class Slot(BaseModel):
    slot_id: str
    service_type: ServiceType
    clinician_id: str
    site: str
    modality: Modality
    start_time: datetime
    duration_minutes: int


class Bid(BaseModel):
    slot_id: str
    utility: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


class CandidateEvaluation(BaseModel):
    slot: Slot
    feasible: bool
    veto_reasons: List[str] = Field(default_factory=list)
    utility: Optional[float] = None
    breakdown: Dict[str, float] = Field(default_factory=dict)


class BlockerSummary(BaseModel):
    reason_counts: Dict[str, int] = Field(default_factory=dict)
    ranked_reasons: List[str] = Field(default_factory=list)


class RelaxationQuestion(BaseModel):
    key: str
    prompt: str


class BookingOffer(BaseModel):
    slot_id: str
    service_type: ServiceType
    site: str
    modality: Modality
    clinician_id: str
    start_time: datetime
    duration_minutes: int
    utility: float
    tie_break_reason: str


class AuditRound(BaseModel):
    round_number: int
    candidate_count: int
    feasible_count: int
    blocker_summary: BlockerSummary
    candidate_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    selected_slot_id: Optional[str] = None
    tie_break_reason: Optional[str] = None
    relaxation_offered: List[str] = Field(default_factory=list)


class AuditLog(BaseModel):
    run_id: str
    status: RunStatus
    safety: SafetyGateResult
    intake_summary: Optional[IntakeSummary] = None
    routing_decision: Optional[RoutingDecision] = None
    round_audits: List[AuditRound] = Field(default_factory=list)
    relaxation_history: List[str] = Field(default_factory=list)
    final_booking: Optional[BookingOffer] = None
    failure_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ScheduleOfferRequest(BaseModel):
    run_id: str
    routing_decision: RoutingDecision
    preferences: PatientPreferences


class ScheduleOfferResponse(BaseModel):
    run_id: str
    status: RunStatus
    round_number: int
    booking: Optional[BookingOffer] = None
    slot_scores: List["SlotScore"] = Field(default_factory=list)
    blocker_summary: Optional[BlockerSummary] = None
    relaxation_questions: List[RelaxationQuestion] = Field(default_factory=list)
    message: str


class SlotScore(BaseModel):
    slot_number: int
    slot_id: str
    service_type: ServiceType
    clinician_id: str
    modality: Modality
    site: str
    start_time: datetime
    feasible: bool
    utility: Optional[float] = None
    breakdown: Dict[str, float] = Field(default_factory=dict)
    veto_reasons: List[str] = Field(default_factory=list)


class ScheduleRelaxRequest(BaseModel):
    run_id: str
    preferences: PatientPreferences
    answers: Dict[str, bool]


class ScheduleRelaxResponse(BaseModel):
    run_id: str
    round_number: int
    updated_preferences: PatientPreferences
    applied_relaxations: List[str] = Field(default_factory=list)
    rejected_relaxations: List[str] = Field(default_factory=list)
    message: str


class RunState(BaseModel):
    run_id: str
    status: RunStatus = "open"
    safety: SafetyGateResult = Field(default_factory=lambda: SafetyGateResult(triggered=False))
    intake_summary: Optional[IntakeSummary] = None
    routing_decision: Optional[RoutingDecision] = None
    current_preferences: Optional[PatientPreferences] = None
    round_number: int = 0
    pending_relaxations: List[RelaxationQuestion] = Field(default_factory=list)
    round_audits: List[AuditRound] = Field(default_factory=list)
    relaxation_history: List[str] = Field(default_factory=list)
    final_booking: Optional[BookingOffer] = None
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_safety_consistency(self) -> "RunState":
        if self.safety.triggered and self.status not in {"safety_escalation", "open"}:
            raise ValueError("safety-triggered runs must stay in escalation/open states")
        return self
