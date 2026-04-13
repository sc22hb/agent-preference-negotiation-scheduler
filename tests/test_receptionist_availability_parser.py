from __future__ import annotations

from datetime import date, datetime

from nhs_demo.agents.rota_agent import RotaAgent
from nhs_demo.agents.receptionist import ReceptionistAgent
from nhs_demo.agents.master_allocator_agent import MasterAllocatorAgent
from nhs_demo.agents.patient_agent import PatientAgent
from nhs_demo.schemas import BlockerSummary, PatientPreferences, RoutingDecision


def test_busy_days_are_excluded_not_preferred() -> None:
    agent = ReceptionistAgent()
    preferences = agent._extract_preferences(
        "I have had a cough for 3 weeks and I am busy on Mondays and Fridays."
    )

    assert preferences.excluded_days == ["Mon", "Fri"]
    assert preferences.preferred_days == []


def test_day_period_phrases_do_not_cross_contaminate() -> None:
    agent = ReceptionistAgent()
    preferences = agent._extract_preferences(
        "I can do Monday mornings and Friday afternoons, but Tuesday mornings do not work."
    )

    preferred_pairs = {(item.day, item.period) for item in preferences.preferred_day_periods}
    excluded_pairs = {(item.day, item.period) for item in preferences.excluded_day_periods}

    assert preferred_pairs == {("Mon", "morning"), ("Fri", "afternoon")}
    assert excluded_pairs == {("Tue", "morning")}
    assert "Tue" not in preferences.excluded_days


def test_working_days_are_treated_as_unavailable() -> None:
    agent = ReceptionistAgent()
    preferences = agent._extract_preferences("I work Mondays and Fridays, and Tuesday works for me.")

    assert preferences.excluded_days == ["Mon", "Fri"]
    assert preferences.preferred_days == ["Tue"]


def test_mri_requests_skip_modality_clarification() -> None:
    agent = ReceptionistAgent()
    text = "I have been told to book an MRI scan using this and I want one next week in the morning."
    preferences = agent._extract_preferences(text)
    preferences = agent._apply_service_modality_policy(
        preferences,
        complaint_category="general_review",
        raw_text=text,
    )

    missing_fields = agent._missing_fields_from_preferences(
        preferences,
        complaint_category="general_review",
        raw_text=text,
    )
    filtered_missing_fields = agent._remove_context_specific_missing_fields(
        ["modality", "availability"],
        complaint_category="general_review",
        raw_text=text,
    )

    assert preferences.preferred_modalities == ["in_person"]
    assert preferences.excluded_modalities == ["phone", "video"]
    assert "modality" not in missing_fields
    assert filtered_missing_fields == ["availability"]


def test_rota_only_returns_future_slots_for_today() -> None:
    fixed_now = datetime(2026, 3, 18, 12, 0)
    agent = RotaAgent(now_provider=lambda: fixed_now)
    routing = RoutingDecision(
        run_id="run-1",
        service_type="GP",
        appointment_length_minutes=15,
        allowed_modalities=["in_person"],
        urgency_band="routine",
        confidence=1.0,
        rule_hit="test",
        explanation="test",
    )

    slots = agent.generate_slots(routing, horizon_days=1)
    same_day_slots = [slot for slot in slots if slot.start_time.date() == fixed_now.date()]

    assert all(slot.start_time >= fixed_now for slot in slots)
    assert [slot.start_time.hour for slot in same_day_slots] == [14, 16]


def test_next_week_means_not_before_next_monday() -> None:
    fixed_now = datetime(2026, 3, 18, 12, 0)
    receptionist = ReceptionistAgent(now_provider=lambda: fixed_now)
    text = "i need mri for next week cant do mornings"

    preferences = receptionist._extract_preferences(text)

    assert preferences.earliest_start_date == date(2026, 3, 23)

    rota = RotaAgent(now_provider=lambda: fixed_now)
    routing = RoutingDecision(
        run_id="run-2",
        service_type="GP",
        appointment_length_minutes=15,
        allowed_modalities=["in_person"],
        urgency_band="routine",
        confidence=1.0,
        rule_hit="test",
        explanation="test",
    )
    allocator = MasterAllocatorAgent()
    patient_agent = PatientAgent()

    result = allocator.allocate(
        routing=routing,
        preferences=preferences,
        candidate_slots=rota.generate_slots(routing, horizon_days=14),
        patient_agent=patient_agent,
    )

    assert result.booking is not None
    assert result.booking.start_time.date() >= date(2026, 3, 23)


def test_availability_followup_is_split_into_preferred_and_blocked() -> None:
    agent = ReceptionistAgent()
    questions = agent._clarification_questions_from_missing(["availability"])

    assert [question.question_id for question in questions] == [
        "availability",
        "availability_exclusion",
    ]
    assert questions[0].prompt == "Which days or times would you prefer?"
    assert questions[1].prompt == "Which days or times cannot you do?"


def test_in_person_only_routes_do_not_ask_for_modality_relaxation() -> None:
    patient_agent = PatientAgent()
    routing = RoutingDecision(
        run_id="run-3",
        service_type="GP",
        appointment_length_minutes=15,
        allowed_modalities=["in_person"],
        urgency_band="routine",
        confidence=1.0,
        rule_hit="test",
        explanation="test",
    )
    preferences = PatientPreferences(excluded_modalities=["phone", "video"])
    blockers = BlockerSummary(
        reason_counts={"excluded_modality": 2},
        ranked_reasons=["excluded_modality"],
    )

    relaxations = patient_agent.propose_relaxations(
        routing=routing,
        blocker_summary=blockers,
        preferences=preferences,
        applied_relaxations=[],
    )

    assert all(question.key != "relax_excluded_modalities" for question in relaxations)
