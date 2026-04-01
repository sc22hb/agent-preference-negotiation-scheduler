from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from nhs_demo.agents.rota_agent import RotaAgent
from nhs_demo.api import create_app
from nhs_demo.orchestrator import DemoOrchestrator
from nhs_demo.schemas import (
    DayPeriodPreference,
    IntakeSummary,
    MultiSchedulePatientRequest,
    MultiScheduleRequest,
    PatientPreferences,
    PreferenceWeightProfile,
    RoutingDecision,
)


FIXED_NOW = datetime(2026, 2, 9, 8, 0)


def _orchestrator_with_fixed_rota() -> DemoOrchestrator:
    orchestrator = DemoOrchestrator()
    orchestrator.rota_agent = RotaAgent(
        base_datetime=FIXED_NOW.replace(hour=0, minute=0),
        now_provider=lambda: FIXED_NOW,
    )
    return orchestrator


def _intake(run_id: str, raw_text: str, complaint_category: str, preferences: PatientPreferences) -> IntakeSummary:
    return IntakeSummary(
        run_id=run_id,
        raw_text=raw_text,
        complaint_category=complaint_category,
        duration_text="Routine review",
        extracted_constraints_text=[],
        preferences=preferences,
        missing_fields=[],
    )


def _routing(run_id: str, service_type: str = "GP", complaint_rule: str = "gp_default") -> RoutingDecision:
    return RoutingDecision(
        run_id=run_id,
        service_type=service_type,
        appointment_length_minutes=15,
        allowed_modalities=["in_person", "phone", "video"],
        urgency_band="routine",
        confidence=1.0,
        rule_hit=complaint_rule,
        explanation="test routing",
    )


def test_schedule_many_reserves_unique_slots_across_patients() -> None:
    orchestrator = _orchestrator_with_fixed_rota()
    shared_preferences = PatientPreferences(
        preferred_modalities=["phone"],
        preferred_days=["Mon", "Wed"],
        preferred_periods=["morning"],
        date_horizon_days=10,
        soonest_weight=70,
    )

    response = orchestrator.schedule_many(
        MultiScheduleRequest(
            conflict_policy="input_order",
            patients=[
                MultiSchedulePatientRequest(
                    patient_id="patient-1",
                    intake_summary=_intake(
                        "patient-1",
                        "Routine medication review.",
                        "general_review",
                        shared_preferences,
                    ),
                    routing_decision=_routing("patient-1"),
                ),
                MultiSchedulePatientRequest(
                    patient_id="patient-2",
                    intake_summary=_intake(
                        "patient-2",
                        "Routine medication review.",
                        "general_review",
                        shared_preferences,
                    ),
                    routing_decision=_routing("patient-2"),
                ),
            ],
        )
    )

    booked_slots = [item.booking.slot_id for item in response.assignments if item.booking]
    assert len(booked_slots) == 2
    assert len(set(booked_slots)) == 2
    assert response.booked_patients == 2
    assert response.unbooked_patients == 0


def test_scarcity_first_prioritises_constrained_patient_for_best_shared_slot() -> None:
    strict_preferences = PatientPreferences(
        preferred_modalities=["video"],
        excluded_modalities=["in_person", "phone"],
        preferred_days=["Wed"],
        excluded_days=["Mon", "Tue", "Thu", "Fri"],
        preferred_periods=["morning"],
        excluded_periods=["afternoon", "evening"],
        preferred_day_periods=[DayPeriodPreference(day="Wed", period="morning")],
        date_horizon_days=10,
        soonest_weight=60,
        weight_profile=PreferenceWeightProfile(
            modality=200,
            day=200,
            period=200,
            day_period_synergy=200,
        ),
    )
    flexible_preferences = PatientPreferences(
        preferred_modalities=["video"],
        preferred_days=["Wed"],
        preferred_periods=["morning"],
        preferred_day_periods=[DayPeriodPreference(day="Wed", period="morning")],
        date_horizon_days=10,
        soonest_weight=60,
        weight_profile=PreferenceWeightProfile(
            modality=200,
            day=200,
            period=200,
            day_period_synergy=200,
        ),
    )

    patients = [
        MultiSchedulePatientRequest(
            patient_id="flexible",
            intake_summary=_intake(
                "flexible",
                "Routine GP review.",
                "general_review",
                flexible_preferences,
            ),
            routing_decision=_routing("flexible"),
        ),
        MultiSchedulePatientRequest(
            patient_id="strict",
            intake_summary=_intake(
                "strict",
                "Routine GP review.",
                "general_review",
                strict_preferences,
            ),
            routing_decision=_routing("strict"),
        ),
    ]

    input_order_response = _orchestrator_with_fixed_rota().schedule_many(
        MultiScheduleRequest(conflict_policy="input_order", patients=patients)
    )
    scarcity_first_response = _orchestrator_with_fixed_rota().schedule_many(
        MultiScheduleRequest(conflict_policy="scarcity_first", patients=patients)
    )

    input_strict = next(item for item in input_order_response.assignments if item.patient_id == "strict")
    scarcity_strict = next(item for item in scarcity_first_response.assignments if item.patient_id == "strict")
    assert input_strict.initial_feasible_count < next(
        item.initial_feasible_count for item in input_order_response.assignments if item.patient_id == "flexible"
    )
    assert input_strict.booking is not None
    assert scarcity_strict.booking is not None
    assert input_strict.booking.slot_id != scarcity_strict.booking.slot_id
    assert scarcity_strict.assigned_round == 1


def test_schedule_many_accepts_form_style_patient_payloads() -> None:
    orchestrator = _orchestrator_with_fixed_rota()
    response = orchestrator.schedule_many(
        MultiScheduleRequest(
            conflict_policy="scarcity_first",
            patients=[
                MultiSchedulePatientRequest(
                    patient_id="admin-case",
                    reason_code="admin_request",
                    preferences=PatientPreferences(
                        preferred_modalities=["phone", "video"],
                        preferred_periods=["afternoon"],
                        date_horizon_days=10,
                        soonest_weight=50,
                    ),
                ),
                MultiSchedulePatientRequest(
                    patient_id="review-case",
                    reason_code="general_review",
                    preferences=PatientPreferences(
                        preferred_modalities=["in_person", "phone"],
                        preferred_days=["Tue", "Thu"],
                        date_horizon_days=10,
                        soonest_weight=60,
                    ),
                ),
            ],
        )
    )

    assert response.total_patients == 2
    assert len(response.assignments) == 2
    assert all(item.routing_decision.service_type for item in response.assignments)


def test_multi_schedule_api_accepts_form_style_payloads() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/schedule/multi",
        json={
            "conflict_policy": "scarcity_first",
            "patients": [
                {
                    "patient_id": "api-admin",
                    "reason_code": "admin_request",
                    "preferences": {
                        "preferred_modalities": ["phone", "video"],
                        "preferred_periods": ["afternoon"],
                        "date_horizon_days": 10,
                        "soonest_weight": 50,
                    },
                },
                {
                    "patient_id": "api-blood",
                    "reason_code": "blood_test",
                    "preferences": {
                        "preferred_modalities": ["in_person"],
                        "excluded_modalities": ["phone", "video"],
                        "preferred_days": ["Tue"],
                        "date_horizon_days": 10,
                        "soonest_weight": 60,
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_patients"] == 2
    assert len(payload["assignments"]) == 2


def test_multi_schedule_request_rejects_duplicate_patient_ids() -> None:
    preferences = PatientPreferences()
    duplicate_patient = MultiSchedulePatientRequest(
        patient_id="dup",
        intake_summary=_intake("dup", "Routine review.", "general_review", preferences),
        routing_decision=_routing("dup"),
    )

    try:
        MultiScheduleRequest(
            patients=[duplicate_patient, duplicate_patient],
            conflict_policy="scarcity_first",
        )
    except ValueError as exc:
        assert "patient_id values must be unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate patient IDs to raise a validation error")


def test_stochastic_rota_rotates_inventory_after_each_completed_run() -> None:
    rota_agent = RotaAgent(
        base_datetime=FIXED_NOW.replace(hour=0, minute=0),
        now_provider=lambda: FIXED_NOW,
    )
    rota_agent.set_stochastic_mode(True)

    initial_inventory = rota_agent.list_inventory(service_type="GP", horizon_days=5)
    initial_slot_ids = [slot.slot_id for slot in initial_inventory["GP"]]
    initial_build_count = rota_agent.database_build_count

    rota_agent.rotate_inventory()

    rotated_inventory = rota_agent.list_inventory(service_type="GP", horizon_days=5)
    rotated_slot_ids = [slot.slot_id for slot in rotated_inventory["GP"]]

    assert rota_agent.stochastic_mode is True
    assert rota_agent.database_build_count == initial_build_count + 1
    assert initial_slot_ids != rotated_slot_ids


def test_rota_mode_api_toggles_stochastic_inventory() -> None:
    client = TestClient(create_app())

    enable_response = client.post("/api/rota/mode", json={"stochastic_mode": True})
    assert enable_response.status_code == 200
    enable_payload = enable_response.json()
    assert enable_payload["stochastic_mode"] is True

    inventory_response = client.get("/api/slots?horizon_days=5")
    assert inventory_response.status_code == 200
    inventory_payload = inventory_response.json()
    assert inventory_payload["stochastic_mode"] is True

    disable_response = client.post("/api/rota/mode", json={"stochastic_mode": False})
    assert disable_response.status_code == 200
    disable_payload = disable_response.json()
    assert disable_payload["stochastic_mode"] is False
