from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on sys.path when run via streamlit.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Support running as a script or module.
try:
    from .conversation import generate_compromises
    from .explain import explain_assignment
    from .llm_extract import extract_request_from_text
    from .metrics import evaluate_run
    from .models import PatientRequest, PreferredTimeWindow, TimeWindow
    from .nhs_engine import negotiate, rank_slots
    from .nhs_slots import build_slot_inventory, BASE_DATE
    from .triage import triage_request
except ImportError:  # pragma: no cover
    from nhs_demo.conversation import generate_compromises
    from nhs_demo.explain import explain_assignment
    from nhs_demo.llm_extract import extract_request_from_text
    from nhs_demo.metrics import evaluate_run
    from nhs_demo.models import PatientRequest, PreferredTimeWindow, TimeWindow
    from nhs_demo.nhs_engine import negotiate, rank_slots
    from nhs_demo.nhs_slots import build_slot_inventory, BASE_DATE
    from nhs_demo.triage import triage_request


DAY_OFFSETS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}


def _make_time_window(day_offset: int, start_hour: int, end_hour: int) -> TimeWindow:
    base = BASE_DATE + timedelta(days=day_offset)
    return TimeWindow(
        start_time=base.replace(hour=start_hour, minute=0),
        end_time=base.replace(hour=end_hour, minute=0),
    )


def _synthetic_request(i: int) -> PatientRequest:
    profiles = [
        ("routine check", {"Mon": 70}, {"IN_PERSON": 80}, 60),
        ("blood test", {"Tue": 70}, {"IN_PERSON": 60}, 40),
        ("sore throat", {"Wed": 70}, {"PHONE": 60}, 70),
        ("persistent cough", {"Thu": 70}, {"VIDEO": 60}, 50),
        ("routine check", {"Fri": 70}, {"IN_PERSON": 80}, 30),
    ]
    free_text, days, modes, soonest = profiles[(i - 1) % len(profiles)]
    urgency, appt_type, role = triage_request(free_text)
    return PatientRequest(
        request_id=f"SIM-{i}",
        free_text_reason=free_text,
        urgency_band=urgency,
        required_appt_type=appt_type,
        must_be_role="GP",
        must_be_mode=None,
        preferred_days=days,
        preferred_modes=modes,
        preferred_time_windows=[],
        preferred_slot_ids=[],
        soonest_weight=soonest,
        consent_to_relax=True,
    )


def main() -> None:
    st.set_page_config(page_title="NHS Multi‑Agent Appointment Demo", layout="wide")
    st.title("NHS Multi‑Agent Appointment Booking Demo")
    st.caption("Deterministic scheduling; LLM only for extraction (stubbed).")

    slots = build_slot_inventory()

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio("Preference capture", ["Form", "Chatbot (stub)"])

    st.sidebar.header("Simulated Calendars")
    if "slot_occupancy" not in st.session_state:
        st.session_state.slot_occupancy = {s.slot_id: False for s in slots}

    def render_calendar(role: str) -> None:
        role_slots = [s for s in slots if s.clinician_role == role]
        calendar_rows = []
        for s in role_slots:
            calendar_rows.append(
                {
                    "Slot ID": s.slot_id,
                    "Day": s.start_time.strftime("%a"),
                    "Time": s.start_time.strftime("%H:%M"),
                    "Mode": s.mode,
                    "Occupied": st.session_state.slot_occupancy.get(s.slot_id, False),
                }
            )
        calendar_df = pd.DataFrame(calendar_rows)
        edited = st.sidebar.data_editor(
            calendar_df,
            hide_index=True,
            use_container_width=True,
            disabled=["Slot ID", "Day", "Time", "Mode"],
            key=f"calendar_{role}",
        )
        for _, row in edited.iterrows():
            st.session_state.slot_occupancy[row["Slot ID"]] = bool(row["Occupied"])

    calendar_role = st.sidebar.radio("Calendar view", ["GP", "NURSE", "PHARMACIST"], index=0)
    render_calendar(calendar_role)

    prebooked_slots = [
        slot_id for slot_id, occupied in st.session_state.slot_occupancy.items() if occupied
    ]

    st.sidebar.caption("Checked slots are treated as already booked.")
    add_simulated = st.sidebar.checkbox("Add simulated patients", value=False)
    sim_count = 0
    if add_simulated:
        sim_count = st.sidebar.slider("Number of simulated patients", 1, 10, 3)

    st.header("Booking Request")

    if input_mode == "Chatbot (stub)":
        free_text = st.text_area("Describe your issue")
        if st.button("Extract Preferences"):
            result = extract_request_from_text(free_text)
            if result.error:
                st.error(result.error)
                return
            request = result.request
        else:
            return
    else:
        reason_options = [
            "repeat prescription",
            "asthma worsening",
            "long-standing rash",
            "blood test",
            "routine check",
            "persistent cough",
            "sore throat",
            "UTI symptoms",
        ]
        free_text = st.selectbox("Reason for appointment", reason_options, index=4)
        urgency, appt_type, role = triage_request(free_text)
        st.write(f"Triage: {urgency}, {appt_type}, {role}")

        preferred_days_selection = st.multiselect(
            "Preferred day(s)",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            default=["Mon", "Tue"],
        )
        preferred_modes_selection = st.multiselect(
            "Preferred mode(s)",
            ["IN_PERSON", "PHONE", "VIDEO"],
            default=["IN_PERSON"],
        )
        time_window_choices = st.multiselect(
            "Preferred time window(s)",
            ["Early Morning", "Morning", "Midday", "Afternoon", "Late", "Evening"],
            default=["Morning"],
        )
        hard_days = st.multiselect(
            "Unavailable day(s) (hard constraint)",
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            default=[],
        )
        hard_time_windows = st.multiselect(
            "Unavailable time windows (hard constraint)",
            ["Early Morning", "Morning", "Midday", "Afternoon", "Late", "Evening"],
            default=[],
        )
        apply_unavailable_all_days = st.checkbox(
            "Apply unavailable time windows to all days",
            value=True,
        )
        must_be_mode = st.selectbox("Must be mode (hard constraint)", ["No constraint", "IN_PERSON", "PHONE", "VIDEO"], index=0)
        role_choice = st.selectbox(
            "Clinician role (hard constraint)",
            ["Auto (use triage)", "GP", "NURSE", "PHARMACIST"],
            index=0,
        )
        soonest_weight = st.slider("Preference for sooner appointments", 0, 100, 60)
        consent_to_relax = st.checkbox("Allow the system to suggest compromises", value=True)

        preferred_days = {day: 70 for day in preferred_days_selection}
        preferred_modes = {mode: 70 for mode in preferred_modes_selection}
        preferred_time_windows: List[PreferredTimeWindow] = []
        for choice in time_window_choices:
            start_hour, end_hour = {
                "Early Morning": (8, 9),
                "Morning": (9, 12),
                "Midday": (12, 14),
                "Afternoon": (14, 17),
                "Late": (17, 18),
                "Evening": (18, 20),
            }[choice]
            if preferred_days_selection:
                for day in preferred_days_selection:
                    if day not in DAY_OFFSETS:
                        continue
                    window = _make_time_window(DAY_OFFSETS[day], start_hour, end_hour)
                    preferred_time_windows.append(PreferredTimeWindow(window=window, weight=60))

        unavailable_windows = []
        # Unavailable days block the whole day (hard constraint)
        for day in hard_days:
            if day not in DAY_OFFSETS:
                continue
            unavailable_windows.append(_make_time_window(DAY_OFFSETS[day], 0, 23))

        # Unavailable time windows (apply to selected days or all weekdays)
        hard_days_effective = hard_days if hard_days else []
        if apply_unavailable_all_days:
            hard_days_effective = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for day in hard_days_effective:
            if day not in DAY_OFFSETS:
                continue
            for choice in hard_time_windows:
                start_hour, end_hour = {
                    "Early Morning": (8, 9),
                    "Morning": (9, 12),
                    "Midday": (12, 14),
                    "Afternoon": (14, 17),
                    "Late": (17, 18),
                    "Evening": (18, 20),
                }[choice]
                unavailable_windows.append(_make_time_window(DAY_OFFSETS[day], start_hour, end_hour))

        if unavailable_windows and preferred_time_windows:
            preferred_time_windows = [
                p for p in preferred_time_windows
                if all(
                    not (u.start_time <= p.window.start_time < u.end_time)
                    for u in unavailable_windows
                )
            ]
            if not preferred_time_windows:
                st.warning("Preferred time windows conflict with unavailable windows; preferences were cleared.")

        request = PatientRequest(
            request_id="REQ-1",
            free_text_reason=free_text,
            urgency_band=urgency,
            required_appt_type=appt_type,
            must_be_role=role if role_choice == "Auto (use triage)" else role_choice,
            must_be_mode=None if must_be_mode == "No constraint" else must_be_mode,
            preferred_days=preferred_days,
            preferred_modes=preferred_modes,
            preferred_time_windows=preferred_time_windows,
            preferred_slot_ids=[],
            soonest_weight=soonest_weight,
            consent_to_relax=consent_to_relax,
            unavailable_windows=unavailable_windows,
        )

    if st.button("Book Appointment"):
        st.markdown("---")
        st.header("Results")

        slots_filtered = [s for s in slots if s.slot_id not in set(prebooked_slots)]
        requests: Dict[str, PatientRequest] = {"REQ-1": request}
        if add_simulated:
            for i in range(1, sim_count + 1):
                sim_req = _synthetic_request(i)
                requests[sim_req.request_id] = sim_req

        assignments, log = negotiate(requests, slots_filtered, max_rounds=5, seed=42)

        if "REQ-1" not in assignments:
            st.warning("No slot could be assigned. Choose a compromise option to retry:")
            options = generate_compromises(request)
            if not options:
                st.info("No compromise options available.")
            for option in options:
                with st.container():
                    st.info(f"{option.label}: {option.prompt}")
                    if st.button(f"Apply: {option.label}", key=f"apply_{option.label}"):
                        updated_request = request.model_copy(update=option.patch)
                        requests["REQ-1"] = updated_request
                        assignments, log = negotiate(requests, slots_filtered, max_rounds=5, seed=42)
                        if "REQ-1" in assignments:
                            assigned_slot_id = assignments["REQ-1"]
                            slot_map = {s.slot_id: s for s in slots_filtered}
                            assigned_slot = slot_map[assigned_slot_id]
                            st.success(f"Assigned slot: {assigned_slot_id} at {assigned_slot.start_time}")
                            st.write(explain_assignment(assigned_slot, updated_request))
                        else:
                            st.error("Still no eligible slots after applying this compromise.")
            st.subheader("Negotiation Log")
            st.code("\n".join(log))
            return

        assigned_slot_id = assignments["REQ-1"]
        slot_map = {s.slot_id: s for s in slots_filtered}
        assigned_slot = slot_map[assigned_slot_id]

        st.success(f"Assigned slot: {assigned_slot_id} at {assigned_slot.start_time}")
        st.write(explain_assignment(assigned_slot, request))

        metrics = evaluate_run(assignments, requests, slots_filtered, negotiation_rounds=5, escalated_count=0)
        st.subheader("Evaluation Metrics")
        st.write(metrics)

        if prebooked_slots:
            st.subheader("Pre-booked Slots (Manual Simulation)")
            st.write(sorted(prebooked_slots))

        with st.expander("Negotiation Log"):
            st.code("\n".join(log))


if __name__ == "__main__":
    main()
