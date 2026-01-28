from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path

import streamlit as st

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
    from .nhs_slots import build_slot_inventory
    from .triage import triage_request
except ImportError:  # pragma: no cover
    from nhs_demo.conversation import generate_compromises
    from nhs_demo.explain import explain_assignment
    from nhs_demo.llm_extract import extract_request_from_text
    from nhs_demo.metrics import evaluate_run
    from nhs_demo.models import PatientRequest, PreferredTimeWindow, TimeWindow
    from nhs_demo.nhs_engine import negotiate, rank_slots
    from nhs_demo.nhs_slots import build_slot_inventory
    from nhs_demo.triage import triage_request


def _make_time_window(day_offset: int, start_hour: int, end_hour: int) -> TimeWindow:
    base = datetime(2026, 2, 3) + timedelta(days=day_offset)
    return TimeWindow(
        start_time=base.replace(hour=start_hour, minute=0),
        end_time=base.replace(hour=end_hour, minute=0),
    )


def _random_request(i: int) -> PatientRequest:
    # Simple synthetic request for simulated patients
    free_text = "routine check"
    urgency, appt_type, role = triage_request(free_text)
    return PatientRequest(
        request_id=f"SIM-{i}",
        free_text_reason=free_text,
        urgency_band=urgency,
        required_appt_type=appt_type,
        must_be_role=role,
        must_be_mode=None,
        preferred_days={},
        preferred_modes={},
        preferred_time_windows=[],
        preferred_slot_ids=[],
        soonest_weight=40,
        consent_to_relax=True,
    )


def main() -> None:
    st.set_page_config(page_title="NHS Multi‑Agent Appointment Demo", layout="wide")
    st.title("NHS Multi‑Agent Appointment Booking Demo")
    st.caption("Deterministic scheduling; LLM only for extraction (stubbed).")

    slots = build_slot_inventory()

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio("Preference capture", ["Form", "Chatbot (stub)"])
    simulate_others = st.sidebar.slider("Simulated other patients", 0, 6, 3)

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
        ]
        free_text = st.selectbox("Reason for appointment", reason_options, index=4)
        urgency, appt_type, role = triage_request(free_text)
        st.write(f"Triage: {urgency}, {appt_type}, {role}")

        preferred_day = st.selectbox("Preferred day", ["Mon", "Tue", "Wed", "Thu", "Fri"], index=0)
        preferred_mode = st.selectbox("Preferred mode", ["IN_PERSON", "PHONE", "VIDEO"], index=0)
        time_window_choice = st.selectbox("Preferred time window", ["Morning", "Afternoon", "Any"], index=0)

        preferred_days = {preferred_day: 70}
        preferred_modes = {preferred_mode: 70}
        preferred_time_windows: List[PreferredTimeWindow] = []
        if time_window_choice == "Morning":
            window = _make_time_window(0, 9, 12)
            preferred_time_windows.append(PreferredTimeWindow(window=window, weight=60))
        elif time_window_choice == "Afternoon":
            window = _make_time_window(0, 13, 17)
            preferred_time_windows.append(PreferredTimeWindow(window=window, weight=60))

        request = PatientRequest(
            request_id="REQ-1",
            free_text_reason=free_text,
            urgency_band=urgency,
            required_appt_type=appt_type,
            must_be_role=role,
            must_be_mode=None,
            preferred_days=preferred_days,
            preferred_modes=preferred_modes,
            preferred_time_windows=preferred_time_windows,
            preferred_slot_ids=[],
            soonest_weight=50,
            consent_to_relax=True,
        )

    st.markdown("---")
    st.header("Results")

    requests: Dict[str, PatientRequest] = {"REQ-1": request}
    for i in range(simulate_others):
        sim_req = _random_request(i + 1)
        requests[sim_req.request_id] = sim_req

    assignments, log = negotiate(requests, slots, max_rounds=5, seed=42)

    if "REQ-1" not in assignments:
        st.warning("No slot could be assigned. Offering compromise options:")
        for option in generate_compromises(request):
            st.info(f"{option.label}: {option.prompt}")
        return

    assigned_slot_id = assignments["REQ-1"]
    slot_map = {s.slot_id: s for s in slots}
    assigned_slot = slot_map[assigned_slot_id]

    st.success(f"Assigned slot: {assigned_slot_id} at {assigned_slot.start_time}")
    st.write(explain_assignment(assigned_slot, request))

    metrics = evaluate_run(assignments, requests, slots, negotiation_rounds=5, escalated_count=0)
    st.subheader("Evaluation Metrics")
    st.write(metrics)

    with st.expander("Negotiation Log"):
        st.code("\n".join(log))


if __name__ == "__main__":
    main()
