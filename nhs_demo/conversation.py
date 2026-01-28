from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from .models import ApptMode, ClinicianRole, PatientRequest, TimeWindow

FailureType = Literal["NO_ELIGIBLE_SLOTS", "CAPACITY_FULL", "MODE_MISMATCH", "URGENCY_UNMET"]


@dataclass
class CompromiseOption:
    label: str
    prompt: str
    patch: Dict


def diagnose_failure() -> FailureType:
    # Placeholder heuristic; engine-level diagnostics can enrich this later.
    return "NO_ELIGIBLE_SLOTS"


def _widen_time_windows(request: PatientRequest) -> Dict:
    if not request.preferred_time_windows:
        return {"preferred_time_windows": []}
    widened = []
    for pref in request.preferred_time_windows:
        window = pref.window
        widened.append(
            {
                "window": TimeWindow(
                    start_time=window.start_time,
                    end_time=window.end_time,
                ),
                "weight": pref.weight,
            }
        )
    return {"preferred_time_windows": widened}


def _relax_mode(request: PatientRequest) -> Dict:
    return {"must_be_mode": None, "preferred_modes": {"PHONE": 70, "VIDEO": 60, "IN_PERSON": 80}}


def _relax_role(request: PatientRequest) -> Dict:
    return {"must_be_role": None}


def generate_compromises(request: PatientRequest) -> List[CompromiseOption]:
    options: List[CompromiseOption] = []

    options.append(
        CompromiseOption(
            label="Widen time preferences",
            prompt="Would you be open to a wider time range if needed?",
            patch=_widen_time_windows(request),
        )
    )

    if request.must_be_mode is not None:
        options.append(
            CompromiseOption(
                label="Allow phone or video",
                prompt="Would a phone or video appointment work for you?",
                patch=_relax_mode(request),
            )
        )

    if request.must_be_role is not None:
        options.append(
            CompromiseOption(
                label="Allow alternative clinician",
                prompt="If no GP slots are available, can you see a nurse or pharmacist?",
                patch=_relax_role(request),
            )
        )

    return options[:3]


__all__ = ["FailureType", "CompromiseOption", "diagnose_failure", "generate_compromises"]
