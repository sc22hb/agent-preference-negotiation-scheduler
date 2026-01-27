from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

DayName = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ClinicianRole = Literal["GP", "NURSE", "PHARMACIST"]
ApptMode = Literal["IN_PERSON", "PHONE", "VIDEO"]
ApptType = Literal["URGENT", "ROUTINE", "REVIEW", "TEST"]
UrgencyBand = Literal["SAME_DAY", "SOON", "ROUTINE"]

VALID_DAYS: List[DayName] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class TimeWindow(BaseModel):
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def validate_time_order(self) -> "TimeWindow":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class Slot(BaseModel):
    slot_id: str
    clinician_role: ClinicianRole
    mode: ApptMode
    appt_type: ApptType
    start_time: datetime
    duration_minutes: int = Field(gt=0)
    capacity: int = Field(default=1, ge=1)
    tags: List[str] = Field(default_factory=list)


class PreferredTimeWindow(BaseModel):
    window: TimeWindow
    weight: int = Field(ge=0, le=100)


class PatientRequest(BaseModel):
    request_id: str
    free_text_reason: str
    urgency_band: UrgencyBand
    required_appt_type: ApptType

    # Hard constraints
    unavailable_windows: List[TimeWindow] = Field(default_factory=list)
    must_be_mode: Optional[ApptMode] = None
    must_be_role: Optional[ClinicianRole] = None

    # Soft preferences (0–100)
    preferred_time_windows: List[PreferredTimeWindow] = Field(default_factory=list)
    preferred_slot_ids: List[str] = Field(default_factory=list)
    preferred_days: Dict[DayName, int] = Field(default_factory=dict)
    preferred_modes: Dict[ApptMode, int] = Field(default_factory=dict)
    soonest_weight: int = Field(default=50, ge=0, le=100)

    consent_to_relax: bool = True

    @model_validator(mode="after")
    def validate_preference_weights(self) -> "PatientRequest":
        for day, weight in self.preferred_days.items():
            if day not in VALID_DAYS:
                raise ValueError(f"invalid day in preferred_days: {day}")
            if not (0 <= weight <= 100):
                raise ValueError("preferred_days weights must be 0-100")
        for mode, weight in self.preferred_modes.items():
            if not (0 <= weight <= 100):
                raise ValueError("preferred_modes weights must be 0-100")
        return self


__all__ = [
    "ApptMode",
    "ApptType",
    "ClinicianRole",
    "DayName",
    "PreferredTimeWindow",
    "Slot",
    "TimeWindow",
    "UrgencyBand",
    "PatientRequest",
]
