from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .models import ApptMode, ApptType, ClinicianRole, Slot


BASE_DATE = datetime(2026, 2, 3, 9, 0)


def _slot(
    slot_id: str,
    day_offset: int,
    hour: int,
    minute: int,
    clinician_role: ClinicianRole,
    mode: ApptMode,
    appt_type: ApptType,
    duration_minutes: int = 15,
    tags: List[str] | None = None,
) -> Slot:
    start_time = BASE_DATE + timedelta(days=day_offset)
    start_time = start_time.replace(hour=hour, minute=minute)
    return Slot(
        slot_id=slot_id,
        clinician_role=clinician_role,
        mode=mode,
        appt_type=appt_type,
        start_time=start_time,
        duration_minutes=duration_minutes,
        capacity=1,
        tags=tags or [],
    )


def build_slot_inventory() -> List[Slot]:
    """Synthetic NHS-style slot inventory (deterministic)."""
    slots: List[Slot] = []

    # Monday (day_offset=0)
    slots.extend([
        _slot("NHS-MON-0900-GP-IP", 0, 9, 0, "GP", "IN_PERSON", "URGENT", 15, ["same_day"]),
        _slot("NHS-MON-0930-NURSE-PH", 0, 9, 30, "NURSE", "PHONE", "ROUTINE", 15, ["routine"]),
        _slot("NHS-MON-1000-GP-VID", 0, 10, 0, "GP", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-MON-1100-GP-IP", 0, 11, 0, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-MON-1200-NURSE-IP", 0, 12, 0, "NURSE", "IN_PERSON", "TEST", 20, ["test"]),
        _slot("NHS-MON-1330-PHARM-PH", 0, 13, 30, "PHARMACIST", "PHONE", "ROUTINE", 10, ["prescription"]),
        _slot("NHS-MON-1430-NURSE-IP", 0, 14, 30, "NURSE", "IN_PERSON", "TEST", 20, ["test"]),
        _slot("NHS-MON-1600-GP-IP", 0, 16, 0, "GP", "IN_PERSON", "REVIEW", 15, ["review"]),
        _slot("NHS-MON-1700-GP-PH", 0, 17, 0, "GP", "PHONE", "ROUTINE", 10, ["routine"]),
    ])

    # Tuesday (day_offset=1)
    slots.extend([
        _slot("NHS-TUE-0900-GP-PH", 1, 9, 0, "GP", "PHONE", "URGENT", 10, ["same_day"]),
        _slot("NHS-TUE-1000-NURSE-IP", 1, 10, 0, "NURSE", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-TUE-1130-GP-VID", 1, 11, 30, "GP", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-TUE-1230-GP-IP", 1, 12, 30, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-TUE-1300-PHARM-PH", 1, 13, 0, "PHARMACIST", "PHONE", "ROUTINE", 10, ["prescription"]),
        _slot("NHS-TUE-1400-GP-IP", 1, 14, 0, "GP", "IN_PERSON", "URGENT", 15, ["same_day"]),
        _slot("NHS-TUE-1530-NURSE-VID", 1, 15, 30, "NURSE", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-TUE-1630-GP-IP", 1, 16, 30, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-TUE-1730-NURSE-PH", 1, 17, 30, "NURSE", "PHONE", "ROUTINE", 10, ["routine"]),
    ])

    # Wednesday (day_offset=2)
    slots.extend([
        _slot("NHS-WED-0900-GP-IP", 2, 9, 0, "GP", "IN_PERSON", "URGENT", 15, ["same_day"]),
        _slot("NHS-WED-1030-NURSE-PH", 2, 10, 30, "NURSE", "PHONE", "ROUTINE", 15, ["routine"]),
        _slot("NHS-WED-1200-GP-VID", 2, 12, 0, "GP", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-WED-1330-PHARM-PH", 2, 13, 30, "PHARMACIST", "PHONE", "ROUTINE", 10, ["prescription"]),
        _slot("NHS-WED-1500-NURSE-IP", 2, 15, 0, "NURSE", "IN_PERSON", "TEST", 20, ["test"]),
        _slot("NHS-WED-1600-GP-IP", 2, 16, 0, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-WED-1700-GP-PH", 2, 17, 0, "GP", "PHONE", "REVIEW", 10, ["review"]),
    ])

    # Thursday (day_offset=3)
    slots.extend([
        _slot("NHS-THU-0900-NURSE-IP", 3, 9, 0, "NURSE", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-THU-1000-GP-IP", 3, 10, 0, "GP", "IN_PERSON", "URGENT", 15, ["same_day"]),
        _slot("NHS-THU-1100-GP-VID", 3, 11, 0, "GP", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-THU-1230-PHARM-PH", 3, 12, 30, "PHARMACIST", "PHONE", "ROUTINE", 10, ["prescription"]),
        _slot("NHS-THU-1330-NURSE-IP", 3, 13, 30, "NURSE", "IN_PERSON", "TEST", 20, ["test"]),
        _slot("NHS-THU-1500-GP-IP", 3, 15, 0, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-THU-1630-GP-PH", 3, 16, 30, "GP", "PHONE", "REVIEW", 10, ["review"]),
    ])

    # Friday (day_offset=4)
    slots.extend([
        _slot("NHS-FRI-0900-GP-IP", 4, 9, 0, "GP", "IN_PERSON", "URGENT", 15, ["same_day"]),
        _slot("NHS-FRI-1000-NURSE-PH", 4, 10, 0, "NURSE", "PHONE", "ROUTINE", 15, ["routine"]),
        _slot("NHS-FRI-1100-GP-IP", 4, 11, 0, "GP", "IN_PERSON", "ROUTINE", 15, ["routine"]),
        _slot("NHS-FRI-1200-GP-VID", 4, 12, 0, "GP", "VIDEO", "REVIEW", 20, ["review"]),
        _slot("NHS-FRI-1330-PHARM-PH", 4, 13, 30, "PHARMACIST", "PHONE", "ROUTINE", 10, ["prescription"]),
        _slot("NHS-FRI-1430-NURSE-IP", 4, 14, 30, "NURSE", "IN_PERSON", "TEST", 20, ["test"]),
        _slot("NHS-FRI-1600-GP-IP", 4, 16, 0, "GP", "IN_PERSON", "REVIEW", 15, ["review"]),
    ])

    return slots


__all__ = ["build_slot_inventory", "BASE_DATE"]
