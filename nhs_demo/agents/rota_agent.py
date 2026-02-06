from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

from nhs_demo.config import SYNTHETIC_BASE_DATETIME
from nhs_demo.schemas import RoutingDecision, Slot


class RotaAgent:
    """Generate deterministic synthetic slots from service templates."""

    HOSPITAL_NAME = "Leeds General Infirmary"

    def __init__(self, base_datetime: datetime = SYNTHETIC_BASE_DATETIME) -> None:
        self.base_datetime = base_datetime

    def generate_slots(self, routing: RoutingDecision, horizon_days: int) -> List[Slot]:
        templates = self._templates_for_service(routing.service_type)
        slots: List[Slot] = []

        for day_offset in range(horizon_days):
            day_start = self.base_datetime + timedelta(days=day_offset)
            if day_start.weekday() >= 5:
                continue

            for idx, (hour, minute, modality, clinician_suffix) in enumerate(templates, start=1):
                slot_start = day_start.replace(hour=hour, minute=minute)
                slot_id = (
                    f"{routing.service_type[:2].upper()}-"
                    f"{slot_start.strftime('%Y%m%d-%H%M')}-"
                    f"LGI-{idx}"
                )
                slots.append(
                    Slot(
                        slot_id=slot_id,
                        service_type=routing.service_type,
                        clinician_id=f"{routing.service_type[:2].upper()}-{clinician_suffix}",
                        site=self.HOSPITAL_NAME,
                        modality=modality,
                        start_time=slot_start,
                        duration_minutes=routing.appointment_length_minutes,
                    )
                )

        slots.sort(key=lambda slot: (slot.start_time, slot.slot_id))
        return slots

    def _templates_for_service(self, service_type: str) -> List[Tuple[int, int, str, str]]:
        if service_type == "GP":
            return [
                (9, 0, "in_person", "A"),
                (10, 30, "phone", "A"),
                (11, 30, "video", "B"),
                (14, 0, "in_person", "B"),
                (16, 0, "phone", "C"),
            ]
        if service_type == "Nurse":
            return [
                (9, 30, "in_person", "N1"),
                (11, 0, "in_person", "N2"),
                (13, 30, "phone", "N1"),
                (15, 0, "in_person", "N3"),
            ]
        if service_type == "Pharmacist":
            return [
                (9, 30, "phone", "P1"),
                (12, 0, "in_person", "P1"),
                (15, 30, "phone", "P2"),
            ]
        return [
            (10, 0, "phone", "ADM1"),
            (14, 30, "video", "ADM1"),
            (16, 0, "phone", "ADM2"),
        ]
