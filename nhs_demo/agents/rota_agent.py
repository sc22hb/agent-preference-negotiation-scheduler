from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from nhs_demo.config import SLOT_DATABASE_HORIZON_DAYS, SYNTHETIC_BASE_DATETIME
from nhs_demo.schemas import RoutingDecision, Slot


@dataclass(frozen=True)
class _SlotTemplate:
    slot_id: str
    service_type: str
    clinician_id: str
    site: str
    modality: str
    start_time: datetime
    duration_minutes: int


class RotaAgent:
    """Serve slots from a pre-built deterministic in-memory slot database."""

    HOSPITAL_NAME = "Leeds General Infirmary"

    def __init__(
        self,
        base_datetime: datetime = SYNTHETIC_BASE_DATETIME,
        database_horizon_days: int = SLOT_DATABASE_HORIZON_DAYS,
    ) -> None:
        self.base_datetime = base_datetime
        self.database_horizon_days = max(1, database_horizon_days)
        self._database_build_count = 0
        self._slot_database = self._build_slot_database()

    @property
    def database_build_count(self) -> int:
        return self._database_build_count

    def list_inventory(
        self,
        service_type: str | None = None,
        horizon_days: int | None = None,
    ) -> Dict[str, List[_SlotTemplate]]:
        requested_horizon = self.database_horizon_days if horizon_days is None else horizon_days
        effective_horizon = max(1, min(requested_horizon, self.database_horizon_days))
        horizon_end = self.base_datetime + timedelta(days=effective_horizon)

        available_services = set(self._slot_database.keys())
        if service_type is not None and service_type not in available_services:
            raise ValueError(f"Unknown service_type: {service_type}")

        target_services = [service_type] if service_type else sorted(available_services)
        inventory: Dict[str, List[_SlotTemplate]] = {}
        for service in target_services:
            inventory[service] = [
                slot for slot in self._slot_database[service] if slot.start_time < horizon_end
            ]
        return inventory

    def generate_slots(self, routing: RoutingDecision, horizon_days: int) -> List[Slot]:
        effective_horizon = max(1, min(horizon_days, self.database_horizon_days))
        horizon_end = self.base_datetime + timedelta(days=effective_horizon)

        service_templates = self._slot_database.get(routing.service_type, [])
        return [
            Slot(
                slot_id=template.slot_id,
                service_type=template.service_type,
                clinician_id=template.clinician_id,
                site=template.site,
                modality=template.modality,
                start_time=template.start_time,
                duration_minutes=routing.appointment_length_minutes,
            )
            for template in service_templates
            if template.start_time < horizon_end
        ]

    def _build_slot_database(self) -> Dict[str, List[_SlotTemplate]]:
        self._database_build_count += 1
        database: Dict[str, List[_SlotTemplate]] = {}
        service_order = ["GP", "Nurse", "Pharmacist", "Admin"]

        for service_type in service_order:
            templates = self._templates_for_service(service_type)
            service_slots: List[_SlotTemplate] = []
            service_prefix = self._service_prefix(service_type)

            for day_offset in range(self.database_horizon_days):
                day_start = self.base_datetime + timedelta(days=day_offset)
                if day_start.weekday() >= 5:
                    continue

                for idx, (hour, minute, modality, clinician_suffix) in enumerate(templates, start=1):
                    slot_start = day_start.replace(hour=hour, minute=minute)
                    slot_id = (
                        f"{service_prefix}-"
                        f"{slot_start.strftime('%Y%m%d-%H%M')}-"
                        f"LGI-{idx}"
                    )
                    service_slots.append(
                        _SlotTemplate(
                            slot_id=slot_id,
                            service_type=service_type,
                            clinician_id=f"{service_prefix}-{clinician_suffix}",
                            site=self.HOSPITAL_NAME,
                            modality=modality,
                            start_time=slot_start,
                            duration_minutes=self._default_duration_for_service(service_type),
                        )
                    )

            service_slots.sort(key=lambda slot: (slot.start_time, slot.slot_id))
            database[service_type] = service_slots

        return database

    def _service_prefix(self, service_type: str) -> str:
        return service_type[:2].upper()

    def _default_duration_for_service(self, service_type: str) -> int:
        if service_type == "Pharmacist":
            return 10
        if service_type == "Admin":
            return 10
        return 15

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
