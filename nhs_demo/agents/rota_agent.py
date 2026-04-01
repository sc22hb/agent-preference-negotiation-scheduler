from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import Callable, Dict, List, Tuple

from nhs_demo.config import SLOT_DATABASE_HORIZON_DAYS
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
    """Serve slots from a live in-memory slot database anchored to the current date."""

    HOSPITAL_NAME = "Leeds General Infirmary"

    def __init__(
        self,
        base_datetime: datetime | None = None,
        database_horizon_days: int = SLOT_DATABASE_HORIZON_DAYS,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_provider = now_provider or datetime.now
        self.base_datetime = (base_datetime or self._live_base_datetime()).replace(tzinfo=None)
        self.database_horizon_days = max(1, database_horizon_days)
        self._database_build_count = 0
        self._database_anchor_date = self.base_datetime.date()
        self._stochastic_mode = False
        self._slot_database = self._build_slot_database()

    @property
    def database_build_count(self) -> int:
        return self._database_build_count

    @property
    def stochastic_mode(self) -> bool:
        return self._stochastic_mode

    def list_inventory(
        self,
        service_type: str | None = None,
        horizon_days: int | None = None,
    ) -> Dict[str, List[_SlotTemplate]]:
        self._refresh_slot_database_if_needed()
        now = self._now()
        requested_horizon = self.database_horizon_days if horizon_days is None else horizon_days
        effective_horizon = max(1, min(requested_horizon, self.database_horizon_days))
        horizon_end = now + timedelta(days=effective_horizon)

        available_services = set(self._slot_database.keys())
        if service_type is not None and service_type not in available_services:
            raise ValueError(f"Unknown service_type: {service_type}")

        target_services = [service_type] if service_type else sorted(available_services)
        inventory: Dict[str, List[_SlotTemplate]] = {}
        for service in target_services:
            inventory[service] = [
                slot
                for slot in self._slot_database[service]
                if now <= slot.start_time < horizon_end
            ]
        return inventory

    def generate_slots(self, routing: RoutingDecision, horizon_days: int) -> List[Slot]:
        self._refresh_slot_database_if_needed()
        now = self._now()
        effective_horizon = max(1, min(horizon_days, self.database_horizon_days))
        horizon_end = now + timedelta(days=effective_horizon)

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
            if now <= template.start_time < horizon_end
        ]

    def generate_scaled_slots(self, routing: RoutingDecision, total_slots: int) -> List[Slot]:
        """Build a deterministic synthetic slot pool of an exact size for experiments."""
        if total_slots <= 0:
            return []

        self._refresh_slot_database_if_needed()
        now = self._now()
        templates = self._templates_for_service(routing.service_type)
        if not templates:
            return []

        slots: List[Slot] = []
        service_prefix = self._service_prefix(routing.service_type)
        day_offset = 0

        while len(slots) < total_slots:
            day_start = self.base_datetime + timedelta(days=day_offset)
            day_offset += 1
            if day_start.weekday() >= 5:
                continue

            for template_index, (hour, minute, modality, clinician_suffix) in enumerate(templates, start=1):
                slot_start = day_start.replace(hour=hour, minute=minute)
                if slot_start < now:
                    continue
                slot_number = len(slots) + 1
                slots.append(
                    Slot(
                        slot_id=(
                            f"{service_prefix}-SCALE-"
                            f"{slot_start.strftime('%Y%m%d-%H%M')}-"
                            f"{template_index:02d}-{slot_number:04d}"
                        ),
                        service_type=routing.service_type,
                        clinician_id=f"{service_prefix}-{clinician_suffix}",
                        site=self.HOSPITAL_NAME,
                        modality=modality,
                        start_time=slot_start,
                        duration_minutes=routing.appointment_length_minutes,
                    )
                )
                if len(slots) >= total_slots:
                    break

        return slots

    def set_stochastic_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._stochastic_mode == enabled:
            return
        self._refresh_slot_database_if_needed()
        self._stochastic_mode = enabled
        self._slot_database = self._build_slot_database()

    def rotate_inventory(self) -> None:
        self._refresh_slot_database_if_needed()
        if not self._stochastic_mode:
            return
        self._slot_database = self._build_slot_database()

    def _build_slot_database(self) -> Dict[str, List[_SlotTemplate]]:
        self._database_build_count += 1
        database: Dict[str, List[_SlotTemplate]] = {}
        service_order = ["GP", "Nurse", "Pharmacist", "Admin"]
        rng = self._inventory_rng() if self._stochastic_mode else None

        for service_type in service_order:
            service_slots: List[_SlotTemplate] = []
            service_prefix = self._service_prefix(service_type)

            for day_offset in range(self.database_horizon_days):
                day_start = self.base_datetime + timedelta(days=day_offset)
                if day_start.weekday() >= 5:
                    continue

                templates = self._daily_templates_for_service(service_type, rng)
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

    def _refresh_slot_database_if_needed(self) -> None:
        current_anchor = self._live_base_datetime()
        if current_anchor.date() == self._database_anchor_date:
            return
        self.base_datetime = current_anchor
        self._database_anchor_date = current_anchor.date()
        self._slot_database = self._build_slot_database()

    def _live_base_datetime(self) -> datetime:
        current = self._now()
        return current.replace(hour=0, minute=0, second=0, microsecond=0)

    def _inventory_rng(self) -> random.Random:
        seed = f"{self._database_anchor_date.isoformat()}-{self._database_build_count}"
        return random.Random(seed)

    def _now(self) -> datetime:
        current = self._now_provider()
        return current.replace(tzinfo=None)

    def _service_prefix(self, service_type: str) -> str:
        return service_type[:2].upper()

    def _default_duration_for_service(self, service_type: str) -> int:
        if service_type == "Pharmacist":
            return 10
        if service_type == "Admin":
            return 10
        return 15

    def _daily_templates_for_service(
        self,
        service_type: str,
        rng: random.Random | None,
    ) -> List[Tuple[int, int, str, str]]:
        templates = self._templates_for_service(service_type)
        if rng is None:
            return templates

        template_pool = self._template_pool_for_service(service_type)
        sample_size = min(len(templates), len(template_pool))
        sampled_templates = rng.sample(template_pool, k=sample_size)
        return sorted(sampled_templates, key=lambda item: (item[0], item[1], item[3], item[2]))

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

    def _template_pool_for_service(self, service_type: str) -> List[Tuple[int, int, str, str]]:
        if service_type == "GP":
            return [
                (8, 30, "phone", "A"),
                (9, 0, "in_person", "A"),
                (9, 45, "video", "B"),
                (10, 30, "phone", "A"),
                (11, 30, "video", "B"),
                (13, 0, "phone", "C"),
                (14, 0, "in_person", "B"),
                (15, 0, "video", "C"),
                (16, 0, "phone", "C"),
            ]
        if service_type == "Nurse":
            return [
                (8, 45, "in_person", "N1"),
                (9, 30, "in_person", "N1"),
                (10, 30, "phone", "N2"),
                (11, 0, "in_person", "N2"),
                (13, 30, "phone", "N1"),
                (15, 0, "in_person", "N3"),
                (16, 0, "phone", "N3"),
            ]
        if service_type == "Pharmacist":
            return [
                (9, 0, "phone", "P1"),
                (9, 30, "phone", "P1"),
                (11, 30, "in_person", "P1"),
                (12, 0, "in_person", "P1"),
                (14, 30, "phone", "P2"),
                (15, 30, "phone", "P2"),
            ]
        return [
            (9, 30, "phone", "ADM1"),
            (10, 0, "phone", "ADM1"),
            (11, 30, "video", "ADM1"),
            (14, 0, "phone", "ADM2"),
            (14, 30, "video", "ADM1"),
            (16, 0, "phone", "ADM2"),
        ]
