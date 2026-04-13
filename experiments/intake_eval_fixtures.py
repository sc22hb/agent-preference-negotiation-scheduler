from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class IntakeExpected:
    complaint_category: str
    preferred_days: tuple[str, ...]
    excluded_days: tuple[str, ...]
    preferred_modalities: tuple[str, ...]
    excluded_modalities: tuple[str, ...]
    preferred_periods: tuple[str, ...]
    excluded_periods: tuple[str, ...]
    date_horizon_days: int
    preferred_day_periods: tuple[tuple[str, str], ...]
    duration_text: str


@dataclass(frozen=True)
class IntakePromptCase:
    case_id: str
    prompt_style: str
    prompt: str
    expected: IntakeExpected


def _ordered_unique(items: List[str]) -> tuple[str, ...]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _build_case(
    *,
    case_id: str,
    complaint_phrase: str,
    complaint_category: str,
    duration_text: str,
    horizon_days: int,
    preferred_days: List[str],
    excluded_days: List[str],
    preferred_modalities: List[str],
    excluded_modalities: List[str],
    preferred_periods: List[str],
    excluded_periods: List[str],
    preferred_day_periods: List[tuple[str, str]],
    canonical_prompt: str,
    noisy_prompt: str,
) -> List[IntakePromptCase]:
    expected = IntakeExpected(
        complaint_category=complaint_category,
        preferred_days=_ordered_unique(preferred_days),
        excluded_days=_ordered_unique(excluded_days),
        preferred_modalities=_ordered_unique(preferred_modalities),
        excluded_modalities=_ordered_unique(excluded_modalities),
        preferred_periods=_ordered_unique(preferred_periods),
        excluded_periods=_ordered_unique(excluded_periods),
        date_horizon_days=horizon_days,
        preferred_day_periods=tuple(preferred_day_periods),
        duration_text=duration_text,
    )
    return [
        IntakePromptCase(
            case_id=f"{case_id}-canonical",
            prompt_style="canonical",
            prompt=canonical_prompt.format(complaint=complaint_phrase),
            expected=expected,
        ),
        IntakePromptCase(
            case_id=f"{case_id}-noisy",
            prompt_style="noisy",
            prompt=noisy_prompt.format(complaint=complaint_phrase),
            expected=expected,
        ),
    ]


def intake_prompt_cases() -> List[IntakePromptCase]:
    complaint_catalogue = [
        ("persistent cough", "respiratory"),
        ("uti symptoms", "urinary"),
        ("blood test", "tests_monitoring"),
        ("repeat prescription", "prescription_admin"),
        ("fit note letter", "administrative"),
    ]

    blueprints: List[Dict[str, object]] = [
        {
            "duration_text": "2 weeks",
            "horizon_days": 10,
            "preferred_days": ["Mon"],
            "excluded_days": ["Tue"],
            "preferred_modalities": ["phone"],
            "excluded_modalities": ["video"],
            "preferred_periods": ["morning"],
            "excluded_periods": [],
            "preferred_day_periods": [("Mon", "morning")],
            "canonical_prompt": (
                "I need an appointment about {complaint}. It has been going on for 2 weeks. "
                "Monday morning would be best. I cannot do Tuesday. Phone is preferred and no video please. "
                "Please keep it within the next 10 days."
            ),
            "noisy_prompt": (
                "{complaint}, 2 wks. mon am best, not tue. phone pls, no vid. need within 10d."
            ),
        },
        {
            "duration_text": "3 days",
            "horizon_days": 7,
            "preferred_days": ["Wed"],
            "excluded_days": ["Fri"],
            "preferred_modalities": ["video"],
            "excluded_modalities": ["phone"],
            "preferred_periods": ["afternoon"],
            "excluded_periods": [],
            "preferred_day_periods": [("Wed", "afternoon")],
            "canonical_prompt": (
                "I would like to discuss {complaint}. This started 3 days ago. "
                "Wednesday afternoon is ideal. Friday does not work for me. "
                "Video would be best and I would rather avoid phone calls. "
                "Please keep it within the next 7 days."
            ),
            "noisy_prompt": (
                "{complaint} x3 days. wed pm ideal, fri no. video best, no phone. within 7d."
            ),
        },
        {
            "duration_text": "since yesterday",
            "horizon_days": 14,
            "preferred_days": ["Thu"],
            "excluded_days": ["Mon", "Tue"],
            "preferred_modalities": ["in_person"],
            "excluded_modalities": ["phone"],
            "preferred_periods": ["morning"],
            "excluded_periods": ["evening"],
            "preferred_day_periods": [("Thu", "morning")],
            "canonical_prompt": (
                "I need help with {complaint}. It has been there since yesterday. "
                "Thursday morning would suit me best. I cannot do Monday or Tuesday. "
                "I would prefer to be seen in person and not by phone. Evenings do not work. "
                "Please search within the next 2 weeks."
            ),
            "noisy_prompt": (
                "{complaint} since yday. thu am best. cant mon or tue. in person pls, no phone. no eve. within 2 wks."
            ),
        },
        {
            "duration_text": "1 week",
            "horizon_days": 21,
            "preferred_days": ["Mon", "Wed"],
            "excluded_days": ["Fri"],
            "preferred_modalities": ["phone"],
            "excluded_modalities": ["in_person"],
            "preferred_periods": ["morning"],
            "excluded_periods": ["afternoon"],
            "preferred_day_periods": [("Mon", "morning"), ("Wed", "morning")],
            "canonical_prompt": (
                "I need an appointment for {complaint}. It has been going on for 1 week. "
                "Monday morning or Wednesday morning would be best. Friday is not possible. "
                "Phone is preferred and I would like to avoid in-person appointments. "
                "Afternoons do not work for me. Please keep it within the next 3 weeks."
            ),
            "noisy_prompt": (
                "{complaint} 1 wk. mon am or wed am best. no fri. phone pref, avoid in person. no afternoons. within 3 wks."
            ),
        },
        {
            "duration_text": "5 days",
            "horizon_days": 14,
            "preferred_days": ["Tue"],
            "excluded_days": ["Thu"],
            "preferred_modalities": ["video"],
            "excluded_modalities": ["in_person"],
            "preferred_periods": ["afternoon"],
            "excluded_periods": ["morning"],
            "preferred_day_periods": [("Tue", "afternoon")],
            "canonical_prompt": (
                "I need advice about {complaint}. This has been a problem for 5 days. "
                "Tuesday afternoon would be best. Thursday does not work. "
                "I would prefer video and want to avoid in-person appointments. "
                "Mornings are difficult. Please keep the search within the next 14 days."
            ),
            "noisy_prompt": (
                "{complaint} 5 days. tue pm best, no thu. video pls, avoid in person. mornings bad. within 14d."
            ),
        },
        {
            "duration_text": "2 months",
            "horizon_days": 30,
            "preferred_days": ["Fri"],
            "excluded_days": ["Wed"],
            "preferred_modalities": ["in_person"],
            "excluded_modalities": ["video"],
            "preferred_periods": ["morning"],
            "excluded_periods": ["afternoon"],
            "preferred_day_periods": [("Fri", "morning")],
            "canonical_prompt": (
                "I want to discuss {complaint}. It has been going on for 2 months. "
                "Friday morning would be best. Wednesday is not possible. "
                "I would prefer an in-person appointment and no video please. "
                "Afternoons are not good for me. Please look within the next 30 days."
            ),
            "noisy_prompt": (
                "{complaint} 2 months. fri am best, wed no. in person pls, no video. no afternoons. within 30d."
            ),
        },
        {
            "duration_text": "4 days",
            "horizon_days": 10,
            "preferred_days": ["Mon", "Tue"],
            "excluded_days": ["Fri"],
            "preferred_modalities": ["phone"],
            "excluded_modalities": ["in_person"],
            "preferred_periods": ["morning"],
            "excluded_periods": [],
            "preferred_day_periods": [("Mon", "morning"), ("Tue", "morning")],
            "canonical_prompt": (
                "I need a booking for {complaint}. It has been present for 4 days. "
                "Monday morning or Tuesday morning would suit me best. Friday does not work. "
                "Phone is preferred and I want to avoid in-person appointments. "
                "Please keep it within the next 10 days."
            ),
            "noisy_prompt": (
                "{complaint} 4d. mon/tue am best. no fri. phone ok, avoid in person. within 10d."
            ),
        },
        {
            "duration_text": "6 days",
            "horizon_days": 12,
            "preferred_days": ["Thu"],
            "excluded_days": ["Mon"],
            "preferred_modalities": ["video"],
            "excluded_modalities": ["phone"],
            "preferred_periods": ["afternoon"],
            "excluded_periods": ["evening"],
            "preferred_day_periods": [("Thu", "afternoon")],
            "canonical_prompt": (
                "I need an appointment regarding {complaint}. It has lasted 6 days. "
                "Thursday afternoon would be ideal. Monday is not possible. "
                "Video is preferred and I would rather not have a phone consultation. "
                "Evenings do not work. Please search within the next 12 days."
            ),
            "noisy_prompt": (
                "{complaint} 6d. thu pm ideal. no mon. video pref, no phone. no eve. within 12d."
            ),
        },
        {
            "duration_text": "8 days",
            "horizon_days": 9,
            "preferred_days": ["Wed"],
            "excluded_days": ["Tue"],
            "preferred_modalities": ["phone"],
            "excluded_modalities": ["video"],
            "preferred_periods": ["morning"],
            "excluded_periods": ["afternoon"],
            "preferred_day_periods": [("Wed", "morning")],
            "canonical_prompt": (
                "I need help with {complaint}. It has been going on for 8 days. "
                "Wednesday morning would be best. Tuesday does not work for me. "
                "Phone is preferred and I do not want video. Afternoons do not work either. "
                "Please keep the appointment within the next 9 days."
            ),
            "noisy_prompt": (
                "{complaint} 8 days. wed am best. not tue. phone best, no vid. no aft. within 9d."
            ),
        },
        {
            "duration_text": "10 days",
            "horizon_days": 18,
            "preferred_days": ["Mon"],
            "excluded_days": ["Thu"],
            "preferred_modalities": ["in_person"],
            "excluded_modalities": ["phone"],
            "preferred_periods": ["afternoon"],
            "excluded_periods": ["morning"],
            "preferred_day_periods": [("Mon", "afternoon")],
            "canonical_prompt": (
                "I need an appointment about {complaint}. This has been going on for 10 days. "
                "Monday afternoon would be ideal. Thursday is not possible. "
                "I would prefer an in-person appointment and not a phone consultation. "
                "Mornings do not work. Please keep it within the next 18 days."
            ),
            "noisy_prompt": (
                "{complaint} 10d. mon pm ideal, no thu. in person pls, no phone. mornings bad. within 18d."
            ),
        },
    ]

    prompt_cases: List[IntakePromptCase] = []
    for blueprint_index, blueprint in enumerate(blueprints):
        complaint_phrase, complaint_category = complaint_catalogue[blueprint_index % len(complaint_catalogue)]
        prompt_cases.extend(
            _build_case(
                case_id=f"intake-{blueprint_index:02d}",
                complaint_phrase=complaint_phrase,
                complaint_category=complaint_category,
                duration_text=blueprint["duration_text"],  # type: ignore[arg-type]
                horizon_days=blueprint["horizon_days"],  # type: ignore[arg-type]
                preferred_days=blueprint["preferred_days"],  # type: ignore[arg-type]
                excluded_days=blueprint["excluded_days"],  # type: ignore[arg-type]
                preferred_modalities=blueprint["preferred_modalities"],  # type: ignore[arg-type]
                excluded_modalities=blueprint["excluded_modalities"],  # type: ignore[arg-type]
                preferred_periods=blueprint["preferred_periods"],  # type: ignore[arg-type]
                excluded_periods=blueprint["excluded_periods"],  # type: ignore[arg-type]
                preferred_day_periods=blueprint["preferred_day_periods"],  # type: ignore[arg-type]
                canonical_prompt=blueprint["canonical_prompt"],  # type: ignore[arg-type]
                noisy_prompt=blueprint["noisy_prompt"],  # type: ignore[arg-type]
            )
        )

    additional_cases = []
    for idx in range(10, 50):
        complaint_phrase, complaint_category = complaint_catalogue[idx % len(complaint_catalogue)]
        blueprint = blueprints[idx % len(blueprints)]
        additional_cases.extend(
            _build_case(
                case_id=f"intake-{idx:02d}",
                complaint_phrase=complaint_phrase,
                complaint_category=complaint_category,
                duration_text=blueprint["duration_text"],  # type: ignore[arg-type]
                horizon_days=blueprint["horizon_days"],  # type: ignore[arg-type]
                preferred_days=blueprint["preferred_days"],  # type: ignore[arg-type]
                excluded_days=blueprint["excluded_days"],  # type: ignore[arg-type]
                preferred_modalities=blueprint["preferred_modalities"],  # type: ignore[arg-type]
                excluded_modalities=blueprint["excluded_modalities"],  # type: ignore[arg-type]
                preferred_periods=blueprint["preferred_periods"],  # type: ignore[arg-type]
                excluded_periods=blueprint["excluded_periods"],  # type: ignore[arg-type]
                preferred_day_periods=blueprint["preferred_day_periods"],  # type: ignore[arg-type]
                canonical_prompt=blueprint["canonical_prompt"],  # type: ignore[arg-type]
                noisy_prompt=blueprint["noisy_prompt"],  # type: ignore[arg-type]
            )
        )

    return prompt_cases + additional_cases
