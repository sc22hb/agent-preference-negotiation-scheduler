from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from pydantic import ValidationError

from .models import PatientRequest
from .triage import triage_request


@dataclass
class ExtractionResult:
    request: Optional[PatientRequest]
    error: Optional[str]


def extract_request_from_text(free_text_reason: str) -> ExtractionResult:
    """Deterministic stub: converts free text into a validated PatientRequest.

    This is a placeholder for an LLM extraction layer. It uses triage rules
    and a minimal default preference profile to produce valid JSON-like output.
    """
    try:
        urgency_band, appt_type, role_suggestion = triage_request(free_text_reason)
        request = PatientRequest(
            request_id="REQ-1",
            free_text_reason=free_text_reason,
            urgency_band=urgency_band,
            required_appt_type=appt_type,
            must_be_role=role_suggestion,
            must_be_mode=None,
            preferred_days={},
            preferred_modes={},
            preferred_time_windows=[],
            preferred_slot_ids=[],
            soonest_weight=50,
            consent_to_relax=True,
        )
        return ExtractionResult(request=request, error=None)
    except ValueError as exc:
        return ExtractionResult(request=None, error=str(exc))
    except ValidationError as exc:
        return ExtractionResult(request=None, error=str(exc))


__all__ = ["ExtractionResult", "extract_request_from_text"]
