from __future__ import annotations

from typing import List

from nhs_demo.schemas import SafetyGateResult


class SafetyGateAgent:
    """Detect red-flag language and stop scheduling for safety escalation."""

    RED_FLAG_KEYWORDS = [
        "chest pain",
        "shortness of breath",
        "breathing difficulty",
        "severe bleeding",
        "stroke",
        "unconscious",
        "heart attack",
        "anaphylaxis",
        "seizure",
        "fainting",
    ]

    def assess(self, user_text: str) -> SafetyGateResult:
        text = user_text.lower()
        matches: List[str] = [keyword for keyword in self.RED_FLAG_KEYWORDS if keyword in text]
        if not matches:
            return SafetyGateResult(triggered=False)

        return SafetyGateResult(
            triggered=True,
            matched_keywords=matches,
            message=(
                "This technical demonstrator cannot schedule this request. "
                "Please use an urgent professional care pathway immediately."
            ),
        )
