from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from nhs_demo.config import ROUTING_RULES_PATH
from nhs_demo.schemas import IntakeSummary, RoutingDecision


class TriageRoutingAgent:
    """Deterministic rule-file-based routing from intake summary to service pathway."""

    def __init__(self, rules_path: Path = ROUTING_RULES_PATH) -> None:
        self.rules_path = rules_path
        with rules_path.open("r", encoding="utf-8") as handle:
            self.rules: List[Dict[str, Any]] = json.load(handle)

    def route(self, intake: IntakeSummary) -> RoutingDecision:
        haystack = " ".join(
            [
                intake.raw_text,
                intake.complaint_category,
                intake.duration_text or "",
                " ".join(intake.extracted_constraints_text),
            ]
        ).lower()

        matched_rule = self.rules[-1]
        for rule in self.rules:
            if self._matches(rule.get("match", {}), haystack):
                matched_rule = rule
                break

        route = matched_rule["route"]
        explanation = (
            f"Rule '{matched_rule['id']}' matched; {matched_rule.get('description', 'no description')}."
        )

        return RoutingDecision(
            run_id=intake.run_id,
            service_type=route["service_type"],
            appointment_length_minutes=route["appointment_length_minutes"],
            allowed_modalities=route["allowed_modalities"],
            urgency_band=route["urgency_band"],
            confidence=route["confidence"],
            rule_hit=matched_rule["id"],
            explanation=explanation,
        )

    def _matches(self, match_spec: Dict[str, Any], haystack: str) -> bool:
        any_keywords = [k.lower() for k in match_spec.get("any", [])]
        all_keywords = [k.lower() for k in match_spec.get("all", [])]

        any_ok = True if not any_keywords else any(keyword in haystack for keyword in any_keywords)
        all_ok = all(keyword in haystack for keyword in all_keywords)
        return any_ok and all_ok
