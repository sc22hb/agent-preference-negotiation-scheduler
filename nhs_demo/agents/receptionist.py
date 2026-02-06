from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from nhs_demo.config import DEFAULT_DATE_HORIZON_DAYS
from nhs_demo.schemas import (
    ClarificationQuestion,
    ExtractorMode,
    IntakeSummary,
    PatientPreferences,
    RelaxationQuestion,
)

class ReceptionistAgent:
    """Extract a structured non-diagnostic intake summary from conversation or form input."""

    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "prescription_admin": ["prescription", "refill", "medication"],
        "respiratory": ["cough", "asthma", "wheeze", "breath"],
        "urinary": ["uti", "urinary", "urine"],
        "skin": ["rash", "skin"],
        "tests_monitoring": ["blood test", "bp", "blood pressure", "monitoring"],
        "administrative": ["fit note", "letter", "certificate", "admin"],
    }

    DAY_PATTERNS: Dict[str, str] = {
        "Mon": r"\b(?:mon|monday|mondays)\b",
        "Tue": r"\b(?:tue|tues|tuesday|tuesdays)\b",
        "Wed": r"\b(?:wed|weds|wednesday|wednesdays|wednseday|wednsedays)\b",
        "Thu": r"\b(?:thu|thur|thurs|thursday|thursdays)\b",
        "Fri": r"\b(?:fri|friday|fridays)\b",
        "Sat": r"\b(?:sat|saturday|saturdays)\b",
        "Sun": r"\b(?:sun|sunday|sundays)\b",
    }
    DAY_ORDER: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    PERIOD_PATTERNS: Dict[str, str] = {
        "morning": r"\bmorning\b|\bearly\b",
        "afternoon": r"\bafternoon\b|\bmidday\b",
        "evening": r"\bevening\b|\blate\b",
    }

    FORM_REASON_TEXT: Dict[str, str] = {
        "repeat_prescription": "repeat prescription",
        "blood_test": "blood test",
        "persistent_cough": "persistent cough",
        "uti_symptoms": "uti symptoms",
        "sore_throat": "sore throat",
        "admin_request": "fit note letter admin request",
        "general_review": "routine check",
    }
    MODALITIES = ["in_person", "phone", "video"]
    PERIODS = ["morning", "afternoon", "evening"]

    def build_intake(
        self,
        run_id: str,
        user_text: str,
        clarification_answers: Dict[str, str],
        extractor: ExtractorMode,
        api_key: str | None = None,
        llm_model: str | None = None,
        preference_hint: PatientPreferences | None = None,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion], str, Dict[str, Any] | None]:
        merged_text = self._merge_text_with_clarifications(user_text, clarification_answers)
        if extractor == "llm":
            llm_result = self._try_llm(
                run_id=run_id,
                text=merged_text,
                preference_hint=preference_hint,
                api_key=api_key,
                llm_model=llm_model,
            )
            if llm_result is None:
                raise ValueError(
                    "LLM extraction failed. Check API key/model or use Form-based mode."
                )
            return llm_result[0], llm_result[1], "llm", llm_result[2]

        intake, questions = self._build_intake_rule_from_text(run_id=run_id, merged_text=merged_text)
        return intake, questions, "rule", None

    def build_form_intake(
        self,
        run_id: str,
        reason_code: str,
        preferences: PatientPreferences,
        extractor: ExtractorMode,
        api_key: str | None = None,
        llm_model: str | None = None,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion], str, Dict[str, Any] | None]:
        reason_text = self.FORM_REASON_TEXT.get(reason_code, "routine check")
        merged_text = self._compose_form_text(
            reason_text=reason_text,
            preferences=preferences,
        )

        if extractor == "llm":
            llm_result = self._try_llm(
                run_id=run_id,
                text=merged_text,
                preference_hint=preferences,
                api_key=api_key,
                llm_model=llm_model,
            )
            if llm_result is None:
                raise ValueError(
                    "LLM extraction failed. Check API key/model or use Form-based mode."
                )
            intake_llm, _questions, llm_payload = llm_result
            intake_llm = intake_llm.model_copy(update={"missing_fields": []})
            return intake_llm, [], "llm", llm_payload

        intake = IntakeSummary(
            run_id=run_id,
            raw_text=merged_text,
            complaint_category=self._extract_category(merged_text),
            duration_text=self._extract_duration(merged_text),
            extracted_constraints_text=self._constraints_from_preferences(preferences),
            preferences=preferences,
            missing_fields=[],
        )
        return intake, [], "rule", None

    def build_relaxation_questions(
        self,
        candidate_questions: List[RelaxationQuestion],
    ) -> List[RelaxationQuestion]:
        prompt_overrides = {
            "relax_excluded_periods": "Would you allow other times of day if needed?",
            "relax_excluded_days": "Would you allow additional days if needed?",
            "relax_excluded_modalities": "Would you allow additional consultation formats if needed?",
            "extend_date_horizon": "Would you allow us to search a few more days ahead?",
        }
        refined: List[RelaxationQuestion] = []
        for question in candidate_questions[:2]:
            refined.append(
                RelaxationQuestion(
                    key=question.key,
                    prompt=prompt_overrides.get(question.key, question.prompt),
                )
            )
        return refined

    def _try_llm(
        self,
        run_id: str,
        text: str,
        preference_hint: PatientPreferences | None,
        api_key: str | None,
        llm_model: str | None,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion], Dict[str, Any]] | None:
        effective_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if (not effective_key) or ("YOUR_OPENAI_API_KEY" in effective_key):
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        hint = preference_hint.model_dump(mode="json") if preference_hint else {}
        model = (llm_model or os.getenv("MAS_LLM_MODEL", "gpt-4o-mini")).strip()
        prompt = (
            "Return valid JSON only with keys: complaint_category, duration_text, "
            "extracted_constraints_text, preferences, missing_fields, clarification_questions. "
            "No diagnosis. No treatment advice. "
            "Use modalities from [in_person, phone, video], "
            "days from [Mon..Sun], periods from [morning, afternoon, evening]. "
            "clarification_questions max length 2.\n\n"
            f"Text:\n{text}\n\n"
            f"Preference hint:\n{json.dumps(hint)}"
        )

        try:
            response = OpenAI(api_key=effective_key).responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a strict JSON extractor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            raw = getattr(response, "output_text", "") or ""
            cleaned = raw.strip()
            cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
            payload = json.loads(cleaned)

            rule_preferences = self._extract_preferences(text)
            preferences = self._coerce_llm_preferences(
                payload=payload,
                text=text,
                rule_preferences=rule_preferences,
                preference_hint=preference_hint,
            )
            questions = [
                ClarificationQuestion(**item)
                for item in payload.get("clarification_questions", [])[:2]
            ]

            intake = IntakeSummary(
                run_id=run_id,
                raw_text=text,
                complaint_category=str(payload.get("complaint_category", "general_query")),
                duration_text=payload.get("duration_text"),
                extracted_constraints_text=self._normalize_constraints(payload),
                preferences=preferences,
                missing_fields=[str(item) for item in payload.get("missing_fields", [])],
            )
            return intake, questions, payload
        except Exception:
            return None

    def _normalize_constraints(self, payload: Dict[str, Any]) -> List[str]:
        raw = payload.get("extracted_constraints_text", [])
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return []

    def _coerce_llm_preferences(
        self,
        payload: Dict[str, Any],
        text: str,
        rule_preferences: PatientPreferences,
        preference_hint: PatientPreferences | None,
    ) -> PatientPreferences:
        pref_payload = payload.get("preferences", {})
        if not isinstance(pref_payload, dict):
            pref_payload = {}

        preferred_modalities = self._extract_modalities(pref_payload.get("preferred_modalities", []))
        generic_modalities = self._extract_modalities(pref_payload.get("modalities", []))
        if not preferred_modalities:
            preferred_modalities = list(generic_modalities)

        excluded_modalities = self._extract_modalities(
            pref_payload.get("excluded_modalities", pref_payload.get("blocked_modalities", []))
        )

        preferred_days = self._extract_days(pref_payload.get("preferred_days", []))
        generic_days = self._extract_days(pref_payload.get("days", []))
        if not preferred_days:
            preferred_days = list(generic_days)
        excluded_days = self._extract_days(
            pref_payload.get("excluded_days", pref_payload.get("blocked_days", []))
        )

        preferred_periods = self._extract_periods(pref_payload.get("preferred_periods", []))
        generic_periods = self._extract_periods(pref_payload.get("periods", []))
        if not preferred_periods:
            preferred_periods = list(generic_periods)
        excluded_periods = self._extract_periods(
            pref_payload.get("excluded_periods", pref_payload.get("blocked_periods", []))
        )

        # Preserve hard constraints from deterministic parser.
        excluded_modalities = self._ordered_union(
            excluded_modalities,
            list(rule_preferences.excluded_modalities),
            self.MODALITIES,
        )
        excluded_days = self._ordered_union(excluded_days, list(rule_preferences.excluded_days), self.DAY_ORDER)
        excluded_periods = self._ordered_union(
            excluded_periods,
            list(rule_preferences.excluded_periods),
            self.PERIODS,
        )

        if not preferred_modalities:
            preferred_modalities = list(rule_preferences.preferred_modalities)
        if not preferred_days:
            preferred_days = list(rule_preferences.preferred_days)
        if not preferred_periods:
            preferred_periods = list(rule_preferences.preferred_periods)

        strict_modality = self._strict_modality_from_text(text)
        if strict_modality:
            preferred_modalities = [strict_modality]
            excluded_modalities = self._ordered_union(
                excluded_modalities,
                [m for m in self.MODALITIES if m != strict_modality],
                self.MODALITIES,
            )

        # Exclusions override preferences.
        preferred_modalities = self._remove_excluded(preferred_modalities, excluded_modalities, self.MODALITIES)
        preferred_days = self._remove_excluded(preferred_days, excluded_days, self.DAY_ORDER)
        preferred_periods = self._remove_excluded(preferred_periods, excluded_periods, self.PERIODS)

        hint_horizon = preference_hint.date_horizon_days if preference_hint else DEFAULT_DATE_HORIZON_DAYS
        hint_soonest = preference_hint.soonest_weight if preference_hint else 60
        hint_flex = preference_hint.flexibility if preference_hint else None

        date_horizon = pref_payload.get("date_horizon_days", hint_horizon)
        soonest_weight = pref_payload.get("soonest_weight", hint_soonest)

        try:
            date_horizon = int(date_horizon)
        except Exception:
            date_horizon = hint_horizon
        try:
            soonest_weight = int(soonest_weight)
        except Exception:
            soonest_weight = hint_soonest

        return PatientPreferences(
            preferred_modalities=preferred_modalities,
            excluded_modalities=excluded_modalities,
            preferred_days=preferred_days,
            excluded_days=excluded_days,
            preferred_periods=preferred_periods,
            excluded_periods=excluded_periods,
            date_horizon_days=max(1, min(30, date_horizon)),
            soonest_weight=max(0, min(100, soonest_weight)),
            flexibility=hint_flex if hint_flex else PatientPreferences().flexibility,
        )

    def _ordered_union(self, left: List[str], right: List[str], order: List[str]) -> List[str]:
        merged = set(left) | set(right)
        return [item for item in order if item in merged]

    def _remove_excluded(self, preferred: List[str], excluded: List[str], order: List[str]) -> List[str]:
        preferred_set = set(preferred) - set(excluded)
        return [item for item in order if item in preferred_set]

    def _extract_modalities(self, raw: Any) -> List[str]:
        items = self._listify(raw)
        result: List[str] = []
        for item in items:
            token = item.lower().strip()
            if "phone" in token or "call" in token:
                result.append("phone")
            elif "video" in token or "zoom" in token or "teams" in token:
                result.append("video")
            elif "in person" in token or "in_person" in token or "face" in token or "f2f" in token:
                result.append("in_person")
        return sorted(set(result))

    def _extract_days(self, raw: Any) -> List[str]:
        items = self._listify(raw)
        found: set[str] = set()
        for item in items:
            lowered = self._normalize_text_for_parsing(item.lower())
            for day, pattern in self.DAY_PATTERNS.items():
                if re.search(pattern, lowered):
                    found.add(day)
        return [day for day in self.DAY_ORDER if day in found]

    def _extract_periods(self, raw: Any) -> List[str]:
        items = self._listify(raw)
        found: set[str] = set()
        for item in items:
            lowered = item.lower()
            for period, pattern in self.PERIOD_PATTERNS.items():
                if re.search(pattern, lowered):
                    found.add(period)
        return [period for period in self.PERIODS if period in found]

    def _listify(self, raw: Any) -> List[str]:
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, str):
            return [part.strip() for part in re.split(r"[,;/]", raw) if part.strip()]
        return []

    def _strict_modality_from_text(self, text: str) -> str | None:
        lowered = self._normalize_text_for_parsing(text.lower())
        strict_markers = ["has to", "must", "only", "required"]
        if not any(marker in lowered for marker in strict_markers):
            return None
        if "phone" in lowered or "call" in lowered:
            return "phone"
        if "video" in lowered or "zoom" in lowered or "teams" in lowered:
            return "video"
        if "in person" in lowered or "face to face" in lowered or "f2f" in lowered:
            return "in_person"
        return None

    def _build_intake_rule_from_text(
        self,
        run_id: str,
        merged_text: str,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion]]:
        complaint_category = self._extract_category(merged_text)
        duration_text = self._extract_duration(merged_text)
        preferences = self._extract_preferences(merged_text)
        constraints = self._extract_constraints_text(merged_text)

        missing_fields = self._missing_fields_from_preferences(preferences)
        questions = self._clarification_questions_from_missing(missing_fields)

        intake = IntakeSummary(
            run_id=run_id,
            raw_text=merged_text,
            complaint_category=complaint_category,
            duration_text=duration_text,
            extracted_constraints_text=constraints,
            preferences=preferences,
            missing_fields=missing_fields,
        )
        return intake, questions

    def _missing_fields_from_preferences(self, preferences: PatientPreferences) -> List[str]:
        missing_fields: List[str] = []

        if not preferences.preferred_modalities and not preferences.excluded_modalities:
            missing_fields.append("modality")

        has_availability_signal = any(
            [
                preferences.preferred_days,
                preferences.excluded_days,
                preferences.preferred_periods,
                preferences.excluded_periods,
            ]
        )
        if not has_availability_signal:
            missing_fields.append("availability")

        return missing_fields

    def _clarification_questions_from_missing(
        self,
        missing_fields: List[str],
    ) -> List[ClarificationQuestion]:
        questions: List[ClarificationQuestion] = []
        if "modality" in missing_fields:
            questions.append(
                ClarificationQuestion(
                    question_id="preferred_modality",
                    prompt="Do you prefer in-person, phone, or video?",
                )
            )
        if "availability" in missing_fields:
            questions.append(
                ClarificationQuestion(
                    question_id="availability",
                    prompt="Any preferred days or times, or times to avoid?",
                )
            )
        return questions[:2]

    def _merge_text_with_clarifications(self, user_text: str, clarifications: Dict[str, str]) -> str:
        if not clarifications:
            return user_text
        ordered = [f"{key}: {value}" for key, value in sorted(clarifications.items()) if value.strip()]
        return f"{user_text} {' '.join(ordered)}".strip()

    def _compose_form_text(
        self,
        reason_text: str,
        preferences: PatientPreferences,
    ) -> str:
        parts = [f"Reason: {reason_text}"]
        if preferences.preferred_modalities:
            parts.append("Preferred modalities: " + ", ".join(preferences.preferred_modalities))
        if preferences.excluded_modalities:
            parts.append("Excluded modalities: " + ", ".join(preferences.excluded_modalities))
        if preferences.preferred_days:
            parts.append("Preferred days: " + ", ".join(preferences.preferred_days))
        if preferences.excluded_days:
            parts.append("Excluded days: " + ", ".join(preferences.excluded_days))
        if preferences.preferred_periods:
            parts.append("Preferred periods: " + ", ".join(preferences.preferred_periods))
        if preferences.excluded_periods:
            parts.append("Excluded periods: " + ", ".join(preferences.excluded_periods))
        parts.append(f"Date horizon days: {preferences.date_horizon_days}")
        parts.append(f"Soonest weight: {preferences.soonest_weight}")
        return ". ".join(parts)

    def _constraints_from_preferences(self, preferences: PatientPreferences) -> List[str]:
        constraints: List[str] = []
        if preferences.excluded_modalities:
            constraints.append("excluded_modalities")
        if preferences.excluded_days:
            constraints.append("excluded_days")
        if preferences.excluded_periods:
            constraints.append("excluded_periods")
        return constraints

    def _extract_category(self, text: str) -> str:
        lowered = text.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return category
        return "general_query"

    def _extract_duration(self, text: str) -> str | None:
        match = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", text.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}"
        if "since yesterday" in text.lower():
            return "since yesterday"
        return None

    def _extract_constraints_text(self, text: str) -> List[str]:
        lowered = text.lower()
        snippets: List[str] = []
        for marker in ["no ", "not ", "avoid ", "cannot ", "can't ", "unavailable"]:
            if marker in lowered:
                snippets.append(marker.strip())
        return sorted(set(snippets))

    def _extract_preferences(self, text: str) -> PatientPreferences:
        lowered = self._normalize_text_for_parsing(text.lower())

        preferred_modalities: List[str] = []
        excluded_modalities: List[str] = []

        self._collect_modality_preferences(lowered, preferred_modalities, excluded_modalities)
        preferred_days, excluded_days = self._collect_days(lowered)
        preferred_periods, excluded_periods = self._collect_periods(lowered)

        return PatientPreferences(
            preferred_modalities=preferred_modalities,
            excluded_modalities=excluded_modalities,
            preferred_days=preferred_days,
            excluded_days=excluded_days,
            preferred_periods=preferred_periods,
            excluded_periods=excluded_periods,
            date_horizon_days=DEFAULT_DATE_HORIZON_DAYS,
            soonest_weight=60,
        )

    def _normalize_text_for_parsing(self, lowered: str) -> str:
        lowered = lowered.replace("can’t", "cannot")
        lowered = lowered.replace("can't", "cannot")
        lowered = lowered.replace("cant", "cannot")
        lowered = lowered.replace("won't", "will not")
        lowered = lowered.replace("wont", "will not")
        return re.sub(r"\s+", " ", lowered).strip()

    def _collect_modality_preferences(
        self,
        lowered: str,
        preferred_modalities: List[str],
        excluded_modalities: List[str],
    ) -> None:
        modality_patterns = {
            "phone": r"\bphone\b|\bcall\b",
            "video": r"\bvideo\b|\bzoom\b|\bteams\b",
            "in_person": r"\bin person\b|\bin-person\b|\bface to face\b|\bf2f\b",
        }
        for modality, pattern in modality_patterns.items():
            has_modality = re.search(pattern, lowered) is not None
            if not has_modality:
                continue
            negated = re.search(rf"\b(no|not|avoid|cannot)\s+(?:{pattern})", lowered) is not None
            if negated:
                excluded_modalities.append(modality)
            else:
                preferred_modalities.append(modality)

    def _collect_days(self, lowered: str) -> Tuple[List[str], List[str]]:
        preferred_days: set[str] = set()
        excluded_days: set[str] = set()

        for day, pattern in self.DAY_PATTERNS.items():
            for match in re.finditer(pattern, lowered):
                context_start = max(
                    lowered.rfind(".", 0, match.start()),
                    lowered.rfind(",", 0, match.start()),
                    lowered.rfind(";", 0, match.start()),
                    lowered.rfind("\n", 0, match.start()),
                )
                context = lowered[context_start + 1 : match.start()]

                neg_idx = self._last_cue_index(
                    context,
                    [r"\bno\b", r"\bnot\b", r"\bavoid\b", r"\bcannot\b", r"\bunavailable\b"],
                )
                pos_idx = self._last_cue_index(
                    context,
                    [
                        r"\bprefer(?:red)?\b",
                        r"\bideally\b",
                        r"\bcan do\b",
                        r"\bworks?\b",
                        r"\bavailable\b",
                        r"\bbest\b",
                    ],
                )

                if neg_idx >= 0 and neg_idx >= pos_idx:
                    excluded_days.add(day)
                else:
                    preferred_days.add(day)

        preferred_days.difference_update(excluded_days)
        ordered_preferred = [day for day in self.DAY_ORDER if day in preferred_days]
        ordered_excluded = [day for day in self.DAY_ORDER if day in excluded_days]
        return ordered_preferred, ordered_excluded

    def _collect_periods(self, lowered: str) -> Tuple[List[str], List[str]]:
        preferred_periods: List[str] = []
        excluded_periods: List[str] = []
        for period, pattern in self.PERIOD_PATTERNS.items():
            if re.search(pattern, lowered) is None:
                continue
            negated = re.search(rf"\b(no|not|avoid|cannot|unavailable)\s+(?:{pattern})", lowered)
            if negated:
                excluded_periods.append(period)
            else:
                preferred_periods.append(period)
        return preferred_periods, excluded_periods

    def _last_cue_index(self, text: str, patterns: List[str]) -> int:
        last_idx = -1
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if match.start() > last_idx:
                    last_idx = match.start()
        return last_idx
