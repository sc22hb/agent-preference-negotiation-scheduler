from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from nhs_demo.config import DEFAULT_DATE_HORIZON_DAYS
from nhs_demo.schemas import (
    ClarificationQuestion,
    DayPeriodPreference,
    ExtractorMode,
    IntakeSummary,
    PatientPreferences,
    PreferenceWeightProfile,
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
    IN_PERSON_ONLY_CATEGORIES = {"tests_monitoring"}
    IN_PERSON_ONLY_KEYWORDS = ["blood test", "bp check", "blood pressure", "vaccination"]
    QUESTION_ID_BY_FIELD: Dict[str, str] = {
        "complaint_context": "complaint_context",
        "modality": "preferred_modality",
        "availability": "availability",
        "availability_exclusion": "availability_exclusion",
        "duration": "duration_text",
    }
    CANONICAL_PROMPTS: Dict[str, str] = {
        "preferred_modality": "Do you prefer an in-person, phone, or video consultation?",
    }

    def __init__(self) -> None:
        self._last_llm_error: str | None = None

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
                clarification_answers=clarification_answers,
            )
            if llm_result is None:
                raise ValueError(
                    f"LLM extraction failed: {self._last_llm_error or 'unknown reason'}. "
                    "Check API key/model or use Form-based mode."
                )
            return llm_result[0], llm_result[1], "llm", llm_result[2]

        intake, questions = self._build_intake_rule_from_text(
            run_id=run_id,
            merged_text=merged_text,
            clarification_answers=clarification_answers,
        )
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
                clarification_answers=None,
            )
            if llm_result is None:
                raise ValueError(
                    f"LLM extraction failed: {self._last_llm_error or 'unknown reason'}. "
                    "Check API key/model or use Form-based mode."
                )
            intake_llm, _questions, llm_payload = llm_result
            intake_llm = intake_llm.model_copy(update={"missing_fields": []})
            return intake_llm, [], "llm", llm_payload

        complaint_category = self._extract_category(merged_text)
        normalized_preferences = self._apply_service_modality_policy(
            preferences,
            complaint_category=complaint_category,
            raw_text=merged_text,
        )
        intake = IntakeSummary(
            run_id=run_id,
            raw_text=merged_text,
            complaint_category=complaint_category,
            duration_text=self._extract_duration(merged_text),
            extracted_constraints_text=self._constraints_from_preferences(normalized_preferences),
            preferences=normalized_preferences,
            missing_fields=[],
        )
        return intake, [], "rule", None

    def build_relaxation_questions(
        self,
        candidate_questions: List[RelaxationQuestion],
    ) -> List[RelaxationQuestion]:
        prompt_overrides = {
            "relax_excluded_periods": "Would you allow other times of day if needed?",
            "relax_excluded_days": "Would you allow additional days if needed? You can pick specific days.",
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
        clarification_answers: Dict[str, str] | None = None,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion], Dict[str, Any]] | None:
        self._last_llm_error = None
        effective_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if (not effective_key) or ("YOUR_OPENAI_API_KEY" in effective_key):
            self._last_llm_error = "missing or placeholder OpenAI API key"
            return None

        try:
            from openai import OpenAI
        except Exception:
            OpenAI = None  # type: ignore[assignment]

        hint = preference_hint.model_dump(mode="json") if preference_hint else {}
        model = (llm_model or os.getenv("MAS_LLM_MODEL", "gpt-4o-mini")).strip()
        prompt = (
            "Return valid JSON only with keys: complaint_category, duration_text, "
            "extracted_constraints_text, preferences, missing_fields, clarification_questions. "
            "No diagnosis. No treatment advice. "
            "Use modalities from [in_person, phone, video], "
            "days from [Mon..Sun], periods from [morning, afternoon, evening]. "
            "preferences may include preferred_day_periods/excluded_day_periods as "
            "objects with keys day and period. "
            "preferences may include weight_profile with keys modality/day/period/day_period_synergy "
            "in range 0..200. "
            "clarification_questions max length 2.\n\n"
            f"Text:\n{text}\n\n"
            f"Preference hint:\n{json.dumps(hint)}"
        )
        request_input = [
            {"role": "system", "content": "You are a strict JSON extractor."},
            {"role": "user", "content": prompt},
        ]

        raw = ""
        try:
            if OpenAI is not None:
                response = OpenAI(api_key=effective_key).responses.create(
                    model=model,
                    input=request_input,
                    temperature=0,
                )
                raw = getattr(response, "output_text", "") or ""
                if not raw.strip():
                    raw = self._extract_output_text_from_http_payload(
                        self._coerce_response_to_dict(response)
                    )
            else:
                raw = self._call_openai_responses_http(
                    api_key=effective_key,
                    model=model,
                    request_input=request_input,
                ) or ""

            if not raw.strip():
                self._last_llm_error = "LLM response did not include output text"
                return None

            payload = self._extract_json_payload(raw)
            if payload is None:
                self._last_llm_error = "LLM response was not valid JSON payload"
                return None

            rule_preferences = self._extract_preferences(text)
            complaint_category = self._normalize_text_value(
                payload.get("complaint_category"),
                fallback="general_query",
            )
            preferences = self._coerce_llm_preferences(
                payload=payload,
                text=text,
                rule_preferences=rule_preferences,
                preference_hint=preference_hint,
            )
            preferences = self._apply_service_modality_policy(
                preferences,
                complaint_category=complaint_category,
                raw_text=text,
            )
            questions = self._parse_clarification_questions(payload.get("clarification_questions", []))
            llm_missing_fields = self._parse_missing_fields(payload.get("missing_fields", []))
            inferred_missing_fields = self._missing_fields_from_preferences(
                preferences,
                complaint_category=complaint_category,
                raw_text=text,
            )
            if complaint_category == "general_query":
                inferred_missing_fields.append("complaint_context")
            merged_missing_fields = self._merge_unique_fields(
                llm_missing_fields,
                inferred_missing_fields,
            )
            answered_question_ids = self._answered_clarification_ids(clarification_answers)
            questions = [q for q in questions if q.question_id not in answered_question_ids]
            has_user_clarification = bool(
                clarification_answers and any(value.strip() for value in clarification_answers.values())
            )
            fallback_questions = self._clarification_questions_from_missing(
                merged_missing_fields,
                answered_question_ids=answered_question_ids,
                existing_question_ids={q.question_id for q in questions},
            )
            questions = (questions + fallback_questions)[:2]
            if has_user_clarification and all(q.question_id in answered_question_ids for q in questions):
                questions = []

            intake = IntakeSummary(
                run_id=run_id,
                raw_text=text,
                complaint_category=complaint_category,
                duration_text=self._normalize_text_value(payload.get("duration_text"), fallback=""),
                extracted_constraints_text=self._normalize_constraints(payload),
                preferences=preferences,
                missing_fields=merged_missing_fields,
            )
            return intake, questions, self._build_normalized_llm_payload(intake, questions, payload)
        except Exception as exc:
            self._last_llm_error = str(exc)
            return None

    def _coerce_response_to_dict(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(response, "to_json"):
            try:
                parsed = json.loads(response.to_json())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _extract_json_payload(self, raw: str) -> Dict[str, Any] | None:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        for candidate in self._json_object_candidates(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _json_object_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        start = -1
        depth = 0
        in_string = False
        escaping = False

        for idx, char in enumerate(text):
            if in_string:
                if escaping:
                    escaping = False
                    continue
                if char == "\\":
                    escaping = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
                continue

            if char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : idx + 1])

        return candidates

    def _parse_clarification_questions(self, raw: Any) -> List[ClarificationQuestion]:
        if not isinstance(raw, list):
            return []

        questions: List[ClarificationQuestion] = []
        for idx, item in enumerate(raw):
            if len(questions) >= 2:
                break

            if isinstance(item, dict):
                prompt = str(item.get("prompt", "")).strip()
                question_id = str(item.get("question_id", "")).strip()
            else:
                prompt = str(item).strip()
                question_id = ""

            if not prompt:
                continue
            inferred = self._infer_question_id_from_prompt(prompt)
            if not question_id or re.match(r"^q_\d+$", question_id):
                question_id = inferred or f"q_{idx + 1}"
            prompt = self._canonical_prompt(question_id, prompt)
            questions.append(ClarificationQuestion(question_id=question_id, prompt=prompt))

        return questions

    def _infer_question_id_from_prompt(self, prompt: str) -> str | None:
        lowered = self._normalize_text_for_parsing(prompt.lower())
        if any(token in lowered for token in ["avoid", "cannot", "can't", "cant", "unavailable", "not available"]):
            if any(token in lowered for token in ["day", "days", "time", "times", "morning", "afternoon", "evening"]):
                return "availability_exclusion"
        if any(token in lowered for token in ["in-person", "in person", "phone", "video", "modality"]):
            return "preferred_modality"
        if any(token in lowered for token in ["day", "days", "time", "times", "availability", "morning", "afternoon", "evening"]):
            return "availability"
        if any(token in lowered for token in ["how long", "duration", "how many days", "how many weeks"]):
            return "duration_text"
        if any(token in lowered for token in ["what is this about", "appointment about", "main concern"]):
            return "complaint_context"
        return None

    def _parse_missing_fields(self, raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        parsed = [str(item).strip() for item in raw if str(item).strip()]
        return self._merge_unique_fields(parsed)

    def _answered_clarification_ids(self, clarification_answers: Dict[str, str] | None) -> set[str]:
        if not clarification_answers:
            return set()
        return {
            key.strip()
            for key, value in clarification_answers.items()
            if key.strip() and value.strip()
        }

    def _merge_unique_fields(self, *groups: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                if item in seen:
                    continue
                seen.add(item)
                ordered.append(item)
        return ordered

    def _build_normalized_llm_payload(
        self,
        intake: IntakeSummary,
        questions: List[ClarificationQuestion],
        raw_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "complaint_category": intake.complaint_category,
            "duration_text": intake.duration_text or "",
            "extracted_constraints_text": intake.extracted_constraints_text,
            "preferences": intake.preferences.model_dump(mode="json"),
            "missing_fields": intake.missing_fields,
            "clarification_questions": [item.model_dump(mode="json") for item in questions],
            "raw_llm_output": raw_payload,
        }

    def _call_openai_responses_http(
        self,
        api_key: str,
        model: str,
        request_input: List[Dict[str, str]],
    ) -> str | None:
        request_body = {
            "model": model,
            "input": request_input,
            "temperature": 0,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return self._extract_output_text_from_http_payload(payload)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

    def _extract_output_text_from_http_payload(self, payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts: List[str] = []
        output_items = payload.get("output", [])
        if not isinstance(output_items, list):
            return ""

        for item in output_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)

        return "\n".join(parts).strip()

    def _normalize_constraints(self, payload: Dict[str, Any]) -> List[str]:
        raw = payload.get("extracted_constraints_text", [])
        if isinstance(raw, str):
            cleaned = raw.strip()
            return [cleaned] if cleaned else []
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return []

    def _normalize_text_value(self, raw: Any, fallback: str) -> str:
        if raw is None:
            return fallback
        value = str(raw).strip()
        if not value or value.lower() in {"none", "null", "n/a"}:
            return fallback
        return value

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

        preferred_day_periods = self._extract_day_period_pairs(
            pref_payload.get("preferred_day_periods", pref_payload.get("day_periods", []))
        )
        excluded_day_periods = self._extract_day_period_pairs(
            pref_payload.get("excluded_day_periods", pref_payload.get("blocked_day_periods", []))
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
        excluded_day_periods = self._merge_day_period_pairs(
            excluded_day_periods,
            list(rule_preferences.excluded_day_periods),
        )

        if not preferred_modalities:
            preferred_modalities = list(rule_preferences.preferred_modalities)
        if not preferred_days:
            preferred_days = list(rule_preferences.preferred_days)
        if not preferred_periods:
            preferred_periods = list(rule_preferences.preferred_periods)
        if not preferred_day_periods:
            preferred_day_periods = list(rule_preferences.preferred_day_periods)

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
        preferred_day_periods = self._remove_excluded_day_period_pairs(
            preferred_day_periods,
            excluded_day_periods,
        )

        hint_horizon = preference_hint.date_horizon_days if preference_hint else DEFAULT_DATE_HORIZON_DAYS
        hint_soonest = preference_hint.soonest_weight if preference_hint else 60
        hint_flex = preference_hint.flexibility if preference_hint else None
        hint_weight_profile = (
            preference_hint.weight_profile if preference_hint else PreferenceWeightProfile()
        )

        date_horizon = pref_payload.get("date_horizon_days", hint_horizon)
        soonest_weight = pref_payload.get("soonest_weight", hint_soonest)
        weight_profile = self._extract_weight_profile(
            pref_payload.get("weight_profile", {}),
            hint_weight_profile,
            text,
        )

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
            preferred_day_periods=preferred_day_periods,
            excluded_day_periods=excluded_day_periods,
            date_horizon_days=max(1, min(30, date_horizon)),
            soonest_weight=max(0, min(100, soonest_weight)),
            weight_profile=weight_profile,
            flexibility=hint_flex if hint_flex else PatientPreferences().flexibility,
        )

    def _ordered_union(self, left: List[str], right: List[str], order: List[str]) -> List[str]:
        merged = set(left) | set(right)
        return [item for item in order if item in merged]

    def _remove_excluded(self, preferred: List[str], excluded: List[str], order: List[str]) -> List[str]:
        preferred_set = set(preferred) - set(excluded)
        return [item for item in order if item in preferred_set]

    def _extract_day_period_pairs(self, raw: Any) -> List[DayPeriodPreference]:
        if not isinstance(raw, list):
            return []

        pairs: List[DayPeriodPreference] = []
        for item in raw:
            if isinstance(item, dict):
                days = self._extract_days([str(item.get("day", ""))])
                periods = self._extract_periods([str(item.get("period", ""))])
            else:
                item_text = str(item)
                days = self._extract_days([item_text])
                periods = self._extract_periods([item_text])

            for day in days:
                for period in periods:
                    pairs.append(DayPeriodPreference(day=day, period=period))
        return self._dedupe_day_period_pairs(pairs)

    def _dedupe_day_period_pairs(self, pairs: List[DayPeriodPreference]) -> List[DayPeriodPreference]:
        seen: set[tuple[str, str]] = set()
        ordered: List[DayPeriodPreference] = []
        day_idx = {day: idx for idx, day in enumerate(self.DAY_ORDER)}
        period_idx = {period: idx for idx, period in enumerate(self.PERIODS)}
        for pair in sorted(pairs, key=lambda item: (day_idx[item.day], period_idx[item.period])):
            key = (pair.day, pair.period)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(pair)
        return ordered

    def _merge_day_period_pairs(
        self,
        left: List[DayPeriodPreference],
        right: List[DayPeriodPreference],
    ) -> List[DayPeriodPreference]:
        return self._dedupe_day_period_pairs(left + right)

    def _remove_excluded_day_period_pairs(
        self,
        preferred: List[DayPeriodPreference],
        excluded: List[DayPeriodPreference],
    ) -> List[DayPeriodPreference]:
        excluded_keys = {(pair.day, pair.period) for pair in excluded}
        filtered = [pair for pair in preferred if (pair.day, pair.period) not in excluded_keys]
        return self._dedupe_day_period_pairs(filtered)

    def _extract_weight_profile(
        self,
        raw: Any,
        hint: PreferenceWeightProfile,
        text: str,
    ) -> PreferenceWeightProfile:
        payload = raw if isinstance(raw, dict) else {}

        def coerce(name: str, fallback: int) -> int:
            value = payload.get(name, fallback)
            try:
                parsed = int(value)
            except Exception:
                parsed = fallback
            return max(0, min(200, parsed))

        profile = PreferenceWeightProfile(
            modality=coerce("modality", hint.modality),
            day=coerce("day", hint.day),
            period=coerce("period", hint.period),
            day_period_synergy=coerce("day_period_synergy", hint.day_period_synergy),
        )
        return self._apply_text_weight_cues(profile, text)

    def _apply_text_weight_cues(
        self,
        profile: PreferenceWeightProfile,
        text: str,
    ) -> PreferenceWeightProfile:
        lowered = self._normalize_text_for_parsing(text.lower())

        # Strong language boosts importance of temporal and modality matching in scoring.
        strict_markers = ["must", "has to", "only", "required", "urgent", "asap", "soon"]
        if any(marker in lowered for marker in strict_markers):
            profile = profile.model_copy(
                update={
                    "modality": min(200, profile.modality + 20),
                    "day": min(200, profile.day + 15),
                    "period": min(200, profile.period + 20),
                    "day_period_synergy": min(200, profile.day_period_synergy + 25),
                }
            )
        return profile

    def _extract_modalities(self, raw: Any) -> List[str]:
        items = self._listify(raw)
        result: List[str] = []
        for item in items:
            token = self._normalize_text_for_parsing(item.lower().strip())
            if "phone" in token or "call" in token:
                result.append("phone")
            elif "video" in token or "zoom" in token or "teams" in token:
                result.append("video")
            elif (
                "in person" in token
                or "in_person" in token
                or "inperson" in token
                or token == "person"
                or "face" in token
                or "f2f" in token
            ):
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
        clarification_answers: Dict[str, str] | None = None,
    ) -> Tuple[IntakeSummary, List[ClarificationQuestion]]:
        complaint_category = self._extract_category(merged_text)
        duration_text = self._extract_duration(merged_text)
        preferences = self._extract_preferences(merged_text)
        preferences = self._apply_service_modality_policy(
            preferences,
            complaint_category=complaint_category,
            raw_text=merged_text,
        )
        constraints = self._extract_constraints_text(merged_text)

        missing_fields = self._missing_fields_from_preferences(
            preferences,
            complaint_category=complaint_category,
            raw_text=merged_text,
        )
        if complaint_category == "general_query":
            missing_fields = self._merge_unique_fields(["complaint_context"], missing_fields)
        answered_question_ids = self._answered_clarification_ids(clarification_answers)
        questions = self._clarification_questions_from_missing(
            missing_fields,
            answered_question_ids=answered_question_ids,
            existing_question_ids=set(),
        )

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

    def _missing_fields_from_preferences(
        self,
        preferences: PatientPreferences,
        complaint_category: str | None = None,
        raw_text: str = "",
    ) -> List[str]:
        missing_fields: List[str] = []

        in_person_only = self._is_in_person_only_context(complaint_category, raw_text)
        if not in_person_only and not preferences.preferred_modalities and not preferences.excluded_modalities:
            missing_fields.append("modality")

        has_availability_signal = any(
            [
                preferences.preferred_days,
                preferences.excluded_days,
                preferences.preferred_periods,
                preferences.excluded_periods,
                preferences.preferred_day_periods,
                preferences.excluded_day_periods,
            ]
        )
        if not has_availability_signal:
            missing_fields.append("availability")

        return missing_fields

    def _apply_service_modality_policy(
        self,
        preferences: PatientPreferences,
        complaint_category: str | None,
        raw_text: str,
    ) -> PatientPreferences:
        if not self._is_in_person_only_context(complaint_category, raw_text):
            return preferences
        return preferences.model_copy(
            update={
                "preferred_modalities": ["in_person"],
                "excluded_modalities": ["phone", "video"],
            }
        )

    def _is_in_person_only_context(self, complaint_category: str | None, raw_text: str) -> bool:
        if complaint_category in self.IN_PERSON_ONLY_CATEGORIES:
            return True
        lowered = self._normalize_text_for_parsing(raw_text.lower())
        return any(keyword in lowered for keyword in self.IN_PERSON_ONLY_KEYWORDS)

    def _clarification_questions_from_missing(
        self,
        missing_fields: List[str],
        answered_question_ids: set[str] | None = None,
        existing_question_ids: set[str] | None = None,
    ) -> List[ClarificationQuestion]:
        answered = answered_question_ids or set()
        existing = existing_question_ids or set()
        questions: List[ClarificationQuestion] = []

        for field in missing_fields:
            question = self._question_for_missing_field(field)
            if question is None:
                continue
            if question.question_id in answered or question.question_id in existing:
                continue
            questions.append(question)
            if len(questions) == 2:
                break

        return questions[:2]

    def _question_for_missing_field(self, field: str) -> ClarificationQuestion | None:
        if field == "complaint_context":
            return ClarificationQuestion(
                question_id=self.QUESTION_ID_BY_FIELD[field],
                prompt="Could you briefly tell me what this appointment is about?",
            )
        if field == "modality":
            return ClarificationQuestion(
                question_id=self.QUESTION_ID_BY_FIELD[field],
                prompt=self.CANONICAL_PROMPTS["preferred_modality"],
            )
        if field == "availability":
            return ClarificationQuestion(
                question_id=self.QUESTION_ID_BY_FIELD[field],
                prompt="What days or times work best, and are there any you cannot do?",
            )
        if field == "duration":
            return ClarificationQuestion(
                question_id=self.QUESTION_ID_BY_FIELD[field],
                prompt="How long has this been going on?",
            )
        return None

    def _canonical_prompt(self, question_id: str, prompt: str) -> str:
        return self.CANONICAL_PROMPTS.get(question_id, prompt)

    def _merge_text_with_clarifications(self, user_text: str, clarifications: Dict[str, str]) -> str:
        if not clarifications:
            return user_text
        ordered: List[str] = []
        for key, value in sorted(clarifications.items()):
            cleaned_value = value.strip()
            if not cleaned_value:
                continue
            normalized_key = key.strip()
            lowered_value = self._normalize_text_for_parsing(cleaned_value.lower())
            has_negation = any(
                token in lowered_value for token in [" no ", " not ", " avoid ", " cannot ", " unavailable "]
            ) or lowered_value.startswith(("no ", "not ", "avoid ", "cannot ", "unavailable "))

            if normalized_key == "availability_exclusion":
                if has_negation:
                    ordered.append(cleaned_value)
                else:
                    ordered.append(f"avoid {cleaned_value}")
                continue

            if re.match(r"^q_\d+$", key.strip()):
                ordered.append(cleaned_value)
            else:
                ordered.append(f"{key}: {cleaned_value}")
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
        if preferences.preferred_day_periods:
            pairs = [f"{pair.day}-{pair.period}" for pair in preferences.preferred_day_periods]
            parts.append("Preferred day-periods: " + ", ".join(pairs))
        if preferences.excluded_day_periods:
            pairs = [f"{pair.day}-{pair.period}" for pair in preferences.excluded_day_periods]
            parts.append("Excluded day-periods: " + ", ".join(pairs))
        parts.append(f"Date horizon days: {preferences.date_horizon_days}")
        parts.append(f"Soonest weight: {preferences.soonest_weight}")
        parts.append(
            "Weight profile: "
            f"modality={preferences.weight_profile.modality}, "
            f"day={preferences.weight_profile.day}, "
            f"period={preferences.weight_profile.period}, "
            f"synergy={preferences.weight_profile.day_period_synergy}"
        )
        return ". ".join(parts)

    def _constraints_from_preferences(self, preferences: PatientPreferences) -> List[str]:
        constraints: List[str] = []
        if preferences.excluded_modalities:
            constraints.append("excluded_modalities")
        if preferences.excluded_days:
            constraints.append("excluded_days")
        if preferences.excluded_periods:
            constraints.append("excluded_periods")
        if preferences.excluded_day_periods:
            constraints.append("excluded_day_periods")
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
        preferred_day_periods, excluded_day_periods = self._collect_day_period_preferences(lowered)
        weight_profile = self._apply_text_weight_cues(PreferenceWeightProfile(), lowered)

        return PatientPreferences(
            preferred_modalities=preferred_modalities,
            excluded_modalities=excluded_modalities,
            preferred_days=preferred_days,
            excluded_days=excluded_days,
            preferred_periods=preferred_periods,
            excluded_periods=excluded_periods,
            preferred_day_periods=preferred_day_periods,
            excluded_day_periods=excluded_day_periods,
            date_horizon_days=DEFAULT_DATE_HORIZON_DAYS,
            soonest_weight=60,
            weight_profile=weight_profile,
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
                    window = lowered[max(0, match.start() - 20) : min(len(lowered), match.end() + 30)]
                    period_in_window = any(
                        re.search(period_pattern, window) is not None
                        for period_pattern in self.PERIOD_PATTERNS.values()
                    )
                    is_all_day = "all day" in window
                    if period_in_window and not is_all_day:
                        continue
                    excluded_days.add(day)
                else:
                    preferred_days.add(day)

        preferred_days.difference_update(excluded_days)
        ordered_preferred = [day for day in self.DAY_ORDER if day in preferred_days]
        ordered_excluded = [day for day in self.DAY_ORDER if day in excluded_days]
        return ordered_preferred, ordered_excluded

    def _collect_day_period_preferences(
        self,
        lowered: str,
    ) -> Tuple[List[DayPeriodPreference], List[DayPeriodPreference]]:
        preferred: List[DayPeriodPreference] = []
        excluded: List[DayPeriodPreference] = []
        clauses = re.split(r"[,;.\n]|\bbut\b", lowered)
        negation_pattern = re.compile(r"\b(no|not|avoid|cannot|unavailable)\b")

        for clause in clauses:
            clause = clause.strip()
            if not clause or "all day" in clause:
                continue

            days = [day for day, pattern in self.DAY_PATTERNS.items() if re.search(pattern, clause)]
            periods = [period for period, pattern in self.PERIOD_PATTERNS.items() if re.search(pattern, clause)]
            if not days or not periods:
                continue

            target = excluded if negation_pattern.search(clause) else preferred
            for day in days:
                for period in periods:
                    target.append(DayPeriodPreference(day=day, period=period))

        return self._dedupe_day_period_pairs(preferred), self._dedupe_day_period_pairs(excluded)

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
