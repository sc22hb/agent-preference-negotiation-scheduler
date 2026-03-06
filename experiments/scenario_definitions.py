from __future__ import annotations

from typing import Any, Dict, List

WORKING_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ALL_PERIODS = ["morning", "afternoon", "evening"]


def _scenario(
    scenario_id: str,
    title: str,
    raw_text: str,
    complaint_category: str,
    duration_text: str,
    preferences: Dict[str, Any],
    routing: Dict[str, Any] | None = None,
    urgency_flags: Dict[str, bool] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "title": title,
        "intake": {
            "raw_text": raw_text,
            "complaint_category": complaint_category,
            "duration_text": duration_text,
            "urgency_flags": urgency_flags or {"same_day": False, "red_flag": False},
            "extracted_constraints_text": list(
                extra.get("extracted_constraints_text", []) if extra else []
            ),
        },
        "preferences": preferences,
    }
    if routing is not None:
        payload["routing"] = routing
    if extra:
        payload.update(extra)
    return payload


def _routing(
    service_type: str,
    appointment_length_minutes: int,
    allowed_modalities: List[str],
    urgency_band: str,
    rule_hit: str,
    explanation: str,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    return {
        "service_type": service_type,
        "appointment_length_minutes": appointment_length_minutes,
        "allowed_modalities": allowed_modalities,
        "urgency_band": urgency_band,
        "confidence": confidence,
        "rule_hit": rule_hit,
        "explanation": explanation,
    }


def _preferences(
    *,
    preferred_modalities: List[str] | None = None,
    excluded_modalities: List[str] | None = None,
    preferred_days: List[str] | None = None,
    excluded_days: List[str] | None = None,
    preferred_periods: List[str] | None = None,
    excluded_periods: List[str] | None = None,
    preferred_day_periods: List[Dict[str, str]] | None = None,
    excluded_day_periods: List[Dict[str, str]] | None = None,
    date_horizon_days: int = 10,
    soonest_weight: int = 60,
    weight_profile: Dict[str, int] | None = None,
    flexibility: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    return {
        "preferred_modalities": preferred_modalities or [],
        "excluded_modalities": excluded_modalities or [],
        "preferred_days": preferred_days or [],
        "excluded_days": excluded_days or [],
        "preferred_periods": preferred_periods or [],
        "excluded_periods": excluded_periods or [],
        "preferred_day_periods": preferred_day_periods or [],
        "excluded_day_periods": excluded_day_periods or [],
        "date_horizon_days": date_horizon_days,
        "soonest_weight": soonest_weight,
        "weight_profile": weight_profile
        or {
            "modality": 100,
            "day": 100,
            "period": 100,
            "day_period_synergy": 100,
        },
        "flexibility": flexibility
        or {
            "allow_time_relax": True,
            "allow_modality_relax": True,
            "allow_date_horizon_relax": True,
        },
    }


DETERMINISM_SCENARIOS: List[Dict[str, Any]] = [
    _scenario(
        scenario_id="det_repeat_rx_phone",
        title="Repeat prescription with phone preference",
        raw_text="Repeat prescription review. Phone is preferred and Friday morning would suit best.",
        complaint_category="prescription_admin",
        duration_text="Routine repeat medication request",
        preferences=_preferences(
            preferred_modalities=["phone"],
            preferred_days=["Fri"],
            preferred_periods=["morning"],
            preferred_day_periods=[{"day": "Fri", "period": "morning"}],
            date_horizon_days=10,
            soonest_weight=70,
        ),
        routing=_routing(
            service_type="Pharmacist",
            appointment_length_minutes=10,
            allowed_modalities=["phone", "in_person"],
            urgency_band="routine",
            rule_hit="pharmacist_repeat_prescription",
            explanation="Deterministic experiment scenario for repeat prescription requests.",
        ),
        extra={
            "extracted_constraints_text": [
                "phone preferred",
                "Friday morning preferred",
                "routine request",
            ]
        },
    ),
    _scenario(
        scenario_id="det_blood_test_after_wed",
        title="Blood test after mid-week",
        raw_text="Blood test booking needed. In person only. Cannot attend Monday, Tuesday or Wednesday.",
        complaint_category="tests_monitoring",
        duration_text="Routine blood monitoring",
        preferences=_preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            excluded_days=["Mon", "Tue", "Wed"],
            preferred_periods=["morning"],
            date_horizon_days=12,
            soonest_weight=55,
        ),
        routing=_routing(
            service_type="Nurse",
            appointment_length_minutes=15,
            allowed_modalities=["in_person"],
            urgency_band="routine",
            rule_hit="nurse_blood_test",
            explanation="Deterministic experiment scenario for in-person blood testing.",
        ),
        extra={
            "extracted_constraints_text": [
                "in person only",
                "cannot attend Monday to Wednesday",
            ]
        },
    ),
    _scenario(
        scenario_id="det_resp_video_wed_morning",
        title="Respiratory review with single-session preference",
        raw_text="Persistent cough for two weeks. Wednesday morning video would be ideal.",
        complaint_category="respiratory",
        duration_text="Two-week cough review",
        preferences=_preferences(
            preferred_modalities=["video"],
            preferred_days=["Wed"],
            preferred_periods=["morning"],
            preferred_day_periods=[{"day": "Wed", "period": "morning"}],
            date_horizon_days=10,
            soonest_weight=65,
        ),
        routing=_routing(
            service_type="GP",
            appointment_length_minutes=20,
            allowed_modalities=["in_person", "phone", "video"],
            urgency_band="soon",
            rule_hit="gp_soon_urinary_or_respiratory",
            explanation="Deterministic experiment scenario for a non-red-flag respiratory complaint.",
        ),
        extra={
            "extracted_constraints_text": [
                "video preferred",
                "Wednesday morning preferred",
            ]
        },
    ),
    _scenario(
        scenario_id="det_admin_remote_afternoon",
        title="Administrative certificate request",
        raw_text="Need a fit note discussion. Remote appointment preferred, afternoons only.",
        complaint_category="administrative",
        duration_text="Administrative note request",
        preferences=_preferences(
            preferred_modalities=["phone", "video"],
            preferred_periods=["afternoon"],
            excluded_periods=["morning", "evening"],
            date_horizon_days=7,
            soonest_weight=40,
        ),
        routing=_routing(
            service_type="Admin",
            appointment_length_minutes=10,
            allowed_modalities=["phone", "video"],
            urgency_band="routine",
            rule_hit="admin_documents",
            explanation="Deterministic experiment scenario for administrative work.",
        ),
        extra={
            "extracted_constraints_text": [
                "remote preferred",
                "afternoons only",
            ]
        },
    ),
    _scenario(
        scenario_id="det_unscheduled_weekdays_blocked",
        title="Unschedulable initial GP case",
        raw_text="Routine GP review, but no Monday, Tuesday, Wednesday, Thursday or Friday appointments are possible.",
        complaint_category="general_review",
        duration_text="Routine review",
        preferences=_preferences(
            excluded_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            date_horizon_days=7,
            soonest_weight=50,
        ),
        routing=_routing(
            service_type="GP",
            appointment_length_minutes=15,
            allowed_modalities=["in_person", "phone", "video"],
            urgency_band="routine",
            rule_hit="gp_default",
            explanation="Deterministic experiment scenario that should fail without negotiation.",
        ),
        extra={
            "extracted_constraints_text": [
                "no weekday appointments available",
            ]
        },
    ),
    _scenario(
        scenario_id="det_urinary_phone_fast",
        title="Urinary telephone review within one week",
        raw_text="Urinary symptoms review. Phone is easiest and Tuesday afternoon would help within the next 6 days.",
        complaint_category="urinary",
        duration_text="Recent urinary symptoms",
        preferences=_preferences(
            preferred_modalities=["phone"],
            preferred_days=["Tue"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Tue", "period": "afternoon"}],
            date_horizon_days=6,
            soonest_weight=72,
        ),
        routing=_routing(
            service_type="GP",
            appointment_length_minutes=15,
            allowed_modalities=["in_person", "phone", "video"],
            urgency_band="soon",
            rule_hit="gp_soon_urinary_or_respiratory",
            explanation="Deterministic experiment scenario for a short-horizon urinary review.",
        ),
        extra={
            "extracted_constraints_text": [
                "phone preferred",
                "Tuesday afternoon preferred",
                "within six days",
            ]
        },
    ),
    _scenario(
        scenario_id="det_gp_in_person_tue_afternoon",
        title="General review with in-person Tuesday preference",
        raw_text="Routine GP review. In person would be best, ideally Tuesday afternoon, within the next 9 days.",
        complaint_category="general_review",
        duration_text="Routine review",
        preferences=_preferences(
            preferred_modalities=["in_person"],
            preferred_days=["Tue"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Tue", "period": "afternoon"}],
            date_horizon_days=9,
            soonest_weight=58,
        ),
        routing=_routing(
            service_type="GP",
            appointment_length_minutes=15,
            allowed_modalities=["in_person", "phone", "video"],
            urgency_band="routine",
            rule_hit="gp_default",
            explanation="Deterministic experiment scenario for a routine GP review with a single preferred session.",
        ),
        extra={
            "extracted_constraints_text": [
                "in person preferred",
                "Tuesday afternoon preferred",
            ]
        },
    ),
    _scenario(
        scenario_id="det_monitoring_in_person_thu_morning",
        title="Monitoring appointment with fixed in-person attendance",
        raw_text="Monitoring blood pressure review. It must be in person, Thursday morning if possible, within the next 14 days.",
        complaint_category="tests_monitoring",
        duration_text="Routine monitoring review",
        preferences=_preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            preferred_days=["Thu"],
            preferred_periods=["morning"],
            preferred_day_periods=[{"day": "Thu", "period": "morning"}],
            date_horizon_days=14,
            soonest_weight=52,
        ),
        routing=_routing(
            service_type="Nurse",
            appointment_length_minutes=15,
            allowed_modalities=["in_person"],
            urgency_band="routine",
            rule_hit="nurse_monitoring",
            explanation="Deterministic experiment scenario for in-person nurse-led monitoring.",
        ),
        extra={
            "extracted_constraints_text": [
                "in person only",
                "Thursday morning preferred",
            ]
        },
    ),
]


TIE_BREAK_SCENARIO: Dict[str, Any] = {
    "scenario_id": "tie_break_same_utility",
    "title": "Synthetic tie-breaking pool",
    "intake": {
        "raw_text": "Synthetic GP tie-break scenario.",
        "complaint_category": "general_review",
        "duration_text": "Synthetic experiment",
        "urgency_flags": {"same_day": False, "red_flag": False},
        "extracted_constraints_text": ["synthetic tie-break case"],
    },
    "preferences": _preferences(
        date_horizon_days=10,
        soonest_weight=0,
    ),
    "routing": _routing(
        service_type="GP",
        appointment_length_minutes=15,
        allowed_modalities=["phone", "video", "in_person"],
        urgency_band="routine",
        rule_hit="synthetic_tie_break",
        explanation="Synthetic tie-breaking scenario with equal utility across slots.",
    ),
}


def allocation_quality_profiles() -> List[Dict[str, Any]]:
    complaints = [
        {
            "reason_code": "repeat_prescription",
            "raw_text": "Repeat prescription review for stable medication.",
            "complaint_category": "prescription_admin",
            "duration_text": "Routine repeat prescription request",
        },
        {
            "reason_code": "blood_test",
            "raw_text": "Blood test follow-up for monitoring.",
            "complaint_category": "tests_monitoring",
            "duration_text": "Routine monitoring blood test",
        },
        {
            "reason_code": "persistent_cough",
            "raw_text": "Persistent cough for two weeks, not worsening acutely.",
            "complaint_category": "respiratory",
            "duration_text": "Two-week cough",
        },
        {
            "reason_code": "admin_request",
            "raw_text": "Need a fit note and administrative letter.",
            "complaint_category": "administrative",
            "duration_text": "Administrative documentation request",
        },
        {
            "reason_code": "general_review",
            "raw_text": "Routine GP review for ongoing tiredness.",
            "complaint_category": "general_review",
            "duration_text": "Routine review",
        },
        {
            "reason_code": "uti_symptoms",
            "raw_text": "UTI symptoms with no red-flag features today.",
            "complaint_category": "urinary",
            "duration_text": "Recent urinary symptoms",
        },
    ]

    strict_templates = [
        _preferences(
            preferred_modalities=["phone"],
            preferred_days=["Mon", "Wed", "Fri"],
            preferred_periods=["morning"],
            excluded_periods=["afternoon", "evening"],
            date_horizon_days=7,
            soonest_weight=75,
            weight_profile={"modality": 140, "day": 130, "period": 140, "day_period_synergy": 150},
        ),
        _preferences(
            preferred_modalities=["video"],
            preferred_days=["Tue", "Thu"],
            preferred_periods=["afternoon"],
            excluded_days=["Mon", "Fri"],
            excluded_periods=["morning", "evening"],
            date_horizon_days=6,
            soonest_weight=70,
            weight_profile={"modality": 145, "day": 135, "period": 145, "day_period_synergy": 150},
        ),
        _preferences(
            preferred_modalities=["phone", "video"],
            preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            preferred_periods=["morning"],
            excluded_days=["Wed"],
            excluded_periods=["afternoon", "evening"],
            date_horizon_days=5,
            soonest_weight=80,
            weight_profile={"modality": 135, "day": 125, "period": 145, "day_period_synergy": 150},
        ),
        _preferences(
            preferred_modalities=["video"],
            preferred_days=["Thu"],
            preferred_periods=["afternoon"],
            excluded_days=["Mon", "Tue", "Wed", "Fri"],
            excluded_periods=["morning", "evening"],
            preferred_day_periods=[
                {"day": "Thu", "period": "afternoon"},
            ],
            date_horizon_days=3,
            soonest_weight=68,
            weight_profile={"modality": 150, "day": 130, "period": 135, "day_period_synergy": 155},
        ),
        _preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            preferred_days=["Tue", "Thu"],
            excluded_days=["Mon", "Wed", "Fri"],
            preferred_periods=["morning"],
            date_horizon_days=4,
            soonest_weight=82,
            weight_profile={"modality": 155, "day": 140, "period": 145, "day_period_synergy": 150},
        ),
        _preferences(
            preferred_modalities=["phone"],
            preferred_days=["Wed"],
            excluded_days=["Mon", "Tue", "Thu", "Fri"],
            preferred_periods=["afternoon"],
            excluded_periods=["morning", "evening"],
            preferred_day_periods=[{"day": "Wed", "period": "afternoon"}],
            date_horizon_days=4,
            soonest_weight=78,
            weight_profile={"modality": 145, "day": 145, "period": 140, "day_period_synergy": 160},
        ),
    ]

    flexible_templates = [
        _preferences(
            preferred_modalities=["phone", "video", "in_person"],
            preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            preferred_periods=["morning", "afternoon"],
            date_horizon_days=12,
            soonest_weight=55,
        ),
        _preferences(
            preferred_modalities=["in_person", "phone"],
            preferred_days=["Tue", "Thu"],
            preferred_periods=["morning", "afternoon"],
            date_horizon_days=14,
            soonest_weight=50,
        ),
        _preferences(
            preferred_modalities=["phone", "video"],
            preferred_days=["Mon", "Wed", "Fri"],
            preferred_periods=["morning", "afternoon"],
            excluded_periods=["evening"],
            date_horizon_days=11,
            soonest_weight=52,
        ),
        _preferences(
            preferred_modalities=["in_person", "phone", "video"],
            preferred_day_periods=[
                {"day": "Wed", "period": "morning"},
                {"day": "Fri", "period": "afternoon"},
            ],
            date_horizon_days=10,
            soonest_weight=58,
        ),
        _preferences(
            preferred_modalities=["phone", "video"],
            preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            preferred_periods=["afternoon", "evening"],
            date_horizon_days=16,
            soonest_weight=46,
        ),
        _preferences(
            preferred_modalities=["in_person", "phone"],
            preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            preferred_periods=["morning", "afternoon", "evening"],
            excluded_day_periods=[{"day": "Wed", "period": "morning"}],
            date_horizon_days=18,
            soonest_weight=48,
        ),
    ]

    profiles: List[Dict[str, Any]] = []
    profile_index = 1
    for group, templates in (("A", strict_templates), ("B", flexible_templates)):
        for complaint in complaints:
            for template_index, template in enumerate(templates, start=1):
                profile_id = f"{group}{profile_index:02d}"
                extracted_constraints = [
                    f"group_{group.lower()}",
                    f"template_{template_index}",
                    "working-hours constrained" if group == "A" else "flexible availability",
                ]
                if complaint["reason_code"] == "blood_test":
                    template = {
                        **template,
                        "preferred_modalities": ["in_person"],
                        "excluded_modalities": ["phone", "video"],
                    }
                profiles.append(
                    _scenario(
                        scenario_id=profile_id,
                        title=f"Allocation quality profile {profile_id}",
                        raw_text=complaint["raw_text"],
                        complaint_category=complaint["complaint_category"],
                        duration_text=complaint["duration_text"],
                        preferences=template,
                        urgency_flags={"same_day": False, "red_flag": False},
                        extra={
                            "group": group,
                            "reason_code": complaint["reason_code"],
                            "extracted_constraints_text": extracted_constraints,
                        },
                    )
                )
                profile_index += 1
    return profiles


def negotiation_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    time_locked_cases = [
        ("neg_time_gp_01", "Routine GP review", "general_review", "Routine review", "general review"),
        ("neg_time_gp_02", "Persistent cough review", "respiratory", "Two-week cough", "persistent cough"),
        ("neg_time_rx_03", "Repeat prescription review", "prescription_admin", "Medication review", "repeat prescription"),
        ("neg_time_admin_04", "Fit note request", "administrative", "Admin request", "fit note"),
        ("neg_time_uti_05", "UTI symptoms review", "urinary", "Recent urinary symptoms", "uti symptoms"),
        ("neg_time_gp_06", "Medication follow-up", "general_review", "Routine follow-up", "routine check"),
        ("neg_time_gp_07", "Asthma review", "respiratory", "Review", "persistent cough"),
        ("neg_time_admin_08", "Letter request", "administrative", "Admin letter", "letter"),
        ("neg_time_rx_25", "Medication request", "prescription_admin", "Medication request", "repeat meds"),
        ("neg_time_gp_26", "Routine medication review", "general_review", "Routine review", "routine gp review"),
        ("neg_time_resp_27", "Wheeze review", "respiratory", "Short respiratory review", "wheeze review"),
        ("neg_time_admin_28", "Certificate update", "administrative", "Admin update", "certificate renewal"),
    ]
    for scenario_id, title, category, duration_text, raw_text in time_locked_cases:
        scenarios.append(
            _scenario(
                scenario_id=scenario_id,
                title=title,
                raw_text=f"{raw_text}. I cannot do mornings, afternoons or evenings at first pass.",
                complaint_category=category,
                duration_text=duration_text,
                preferences=_preferences(
                    excluded_periods=list(ALL_PERIODS),
                    date_horizon_days=10,
                    soonest_weight=60,
                ),
                extra={
                    "negotiation_family": "time_relax",
                    "extracted_constraints_text": ["all periods initially excluded"],
                },
            )
        )

    day_locked_cases = [
        ("neg_day_gp_09", "General review weekday exclusion", "general_review", "Routine review", "general review"),
        ("neg_day_rx_10", "Repeat medication review", "prescription_admin", "Medication review", "repeat prescription"),
        ("neg_day_admin_11", "Admin document", "administrative", "Admin request", "fit note"),
        ("neg_day_uti_12", "UTI review", "urinary", "Recent urinary symptoms", "uti symptoms"),
        ("neg_day_resp_13", "Persistent cough review", "respiratory", "Two-week cough", "persistent cough"),
        ("neg_day_gp_14", "Routine GP query", "general_review", "Routine query", "routine check"),
        ("neg_day_rx_15", "Prescription amendment", "prescription_admin", "Medication request", "repeat prescription"),
        ("neg_day_admin_16", "Certificate request", "administrative", "Admin certificate", "certificate"),
        ("neg_day_resp_29", "Asthma review weekday exclusion", "respiratory", "Review", "asthma review"),
        ("neg_day_gp_30", "General review no weekdays", "general_review", "Routine review", "routine gp review"),
        ("neg_day_admin_31", "Letter discussion no weekdays", "administrative", "Admin request", "admin letter"),
        ("neg_day_uti_32", "Urinary review no weekdays", "urinary", "Recent urinary symptoms", "urinary symptoms"),
    ]
    for scenario_id, title, category, duration_text, raw_text in day_locked_cases:
        scenarios.append(
            _scenario(
                scenario_id=scenario_id,
                title=title,
                raw_text=f"{raw_text}. I cannot attend on any weekday initially.",
                complaint_category=category,
                duration_text=duration_text,
                preferences=_preferences(
                    excluded_days=list(WORKING_WEEK),
                    date_horizon_days=10,
                    soonest_weight=60,
                ),
                extra={
                    "negotiation_family": "day_relax",
                    "extracted_constraints_text": ["all weekdays initially excluded"],
                },
            )
        )

    modality_locked_cases = [
        (
            "neg_mod_rx_17",
            "Repeat prescription modality block",
            "repeat prescription review",
            "prescription_admin",
            "Medication request",
            ["phone", "in_person"],
        ),
        (
            "neg_mod_admin_18",
            "Administrative modality block",
            "fit note request",
            "administrative",
            "Admin request",
            ["phone", "video"],
        ),
        (
            "neg_mod_gp_19",
            "General review modality block",
            "routine check",
            "general_review",
            "Routine review",
            ["in_person", "phone", "video"],
        ),
        (
            "neg_mod_resp_20",
            "Respiratory modality block",
            "persistent cough",
            "respiratory",
            "Two-week cough",
            ["in_person", "phone", "video"],
        ),
        (
            "neg_mod_uti_33",
            "Urinary modality block",
            "uti symptoms",
            "urinary",
            "Recent urinary symptoms",
            ["in_person", "phone", "video"],
        ),
        (
            "neg_mod_admin_34",
            "Administrative all-modality block",
            "certificate request",
            "administrative",
            "Admin request",
            ["phone", "video"],
        ),
        (
            "neg_mod_gp_35",
            "General review remote block",
            "routine review",
            "general_review",
            "Routine review",
            ["phone", "video"],
        ),
        (
            "neg_mod_rx_36",
            "Prescription remote block",
            "medication review",
            "prescription_admin",
            "Medication review",
            ["phone", "video"],
        ),
    ]
    for scenario_id, title, raw_text, category, duration_text, disallowed in modality_locked_cases:
        scenarios.append(
            _scenario(
                scenario_id=scenario_id,
                title=title,
                raw_text=f"{raw_text}. Initial preference excludes every available modality.",
                complaint_category=category,
                duration_text=duration_text,
                preferences=_preferences(
                    excluded_modalities=disallowed,
                    date_horizon_days=10,
                    soonest_weight=60,
                ),
                extra={
                    "negotiation_family": "modality_relax",
                    "extracted_constraints_text": ["all route-compatible modalities initially excluded"],
                },
            )
        )

    combined_cases = [
        ("neg_combo_gp_21", "general review", "general_review", "Routine review"),
        ("neg_combo_resp_22", "persistent cough", "respiratory", "Two-week cough"),
        ("neg_combo_admin_23", "fit note", "administrative", "Admin request"),
        ("neg_combo_rx_24", "repeat prescription", "prescription_admin", "Medication request"),
        ("neg_combo_uti_37", "urinary symptoms", "urinary", "Recent urinary symptoms"),
        ("neg_combo_gp_38", "routine review", "general_review", "Routine review"),
        ("neg_combo_admin_39", "admin certificate", "administrative", "Admin certificate"),
        ("neg_combo_resp_40", "asthma review", "respiratory", "Respiratory review"),
    ]
    for scenario_id, raw_text, category, duration_text in combined_cases:
        scenarios.append(
            _scenario(
                scenario_id=scenario_id,
                title=f"Combined relaxation case {scenario_id}",
                raw_text=f"{raw_text}. No weekday attendance, no periods, and every available modality is excluded initially.",
                complaint_category=category,
                duration_text=duration_text,
                preferences=_preferences(
                    excluded_modalities=["in_person", "phone", "video"],
                    excluded_days=list(WORKING_WEEK),
                    excluded_periods=list(ALL_PERIODS),
                    date_horizon_days=10,
                    soonest_weight=60,
                ),
                extra={
                    "negotiation_family": "combined_multi_round",
                    "extracted_constraints_text": [
                        "all weekdays initially excluded",
                        "all periods initially excluded",
                        "all modalities initially excluded",
                    ],
                },
            )
        )

    return scenarios


LLM_GOLDEN_STORIES: List[Dict[str, Any]] = [
    {
        "story_id": "llm_story_01",
        "reason_code": "repeat_prescription",
        "text": "I need a repeat prescription review. Phone is best, Thursday afternoon if possible, within the next 7 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone"],
            preferred_days=["Thu"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Thu", "period": "afternoon"}],
            date_horizon_days=7,
        ),
    },
    {
        "story_id": "llm_story_02",
        "reason_code": "blood_test",
        "text": "I need a blood test next week. It has to be in person and mornings are easier for me.",
        "expected_preferences": _preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            preferred_periods=["morning"],
            date_horizon_days=10,
        ),
    },
    {
        "story_id": "llm_story_03",
        "reason_code": "persistent_cough",
        "text": "I have had a cough for about two weeks. Video on Wednesday morning would suit me best in the next 10 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["video"],
            preferred_days=["Wed"],
            preferred_periods=["morning"],
            preferred_day_periods=[{"day": "Wed", "period": "morning"}],
            date_horizon_days=10,
        ),
    },
    {
        "story_id": "llm_story_04",
        "reason_code": "admin_request",
        "text": "I need a fit note. Remote is fine, but I cannot do mornings. Please keep it within 5 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone", "video"],
            excluded_periods=["morning"],
            preferred_periods=["afternoon"],
            date_horizon_days=5,
        ),
    },
    {
        "story_id": "llm_story_05",
        "reason_code": "general_review",
        "text": "Routine review please. Monday or Friday mornings are ideal and phone would be easiest within the next 8 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone"],
            preferred_days=["Mon", "Fri"],
            preferred_periods=["morning"],
            preferred_day_periods=[
                {"day": "Mon", "period": "morning"},
                {"day": "Fri", "period": "morning"},
            ],
            date_horizon_days=8,
        ),
    },
    {
        "story_id": "llm_story_06",
        "reason_code": "uti_symptoms",
        "text": "I think I have UTI symptoms. I can do any day except Tuesday, and afternoons are better over the next week.",
        "expected_preferences": _preferences(
            excluded_days=["Tue"],
            preferred_periods=["afternoon"],
            date_horizon_days=7,
        ),
    },
    {
        "story_id": "llm_story_07",
        "reason_code": "repeat_prescription",
        "text": "Medication review needed. I would prefer in person, not on Mondays, and it can be within the next 12 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["in_person"],
            excluded_days=["Mon"],
            date_horizon_days=12,
        ),
    },
    {
        "story_id": "llm_story_08",
        "reason_code": "persistent_cough",
        "text": "Persistent cough review please. Phone or video is fine, but not Friday afternoon, and I need it within 9 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone", "video"],
            excluded_day_periods=[{"day": "Fri", "period": "afternoon"}],
            date_horizon_days=9,
        ),
    },
    {
        "story_id": "llm_story_09",
        "reason_code": "admin_request",
        "text": "I need an admin letter. Video is preferred, Wednesday afternoon would be ideal, and nothing beyond 6 days please.",
        "expected_preferences": _preferences(
            preferred_modalities=["video"],
            preferred_days=["Wed"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Wed", "period": "afternoon"}],
            date_horizon_days=6,
        ),
    },
    {
        "story_id": "llm_story_10",
        "reason_code": "general_review",
        "text": "Routine GP review. I cannot do Wednesday or Thursday, and mornings only in the next 11 days.",
        "expected_preferences": _preferences(
            excluded_days=["Wed", "Thu"],
            preferred_periods=["morning"],
            date_horizon_days=11,
        ),
    },
    {
        "story_id": "llm_story_11",
        "reason_code": "blood_test",
        "text": "I need monitoring bloods. In person only, Tuesday or Thursday morning, within the next 14 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            preferred_days=["Tue", "Thu"],
            preferred_periods=["morning"],
            preferred_day_periods=[
                {"day": "Tue", "period": "morning"},
                {"day": "Thu", "period": "morning"},
            ],
            date_horizon_days=14,
        ),
    },
    {
        "story_id": "llm_story_12",
        "reason_code": "uti_symptoms",
        "text": "UTI symptoms. Phone is easiest, any weekday afternoon is okay, and I would like it in the next 4 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone"],
            preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
            preferred_periods=["afternoon"],
            date_horizon_days=4,
        ),
    },
    {
        "story_id": "llm_story_13",
        "reason_code": "repeat_prescription",
        "text": "Could I have a medication review? Video would be easiest, not on Monday, and Wednesday afternoon would suit within 13 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["video"],
            excluded_days=["Mon"],
            preferred_days=["Wed"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Wed", "period": "afternoon"}],
            date_horizon_days=13,
        ),
    },
    {
        "story_id": "llm_story_14",
        "reason_code": "blood_test",
        "text": "Need blood pressure and bloods checked. It must be in person, Tuesday afternoon is best, and I can wait up to 15 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["in_person"],
            excluded_modalities=["phone", "video"],
            preferred_days=["Tue"],
            preferred_periods=["afternoon"],
            preferred_day_periods=[{"day": "Tue", "period": "afternoon"}],
            date_horizon_days=15,
        ),
    },
    {
        "story_id": "llm_story_15",
        "reason_code": "admin_request",
        "text": "I need a certificate discussion. Phone or video is fine, but not Friday, and afternoons please within the next 9 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone", "video"],
            excluded_days=["Fri"],
            preferred_periods=["afternoon"],
            date_horizon_days=9,
        ),
    },
    {
        "story_id": "llm_story_16",
        "reason_code": "general_review",
        "text": "Routine GP review. I can do Tuesday morning or Thursday afternoon in the next 12 days, ideally in person.",
        "expected_preferences": _preferences(
            preferred_modalities=["in_person"],
            preferred_days=["Tue", "Thu"],
            preferred_periods=["morning", "afternoon"],
            preferred_day_periods=[
                {"day": "Tue", "period": "morning"},
                {"day": "Thu", "period": "afternoon"},
            ],
            date_horizon_days=12,
        ),
    },
    {
        "story_id": "llm_story_17",
        "reason_code": "uti_symptoms",
        "text": "Urine symptoms review please. Phone is easiest, not Wednesday morning, and within 5 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["phone"],
            excluded_day_periods=[{"day": "Wed", "period": "morning"}],
            date_horizon_days=5,
        ),
    },
    {
        "story_id": "llm_story_18",
        "reason_code": "persistent_cough",
        "text": "Cough review. I can do any weekday except Monday, mornings or afternoons are fine, and video would be preferred within the next 6 days.",
        "expected_preferences": _preferences(
            preferred_modalities=["video"],
            excluded_days=["Mon"],
            preferred_periods=["morning", "afternoon"],
            date_horizon_days=6,
        ),
    },
]

LLM_GOLDEN_STORIES.extend(
    [
        {
            "story_id": "llm_story_19",
            "reason_code": "admin_request",
            "text": "Could I sort a fit note? Remote is easiest. Please avoid Tuesday morning and Friday afternoon, and keep it within 7 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                excluded_day_periods=[
                    {"day": "Tue", "period": "morning"},
                    {"day": "Fri", "period": "afternoon"},
                ],
                date_horizon_days=7,
            ),
        },
        {
            "story_id": "llm_story_20",
            "reason_code": "general_review",
            "text": "I need a general review. It has to be on the phone, not on Monday or Thursday, and next week is fine.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_modalities=["in_person", "video"],
                excluded_days=["Mon", "Thu"],
                date_horizon_days=7,
            ),
        },
        {
            "story_id": "llm_story_21",
            "reason_code": "blood_test",
            "text": "Blood test booking please. Thursday morning only works, and nothing after 9 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                preferred_days=["Thu"],
                preferred_periods=["morning"],
                preferred_day_periods=[{"day": "Thu", "period": "morning"}],
                date_horizon_days=9,
            ),
        },
        {
            "story_id": "llm_story_22",
            "reason_code": "persistent_cough",
            "text": "Cough review please. Phone is okay, I cannot do Wednesdays, and Monday or Tuesday afternoon would suit in the next 8 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_days=["Wed"],
                preferred_days=["Mon", "Tue"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[
                    {"day": "Mon", "period": "afternoon"},
                    {"day": "Tue", "period": "afternoon"},
                ],
                date_horizon_days=8,
            ),
        },
        {
            "story_id": "llm_story_23",
            "reason_code": "uti_symptoms",
            "text": "UTI symptoms. I would prefer video, not Monday or Tuesday, afternoons only, within the next 5 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                excluded_days=["Mon", "Tue"],
                preferred_periods=["afternoon"],
                date_horizon_days=5,
            ),
        },
        {
            "story_id": "llm_story_24",
            "reason_code": "repeat_prescription",
            "text": "Repeat medication review. Tuesday or Thursday morning please, remote is fine, and up to 10 days works.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                preferred_days=["Tue", "Thu"],
                preferred_periods=["morning"],
                preferred_day_periods=[
                    {"day": "Tue", "period": "morning"},
                    {"day": "Thu", "period": "morning"},
                ],
                date_horizon_days=10,
            ),
        },
        {
            "story_id": "llm_story_25",
            "reason_code": "admin_request",
            "text": "Administrative note request. Phone preferred, not Wednesday afternoon, and I need it within 4 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_day_periods=[{"day": "Wed", "period": "afternoon"}],
                date_horizon_days=4,
            ),
        },
        {
            "story_id": "llm_story_26",
            "reason_code": "general_review",
            "text": "Routine review. I can only do afternoons and not Friday, ideally in person, within the next 14 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_days=["Fri"],
                preferred_periods=["afternoon"],
                excluded_periods=["morning", "evening"],
                date_horizon_days=14,
            ),
        },
        {
            "story_id": "llm_story_27",
            "reason_code": "persistent_cough",
            "text": "Persistent cough for ten days. Video on Monday morning or Thursday morning would be ideal, within 12 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                preferred_days=["Mon", "Thu"],
                preferred_periods=["morning"],
                preferred_day_periods=[
                    {"day": "Mon", "period": "morning"},
                    {"day": "Thu", "period": "morning"},
                ],
                date_horizon_days=12,
            ),
        },
        {
            "story_id": "llm_story_28",
            "reason_code": "blood_test",
            "text": "Monitoring blood test needed. It must be in person, not Monday, not Friday afternoon, and within the next 13 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                excluded_days=["Mon"],
                excluded_day_periods=[{"day": "Fri", "period": "afternoon"}],
                date_horizon_days=13,
            ),
        },
        {
            "story_id": "llm_story_29",
            "reason_code": "uti_symptoms",
            "text": "Urinary symptoms review. Phone is easiest, I cannot do mornings, Tuesday is best, and I need it within 6 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                preferred_days=["Tue"],
                excluded_periods=["morning"],
                date_horizon_days=6,
            ),
        },
        {
            "story_id": "llm_story_30",
            "reason_code": "repeat_prescription",
            "text": "Repeat prescription chat. Video preferred, no Tuesday or Thursday, any morning over the next 11 days would work.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                excluded_days=["Tue", "Thu"],
                preferred_periods=["morning"],
                date_horizon_days=11,
            ),
        },
    ]
)

LLM_GOLDEN_STORIES.extend(
    [
        {
            "story_id": "llm_story_31",
            "reason_code": "admin_request",
            "text": "I need an administrative letter. Phone would be best, Monday afternoon or Thursday afternoon would suit, within the next 8 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                preferred_days=["Mon", "Thu"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[
                    {"day": "Mon", "period": "afternoon"},
                    {"day": "Thu", "period": "afternoon"},
                ],
                date_horizon_days=8,
            ),
        },
        {
            "story_id": "llm_story_32",
            "reason_code": "general_review",
            "text": "General review please. I cannot do Tuesday mornings or Friday mornings, video is preferred, and the next 9 days would be fine.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                excluded_day_periods=[
                    {"day": "Tue", "period": "morning"},
                    {"day": "Fri", "period": "morning"},
                ],
                date_horizon_days=9,
            ),
        },
        {
            "story_id": "llm_story_33",
            "reason_code": "blood_test",
            "text": "Blood test follow-up. It has to be in person, not on Wednesday, and Thursday afternoon would be ideal within 10 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                excluded_days=["Wed"],
                preferred_days=["Thu"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[{"day": "Thu", "period": "afternoon"}],
                date_horizon_days=10,
            ),
        },
        {
            "story_id": "llm_story_34",
            "reason_code": "persistent_cough",
            "text": "Cough review. Phone or video is okay, but not Tuesday or Wednesday afternoon, and I need it within 7 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                excluded_day_periods=[
                    {"day": "Tue", "period": "afternoon"},
                    {"day": "Wed", "period": "afternoon"},
                ],
                date_horizon_days=7,
            ),
        },
        {
            "story_id": "llm_story_35",
            "reason_code": "uti_symptoms",
            "text": "UTI review. I would prefer phone, Friday morning is best, and nothing beyond 5 days please.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                preferred_days=["Fri"],
                preferred_periods=["morning"],
                preferred_day_periods=[{"day": "Fri", "period": "morning"}],
                date_horizon_days=5,
            ),
        },
        {
            "story_id": "llm_story_36",
            "reason_code": "repeat_prescription",
            "text": "Repeat medication request. I cannot do mornings, video would be easiest, and Wednesday or Friday afternoon would work over the next 10 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                preferred_days=["Wed", "Fri"],
                preferred_periods=["afternoon"],
                excluded_periods=["morning"],
                preferred_day_periods=[
                    {"day": "Wed", "period": "afternoon"},
                    {"day": "Fri", "period": "afternoon"},
                ],
                date_horizon_days=10,
            ),
        },
        {
            "story_id": "llm_story_37",
            "reason_code": "admin_request",
            "text": "Certificate discussion please. Remote is fine, but avoid Monday and Tuesday, and afternoons only within 6 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                excluded_days=["Mon", "Tue"],
                preferred_periods=["afternoon"],
                excluded_periods=["morning", "evening"],
                date_horizon_days=6,
            ),
        },
        {
            "story_id": "llm_story_38",
            "reason_code": "general_review",
            "text": "Routine review. It has to be in person, not Thursday afternoon, and Monday morning would be ideal within 12 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                preferred_days=["Mon"],
                preferred_periods=["morning"],
                preferred_day_periods=[{"day": "Mon", "period": "morning"}],
                excluded_day_periods=[{"day": "Thu", "period": "afternoon"}],
                date_horizon_days=12,
            ),
        },
        {
            "story_id": "llm_story_39",
            "reason_code": "persistent_cough",
            "text": "Persistent cough review. Video preferred, not on Friday, mornings only, and within the next 9 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                excluded_days=["Fri"],
                preferred_periods=["morning"],
                date_horizon_days=9,
            ),
        },
        {
            "story_id": "llm_story_40",
            "reason_code": "blood_test",
            "text": "Monitoring bloods. In person only. Tuesday morning or Wednesday afternoon would work, within 11 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                preferred_days=["Tue", "Wed"],
                preferred_periods=["morning", "afternoon"],
                preferred_day_periods=[
                    {"day": "Tue", "period": "morning"},
                    {"day": "Wed", "period": "afternoon"},
                ],
                date_horizon_days=11,
            ),
        },
        {
            "story_id": "llm_story_41",
            "reason_code": "uti_symptoms",
            "text": "Urinary symptoms. I cannot do Tuesday or Thursday, evenings are easiest, and phone would be preferred within 8 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_days=["Tue", "Thu"],
                preferred_periods=["evening"],
                date_horizon_days=8,
            ),
        },
        {
            "story_id": "llm_story_42",
            "reason_code": "repeat_prescription",
            "text": "Medication review. Remote preferred, not Wednesday morning, and any afternoon in the next 7 days is okay.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                excluded_day_periods=[{"day": "Wed", "period": "morning"}],
                preferred_periods=["afternoon"],
                date_horizon_days=7,
            ),
        },
        {
            "story_id": "llm_story_43",
            "reason_code": "admin_request",
            "text": "I need a fit note chat. Phone only, not Monday, and Tuesday morning would suit within 5 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_modalities=["in_person", "video"],
                excluded_days=["Mon"],
                preferred_days=["Tue"],
                preferred_periods=["morning"],
                preferred_day_periods=[{"day": "Tue", "period": "morning"}],
                date_horizon_days=5,
            ),
        },
        {
            "story_id": "llm_story_44",
            "reason_code": "general_review",
            "text": "General review. Wednesday or Friday would be best, not mornings, and video would be easiest within the next 13 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                preferred_days=["Wed", "Fri"],
                excluded_periods=["morning"],
                date_horizon_days=13,
            ),
        },
        {
            "story_id": "llm_story_45",
            "reason_code": "persistent_cough",
            "text": "Cough review for two weeks. I would prefer phone, not Tuesday morning, and Thursday afternoon is ideal within 10 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_day_periods=[{"day": "Tue", "period": "morning"}],
                preferred_days=["Thu"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[{"day": "Thu", "period": "afternoon"}],
                date_horizon_days=10,
            ),
        },
        {
            "story_id": "llm_story_46",
            "reason_code": "blood_test",
            "text": "Blood pressure check. It has to be in person, Friday afternoon is preferred, and I cannot do Monday or Tuesday within the next 14 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["in_person"],
                excluded_modalities=["phone", "video"],
                excluded_days=["Mon", "Tue"],
                preferred_days=["Fri"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[{"day": "Fri", "period": "afternoon"}],
                date_horizon_days=14,
            ),
        },
        {
            "story_id": "llm_story_47",
            "reason_code": "uti_symptoms",
            "text": "UTI symptoms review. Video would be best, mornings only, not Friday, within 6 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                excluded_days=["Fri"],
                preferred_periods=["morning"],
                excluded_periods=["afternoon", "evening"],
                date_horizon_days=6,
            ),
        },
        {
            "story_id": "llm_story_48",
            "reason_code": "repeat_prescription",
            "text": "Repeat prescription review. Phone is best, not Wednesday or Thursday afternoon, and the next 8 days are fine.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone"],
                excluded_day_periods=[
                    {"day": "Wed", "period": "afternoon"},
                    {"day": "Thu", "period": "afternoon"},
                ],
                date_horizon_days=8,
            ),
        },
        {
            "story_id": "llm_story_49",
            "reason_code": "admin_request",
            "text": "Administrative letter request. Video is preferred, Monday morning or Friday morning would work, and nothing beyond 9 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["video"],
                preferred_days=["Mon", "Fri"],
                preferred_periods=["morning"],
                preferred_day_periods=[
                    {"day": "Mon", "period": "morning"},
                    {"day": "Fri", "period": "morning"},
                ],
                date_horizon_days=9,
            ),
        },
        {
            "story_id": "llm_story_50",
            "reason_code": "general_review",
            "text": "Routine GP review. Phone or video is fine, not on Monday, Tuesday afternoon is ideal, and I need it within 7 days.",
            "expected_preferences": _preferences(
                preferred_modalities=["phone", "video"],
                excluded_days=["Mon"],
                preferred_days=["Tue"],
                preferred_periods=["afternoon"],
                preferred_day_periods=[{"day": "Tue", "period": "afternoon"}],
                date_horizon_days=7,
            ),
        },
    ]
)


LLM_NOISY_STORIES: List[Dict[str, Any]] = [
    {
        "prompt_id": "noisy_01",
        "reason_code": "repeat_prescription",
        "text": "need repeat meds call pls, thu aft if u can, nxt 7 days",
        "expected_complaint": "prescription_admin",
    },
    {
        "prompt_id": "noisy_02",
        "reason_code": "blood_test",
        "text": "bloods nxt wk, has 2b in person, morns only",
        "expected_complaint": "tests_monitoring",
    },
    {
        "prompt_id": "noisy_03",
        "reason_code": "persistent_cough",
        "text": "cough 2 wks lol not serious, wed am video maybe?",
        "expected_complaint": "respiratory",
    },
    {
        "prompt_id": "noisy_04",
        "reason_code": "admin_request",
        "text": "fit note thing, cant do mornin, remote ok, 5 days max",
        "expected_complaint": "administrative",
    },
    {
        "prompt_id": "noisy_05",
        "reason_code": "general_review",
        "text": "gp review plz, mon/fri am best, phone easiest, in 8 days",
        "expected_complaint": "general_review",
    },
    {
        "prompt_id": "noisy_06",
        "reason_code": "uti_symptoms",
        "text": "uti probs, not tues, aftns r better, this wk if poss",
        "expected_complaint": "urinary",
    },
    {
        "prompt_id": "noisy_07",
        "reason_code": "repeat_prescription",
        "text": "med review, prefer f2f, no mon pls, within 12 days",
        "expected_complaint": "prescription_admin",
    },
    {
        "prompt_id": "noisy_08",
        "reason_code": "persistent_cough",
        "text": "pers cough, phone/video ok, not fri aft, need it <9 days",
        "expected_complaint": "respiratory",
    },
    {
        "prompt_id": "noisy_09",
        "reason_code": "admin_request",
        "text": "admin letta, vid pref, wed pm ideal, 6 day window",
        "expected_complaint": "administrative",
    },
    {
        "prompt_id": "noisy_10",
        "reason_code": "general_review",
        "text": "routine gp, no wed or thu, mornings only, 11 day limit",
        "expected_complaint": "general_review",
    },
    {
        "prompt_id": "noisy_11",
        "reason_code": "blood_test",
        "text": "monitorin bloods, in person only, tue/thu am, 14 days tops",
        "expected_complaint": "tests_monitoring",
    },
    {
        "prompt_id": "noisy_12",
        "reason_code": "uti_symptoms",
        "text": "uti sx. phone easiest. any weekday pm. want it in 4 days",
        "expected_complaint": "urinary",
    },
    {
        "prompt_id": "noisy_13",
        "reason_code": "repeat_prescription",
        "text": "med review pls. vid easiest. not mon. wed pm if poss. 13 day window",
        "expected_complaint": "prescription_admin",
    },
    {
        "prompt_id": "noisy_14",
        "reason_code": "blood_test",
        "text": "need bp + bloods checked. in person only. tue pm best. can wait 15 days",
        "expected_complaint": "tests_monitoring",
    },
    {
        "prompt_id": "noisy_15",
        "reason_code": "admin_request",
        "text": "need cert chat. phone/video ok. not fri. aft pls. nxt 9 days",
        "expected_complaint": "administrative",
    },
    {
        "prompt_id": "noisy_16",
        "reason_code": "general_review",
        "text": "routine gp rev. tue am or thu pm. ideally f2f. 12 day limit",
        "expected_complaint": "general_review",
    },
    {
        "prompt_id": "noisy_17",
        "reason_code": "uti_symptoms",
        "text": "urine probs. phone easiest. not wed am. need in 5 days",
        "expected_complaint": "urinary",
    },
    {
        "prompt_id": "noisy_18",
        "reason_code": "persistent_cough",
        "text": "cough review. vid pref. any weekday but not mon. am/pm ok. 6 days tops",
        "expected_complaint": "respiratory",
    },
]

LLM_NOISY_STORIES.extend(
    [
        {
            "prompt_id": "noisy_19",
            "reason_code": "admin_request",
            "text": "fit note pls. remote easiest. avoid tue am + fri pm. need 7 days",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_20",
            "reason_code": "general_review",
            "text": "gen review. phone only. no mon/thu. nxt wk ok",
            "expected_complaint": "general_review",
        },
        {
            "prompt_id": "noisy_21",
            "reason_code": "blood_test",
            "text": "blood test booking. thu am only. max 9 days",
            "expected_complaint": "tests_monitoring",
        },
        {
            "prompt_id": "noisy_22",
            "reason_code": "persistent_cough",
            "text": "cough review. phone ok. no wed. mon/tue pm best. 8 days",
            "expected_complaint": "respiratory",
        },
        {
            "prompt_id": "noisy_23",
            "reason_code": "uti_symptoms",
            "text": "uti probs. vid pref. no mon or tue. afternoons only. 5 days",
            "expected_complaint": "urinary",
        },
        {
            "prompt_id": "noisy_24",
            "reason_code": "repeat_prescription",
            "text": "repeat meds. tue/thu am pls. remote ok. 10 day limit",
            "expected_complaint": "prescription_admin",
        },
        {
            "prompt_id": "noisy_25",
            "reason_code": "admin_request",
            "text": "admin note. phone pref. not wed pm. 4 day window",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_26",
            "reason_code": "general_review",
            "text": "routine review. afternoons only. not fri. ideally in person. 14 days",
            "expected_complaint": "general_review",
        },
        {
            "prompt_id": "noisy_27",
            "reason_code": "persistent_cough",
            "text": "pers cough 10 days. vid mon am or thu am best. 12 days",
            "expected_complaint": "respiratory",
        },
        {
            "prompt_id": "noisy_28",
            "reason_code": "blood_test",
            "text": "monitoring bloods. in person only. not mon. not fri aft. 13 days",
            "expected_complaint": "tests_monitoring",
        },
        {
            "prompt_id": "noisy_29",
            "reason_code": "uti_symptoms",
            "text": "urinary sx. phone easiest. no mornings. tue best. 6 days",
            "expected_complaint": "urinary",
        },
        {
            "prompt_id": "noisy_30",
            "reason_code": "repeat_prescription",
            "text": "repeat rx. video pref. no tue/thu. any morning. 11 days",
            "expected_complaint": "prescription_admin",
        },
    ]
)

LLM_NOISY_STORIES.extend(
    [
        {
            "prompt_id": "noisy_31",
            "reason_code": "admin_request",
            "text": "admin letta. phone best. mon/thu aft ideal. 8 days",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_32",
            "reason_code": "general_review",
            "text": "gen review. no tue am or fri am. vid pref. 9 days",
            "expected_complaint": "general_review",
        },
        {
            "prompt_id": "noisy_33",
            "reason_code": "blood_test",
            "text": "blood test fup. in person only. no wed. thu aft best. 10 days",
            "expected_complaint": "tests_monitoring",
        },
        {
            "prompt_id": "noisy_34",
            "reason_code": "persistent_cough",
            "text": "cough review. phone/video ok. no tue/wed aft. 7 days",
            "expected_complaint": "respiratory",
        },
        {
            "prompt_id": "noisy_35",
            "reason_code": "uti_symptoms",
            "text": "uti review. phone pref. fri am best. max 5 days",
            "expected_complaint": "urinary",
        },
        {
            "prompt_id": "noisy_36",
            "reason_code": "repeat_prescription",
            "text": "repeat meds. no mornings. vid easiest. wed/fri pm. 10 days",
            "expected_complaint": "prescription_admin",
        },
        {
            "prompt_id": "noisy_37",
            "reason_code": "admin_request",
            "text": "cert chat. remote ok. avoid mon+tue. aft only. 6 days",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_38",
            "reason_code": "general_review",
            "text": "routine review. f2f only. not thu aft. mon am best. 12 days",
            "expected_complaint": "general_review",
        },
        {
            "prompt_id": "noisy_39",
            "reason_code": "persistent_cough",
            "text": "pers cough. vid pref. not fri. mornings only. 9 days",
            "expected_complaint": "respiratory",
        },
        {
            "prompt_id": "noisy_40",
            "reason_code": "blood_test",
            "text": "monitoring bloods. in person only. tue am or wed pm. 11 days",
            "expected_complaint": "tests_monitoring",
        },
        {
            "prompt_id": "noisy_41",
            "reason_code": "uti_symptoms",
            "text": "urinary probs. no tue/thu. eve best. phone pref. 8 days",
            "expected_complaint": "urinary",
        },
        {
            "prompt_id": "noisy_42",
            "reason_code": "repeat_prescription",
            "text": "med review. remote pref. not wed am. any aft ok. 7 days",
            "expected_complaint": "prescription_admin",
        },
        {
            "prompt_id": "noisy_43",
            "reason_code": "admin_request",
            "text": "fit note chat. phone only. not mon. tue am suits. 5 days",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_44",
            "reason_code": "general_review",
            "text": "general review. wed/fri best. not mornings. vid easiest. 13 days",
            "expected_complaint": "general_review",
        },
        {
            "prompt_id": "noisy_45",
            "reason_code": "persistent_cough",
            "text": "2wk cough. phone pref. not tue am. thu aft ideal. 10 days",
            "expected_complaint": "respiratory",
        },
        {
            "prompt_id": "noisy_46",
            "reason_code": "blood_test",
            "text": "bp check. in person only. fri aft pref. no mon/tue. 14 days",
            "expected_complaint": "tests_monitoring",
        },
        {
            "prompt_id": "noisy_47",
            "reason_code": "uti_symptoms",
            "text": "uti sx. vid best. mornings only. not fri. 6 days",
            "expected_complaint": "urinary",
        },
        {
            "prompt_id": "noisy_48",
            "reason_code": "repeat_prescription",
            "text": "repeat rx review. phone best. no wed/thu aft. 8 days",
            "expected_complaint": "prescription_admin",
        },
        {
            "prompt_id": "noisy_49",
            "reason_code": "admin_request",
            "text": "admin letter. vid pref. mon or fri am good. 9 days",
            "expected_complaint": "administrative",
        },
        {
            "prompt_id": "noisy_50",
            "reason_code": "general_review",
            "text": "routine gp review. phone/video ok. not mon. tue aft ideal. 7 days",
            "expected_complaint": "general_review",
        },
    ]
)


RUNTIME_STANDARD_SCENARIO: Dict[str, Any] = _scenario(
    scenario_id="runtime_standard_gp",
    title="Standard runtime scaling scenario",
    raw_text="Routine GP review with flexible preferences and a phone preference.",
    complaint_category="general_review",
    duration_text="Routine review",
    preferences=_preferences(
        preferred_modalities=["phone"],
        preferred_days=["Mon", "Tue", "Wed", "Thu", "Fri"],
        preferred_periods=["morning", "afternoon"],
        date_horizon_days=14,
        soonest_weight=60,
    ),
    extra={
        "extracted_constraints_text": [
            "phone preferred",
            "weekday appointment acceptable",
        ]
    },
)
