from __future__ import annotations

import builtins
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from experiments.common import RESULTS_DIR, basic_stats, ensure_results_dir, extended_stats, write_csv
from experiments.scenario_definitions import LLM_GOLDEN_STORIES, LLM_NOISY_STORIES
from nhs_demo.agents.receptionist import ReceptionistAgent
from nhs_demo.api import _load_local_env, create_app
from nhs_demo.schemas import FormIntakeRequest, IntakeRequest, IntakeSummary, PatientPreferences

LLM_VS_FORM_CSV = RESULTS_DIR / "llm_vs_form.csv"
LLM_VS_FORM_SUMMARY_CSV = RESULTS_DIR / "llm_vs_form_summary.csv"
LLM_VS_FORM_RAW_CSV = RESULTS_DIR / "llm_vs_form_raw.csv"
LLM_VS_FORM_RAW_SUMMARY_CSV = RESULTS_DIR / "llm_vs_form_raw_summary.csv"
LLM_VS_FORM_COMPARISON_CSV = RESULTS_DIR / "llm_vs_form_comparison.csv"
LLM_STORY_ACCURACY_CSV = RESULTS_DIR / "llm_story_accuracy.csv"
LLM_VS_FORM_STORY_METRICS_CSV = RESULTS_DIR / "llm_vs_form_story_metrics.csv"
LLM_VS_FORM_STORY_METRICS_RAW_CSV = RESULTS_DIR / "llm_vs_form_story_metrics_raw.csv"
LLM_VS_FORM_TIMING_SUMMARY_CSV = RESULTS_DIR / "llm_vs_form_timing_summary.csv"
LLM_ROBUSTNESS_CSV = RESULTS_DIR / "llm_robustness.csv"
LLM_ROBUSTNESS_RAW_CSV = RESULTS_DIR / "llm_robustness_raw.csv"
LLM_ROBUSTNESS_SUMMARY_CSV = RESULTS_DIR / "llm_robustness_summary.csv"
LLM_FAILURE_MODES_CSV = RESULTS_DIR / "llm_failure_modes.csv"

FIELDS_TO_COMPARE: Dict[str, Callable[[IntakeSummary], Any]] = {
    "complaint_category": lambda intake: intake.complaint_category,
    "duration_text": lambda intake: intake.duration_text or "",
    "preferred_modalities": lambda intake: intake.preferences.preferred_modalities,
    "excluded_modalities": lambda intake: intake.preferences.excluded_modalities,
    "preferred_days": lambda intake: intake.preferences.preferred_days,
    "excluded_days": lambda intake: intake.preferences.excluded_days,
    "preferred_periods": lambda intake: intake.preferences.preferred_periods,
    "excluded_periods": lambda intake: intake.preferences.excluded_periods,
    "preferred_day_periods": lambda intake: [item.model_dump(mode="json") for item in intake.preferences.preferred_day_periods],
    "excluded_day_periods": lambda intake: [item.model_dump(mode="json") for item in intake.preferences.excluded_day_periods],
    "date_horizon_days": lambda intake: intake.preferences.date_horizon_days,
}


def _serialise(value: Any) -> str:
    if isinstance(value, list):
        normalised = sorted((_serialise(item) for item in value))
        return json.dumps(normalised, ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value)


def _match_category(llm_value: Any, form_value: Any) -> str:
    if _serialise(llm_value) == _serialise(form_value):
        return "exact"

    if isinstance(llm_value, list) and isinstance(form_value, list):
        llm_items = {_serialise(item) for item in llm_value}
        form_items = {_serialise(item) for item in form_value}
        if llm_items & form_items:
            return "partial"
        return "mismatch"

    llm_text = _serialise(llm_value).strip().lower()
    form_text = _serialise(form_value).strip().lower()
    if llm_text and form_text and (llm_text in form_text or form_text in llm_text):
        return "partial"
    return "mismatch"


def _llm_runtime_available(api_key: str | None) -> bool:
    return bool(api_key and api_key.strip() and "YOUR_OPENAI_API_KEY" not in api_key)


def _should_force_http_fallback() -> bool:
    raw_value = os.getenv("LLM_EVAL_FORCE_HTTP", "0").strip().lower()
    return raw_value not in {"0", "false", "no"}


@contextmanager
def _force_openai_http_fallback(force_http: bool):
    if not force_http:
        yield
        return

    real_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("Forced HTTP fallback for bounded LLM evaluation.")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = patched_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _build_form_baseline(agent: ReceptionistAgent, story: Dict[str, Any]) -> IntakeSummary:
    form_intake, _questions, _engine, _payload = agent.build_form_intake(
        run_id=story["story_id"],
        reason_code=story["reason_code"],
        preferences=PatientPreferences.model_validate(story["expected_preferences"]),
        extractor="rule",
    )
    return form_intake


@contextmanager
def _llm_normalisation_mode(enabled: bool):
    if enabled:
        yield
        return

    original_category = ReceptionistAgent._canonicalize_complaint_category
    original_duration = ReceptionistAgent._canonicalize_duration_text
    original_horizon = ReceptionistAgent._extract_date_horizon_days

    def raw_category(self, raw_value, text):
        return self._normalize_text_value(raw_value, fallback="general_query")

    def raw_duration(self, raw_value, text):
        return self._normalize_text_value(raw_value, fallback="")

    def no_horizon(self, text):
        return None

    ReceptionistAgent._canonicalize_complaint_category = raw_category  # type: ignore[assignment]
    ReceptionistAgent._canonicalize_duration_text = raw_duration  # type: ignore[assignment]
    ReceptionistAgent._extract_date_horizon_days = no_horizon  # type: ignore[assignment]
    try:
        yield
    finally:
        ReceptionistAgent._canonicalize_complaint_category = original_category  # type: ignore[assignment]
        ReceptionistAgent._canonicalize_duration_text = original_duration  # type: ignore[assignment]
        ReceptionistAgent._extract_date_horizon_days = original_horizon  # type: ignore[assignment]


def _golden_output_paths(canonicalize: bool) -> tuple[Path, Path, Path, str]:
    if canonicalize:
        return (
            LLM_VS_FORM_CSV,
            LLM_VS_FORM_SUMMARY_CSV,
            LLM_VS_FORM_STORY_METRICS_CSV,
            "canonicalized",
        )
    return (
        LLM_VS_FORM_RAW_CSV,
        LLM_VS_FORM_RAW_SUMMARY_CSV,
        LLM_VS_FORM_STORY_METRICS_RAW_CSV,
        "raw",
    )


def _robustness_output_paths(canonicalize: bool) -> tuple[Path, str]:
    if canonicalize:
        return LLM_ROBUSTNESS_CSV, "canonicalized"
    return LLM_ROBUSTNESS_RAW_CSV, "raw"


def run_golden_case_accuracy(
    api_key: str | None = None,
    llm_model: str | None = None,
    *,
    canonicalize: bool = True,
) -> Path:
    ensure_results_dir()
    _load_local_env()
    effective_key = api_key or os.getenv("OPENAI_API_KEY")
    agent = ReceptionistAgent()
    rows: List[Dict[str, object]] = []
    story_rows: List[Dict[str, object]] = []
    force_http = _should_force_http_fallback()
    detail_csv_path, summary_csv_path, story_csv_path, mode_name = _golden_output_paths(canonicalize)

    if not _llm_runtime_available(effective_key):
        write_csv(
            detail_csv_path,
            ["story_id", "field_name", "llm_value", "form_value", "match_category"],
            [],
        )
        write_csv(
            summary_csv_path,
            [
                "field_name",
                "total_cases",
                "exact_matches",
                "partial_matches",
                "mismatches",
                "accuracy_pct",
                "semantic_accuracy_pct",
            ],
            [],
        )
        write_csv(
            story_csv_path,
            [
                "story_id",
                "form_latency_ms",
                "llm_latency_ms",
                "llm_call_status",
                "exact_matches",
                "partial_matches",
                "mismatches",
                "total_fields",
                "exact_rate_pct",
                "semantic_rate_pct",
            ],
            [],
        )
        print(f"Skipping {mode_name} LLM vs form evaluation because no valid API key is available.")
        return detail_csv_path

    print(f"Running {mode_name} golden-case LLM evaluation with force_http_fallback={force_http}")
    with _force_openai_http_fallback(force_http), _llm_normalisation_mode(canonicalize):
        for index, story in enumerate(LLM_GOLDEN_STORIES, start=1):
            form_start = time.perf_counter()
            form_baseline = _build_form_baseline(agent, story)
            form_latency_ms = round((time.perf_counter() - form_start) * 1000, 3)
            try:
                llm_start = time.perf_counter()
                llm_intake, _questions, _engine, _payload = agent.build_intake(
                    run_id=story["story_id"],
                    user_text=story["text"],
                    clarification_answers={},
                    extractor="llm",
                    api_key=effective_key,
                    llm_model=llm_model,
                    preference_hint=None,
                )
                llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 3)
                assert llm_intake is not None
            except Exception as exc:
                llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 3)
                print(f"[{mode_name} golden {index}/{len(LLM_GOLDEN_STORIES)}] {story['story_id']}: ERROR {exc}")
                for field_name, getter in FIELDS_TO_COMPARE.items():
                    rows.append(
                        {
                            "story_id": story["story_id"],
                            "field_name": field_name,
                            "llm_value": f"ERROR: {exc}",
                            "form_value": _serialise(getter(form_baseline)),
                            "match_category": "mismatch",
                        }
                    )
                story_rows.append(
                    {
                        "story_id": story["story_id"],
                        "form_latency_ms": form_latency_ms,
                        "llm_latency_ms": llm_latency_ms,
                        "llm_call_status": "error",
                        "exact_matches": 0,
                        "partial_matches": 0,
                        "mismatches": len(FIELDS_TO_COMPARE),
                        "total_fields": len(FIELDS_TO_COMPARE),
                        "exact_rate_pct": 0.0,
                        "semantic_rate_pct": 0.0,
                    }
                )
                continue

            exact_count = 0
            partial_count = 0
            mismatch_count = 0
            for field_name, getter in FIELDS_TO_COMPARE.items():
                llm_value = getter(llm_intake)
                form_value = getter(form_baseline)
                match_category = _match_category(llm_value, form_value)
                if match_category == "exact":
                    exact_count += 1
                elif match_category == "partial":
                    partial_count += 1
                elif match_category == "mismatch":
                    mismatch_count += 1
                rows.append(
                    {
                        "story_id": story["story_id"],
                        "field_name": field_name,
                        "llm_value": _serialise(llm_value),
                        "form_value": _serialise(form_value),
                        "match_category": match_category,
                    }
                )
            story_rows.append(
                {
                    "story_id": story["story_id"],
                    "form_latency_ms": form_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "llm_call_status": "ok",
                    "exact_matches": exact_count,
                    "partial_matches": partial_count,
                    "mismatches": mismatch_count,
                    "total_fields": len(FIELDS_TO_COMPARE),
                    "exact_rate_pct": round((exact_count / len(FIELDS_TO_COMPARE)) * 100, 2),
                    "semantic_rate_pct": round(((exact_count + partial_count) / len(FIELDS_TO_COMPARE)) * 100, 2),
                }
            )
            print(
                f"[{mode_name} golden {index}/{len(LLM_GOLDEN_STORIES)}] {story['story_id']}: "
                f"exact={exact_count} partial={partial_count} mismatch={mismatch_count}"
            )

    write_csv(
        detail_csv_path,
        ["story_id", "field_name", "llm_value", "form_value", "match_category"],
        rows,
    )
    write_csv(
        story_csv_path,
        [
            "story_id",
            "form_latency_ms",
            "llm_latency_ms",
            "llm_call_status",
            "exact_matches",
            "partial_matches",
            "mismatches",
            "total_fields",
            "exact_rate_pct",
            "semantic_rate_pct",
        ],
        story_rows,
    )
    summarise_llm_vs_form(detail_csv_path, summary_csv_path)
    return detail_csv_path


def summarise_llm_vs_form(
    detail_csv_path: Path = LLM_VS_FORM_CSV,
    summary_csv_path: Path = LLM_VS_FORM_SUMMARY_CSV,
) -> Path:
    grouped: Dict[str, List[str]] = {}
    total_matches = {"exact": 0, "partial": 0, "mismatch": 0}

    with detail_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            field_name = row["field_name"]
            match_category = row["match_category"]
            grouped.setdefault(field_name, []).append(match_category)
            total_matches[match_category] = total_matches.get(match_category, 0) + 1

    summary_rows: List[Dict[str, object]] = []
    for field_name in sorted(grouped):
        categories = grouped[field_name]
        exact_matches = categories.count("exact")
        partial_matches = categories.count("partial")
        mismatches = categories.count("mismatch")
        total_cases = len(categories)
        summary_rows.append(
            {
                "field_name": field_name,
                "total_cases": total_cases,
                "exact_matches": exact_matches,
                "partial_matches": partial_matches,
                "mismatches": mismatches,
                "accuracy_pct": round((exact_matches / total_cases) * 100, 2) if total_cases else 0.0,
                "semantic_accuracy_pct": round(((exact_matches + partial_matches) / total_cases) * 100, 2) if total_cases else 0.0,
            }
        )

    total_cases = sum(total_matches.values())
    summary_rows.append(
        {
            "field_name": "overall",
            "total_cases": total_cases,
            "exact_matches": total_matches["exact"],
            "partial_matches": total_matches["partial"],
            "mismatches": total_matches["mismatch"],
            "accuracy_pct": round((total_matches["exact"] / total_cases) * 100, 2) if total_cases else 0.0,
            "semantic_accuracy_pct": round(
                ((total_matches["exact"] + total_matches["partial"]) / total_cases) * 100,
                2,
            ) if total_cases else 0.0,
        }
    )

    write_csv(
        summary_csv_path,
        [
            "field_name",
            "total_cases",
            "exact_matches",
            "partial_matches",
            "mismatches",
            "accuracy_pct",
            "semantic_accuracy_pct",
        ],
        summary_rows,
    )
    return summary_csv_path


def run_noisy_prompt_robustness(
    api_key: str | None = None,
    llm_model: str | None = None,
    *,
    canonicalize: bool = True,
) -> Path:
    ensure_results_dir()
    _load_local_env()
    effective_key = api_key or os.getenv("OPENAI_API_KEY")
    agent = ReceptionistAgent()
    rows: List[Dict[str, object]] = []
    force_http = _should_force_http_fallback()
    robustness_csv_path, mode_name = _robustness_output_paths(canonicalize)

    if not _llm_runtime_available(effective_key):
        write_csv(
            robustness_csv_path,
            [
                "prompt_id",
                "valid_schema_flag",
                "extracted_complaint_category",
                "expected_complaint_category",
                "obvious_misinterpretation_flag",
                "notes",
            ],
            [],
        )
        print(f"Skipping {mode_name} noisy-prompt robustness evaluation because no valid API key is available.")
        return robustness_csv_path

    print(f"Running {mode_name} noisy-prompt LLM evaluation with force_http_fallback={force_http}")
    with _force_openai_http_fallback(force_http), _llm_normalisation_mode(canonicalize):
        for index, prompt_case in enumerate(LLM_NOISY_STORIES, start=1):
            try:
                intake, _questions, _engine, _payload = agent.build_intake(
                    run_id=prompt_case["prompt_id"],
                    user_text=prompt_case["text"],
                    clarification_answers={},
                    extractor="llm",
                    api_key=effective_key,
                    llm_model=llm_model,
                    preference_hint=None,
                )
                assert intake is not None
                extracted_complaint = intake.complaint_category
                obvious_misinterpretation = int(extracted_complaint != prompt_case["expected_complaint"])
                notes = "" if not obvious_misinterpretation else "Complaint category differs from expected label."
                rows.append(
                    {
                        "prompt_id": prompt_case["prompt_id"],
                        "valid_schema_flag": 1,
                        "extracted_complaint_category": extracted_complaint,
                        "expected_complaint_category": prompt_case["expected_complaint"],
                        "obvious_misinterpretation_flag": obvious_misinterpretation,
                        "notes": notes,
                    }
                )
                print(
                    f"[{mode_name} noisy {index}/{len(LLM_NOISY_STORIES)}] {prompt_case['prompt_id']}: "
                    f"valid_schema=1 mismatch={obvious_misinterpretation}"
                )
            except Exception as exc:
                rows.append(
                    {
                        "prompt_id": prompt_case["prompt_id"],
                        "valid_schema_flag": 0,
                        "extracted_complaint_category": "None",
                        "expected_complaint_category": prompt_case["expected_complaint"],
                        "obvious_misinterpretation_flag": 1,
                        "notes": str(exc),
                    }
                )
                print(f"[{mode_name} noisy {index}/{len(LLM_NOISY_STORIES)}] {prompt_case['prompt_id']}: ERROR {exc}")

    write_csv(
        robustness_csv_path,
        [
            "prompt_id",
            "valid_schema_flag",
            "extracted_complaint_category",
            "expected_complaint_category",
            "obvious_misinterpretation_flag",
            "notes",
        ],
        rows,
    )
    return robustness_csv_path


def build_llm_comparison_outputs(
    raw_summary_csv_path: Path = LLM_VS_FORM_RAW_SUMMARY_CSV,
    canonical_summary_csv_path: Path = LLM_VS_FORM_SUMMARY_CSV,
    raw_story_metrics_csv_path: Path = LLM_VS_FORM_STORY_METRICS_RAW_CSV,
    canonical_story_metrics_csv_path: Path = LLM_VS_FORM_STORY_METRICS_CSV,
    comparison_csv_path: Path = LLM_VS_FORM_COMPARISON_CSV,
    story_accuracy_csv_path: Path = LLM_STORY_ACCURACY_CSV,
    timing_summary_csv_path: Path = LLM_VS_FORM_TIMING_SUMMARY_CSV,
) -> None:
    raw_summary = {
        row["field_name"]: row for row in csv.DictReader(raw_summary_csv_path.open("r", encoding="utf-8", newline=""))
    }
    canonical_summary = {
        row["field_name"]: row
        for row in csv.DictReader(canonical_summary_csv_path.open("r", encoding="utf-8", newline=""))
    }

    comparison_rows: List[Dict[str, object]] = []
    for field_name in sorted(set(raw_summary) | set(canonical_summary)):
        raw_row = raw_summary.get(field_name, {})
        canonical_row = canonical_summary.get(field_name, {})
        raw_accuracy = float(raw_row.get("accuracy_pct", 0.0))
        canonical_accuracy = float(canonical_row.get("accuracy_pct", 0.0))
        comparison_rows.append(
            {
                "field_name": field_name,
                "raw_accuracy_pct": round(raw_accuracy, 2),
                "canonicalized_accuracy_pct": round(canonical_accuracy, 2),
                "delta_accuracy_pct": round(canonical_accuracy - raw_accuracy, 2),
            }
        )

    write_csv(
        comparison_csv_path,
        ["field_name", "raw_accuracy_pct", "canonicalized_accuracy_pct", "delta_accuracy_pct"],
        comparison_rows,
    )

    raw_story_metrics = {
        row["story_id"]: row
        for row in csv.DictReader(raw_story_metrics_csv_path.open("r", encoding="utf-8", newline=""))
    }
    canonical_story_metrics = {
        row["story_id"]: row
        for row in csv.DictReader(canonical_story_metrics_csv_path.open("r", encoding="utf-8", newline=""))
    }
    story_rows = []
    for story_id in sorted(set(raw_story_metrics) | set(canonical_story_metrics)):
        raw_story = raw_story_metrics.get(story_id, {})
        canonical_story = canonical_story_metrics.get(story_id, {})
        raw_rate = float(raw_story.get("exact_rate_pct", 0.0))
        canonical_rate = float(canonical_story.get("exact_rate_pct", 0.0))
        story_rows.append(
            {
                "story_id": story_id,
                "raw_exact_rate_pct": raw_rate,
                "canonicalized_exact_rate_pct": canonical_rate,
                "delta_exact_rate_pct": round(canonical_rate - raw_rate, 2),
                "raw_semantic_rate_pct": float(raw_story.get("semantic_rate_pct", 0.0)),
                "canonicalized_semantic_rate_pct": float(canonical_story.get("semantic_rate_pct", 0.0)),
                "form_latency_ms": float(canonical_story.get("form_latency_ms", raw_story.get("form_latency_ms", 0.0)) or 0.0),
                "raw_llm_latency_ms": float(raw_story.get("llm_latency_ms", 0.0) or 0.0),
                "canonicalized_llm_latency_ms": float(canonical_story.get("llm_latency_ms", 0.0) or 0.0),
                "raw_llm_call_status": raw_story.get("llm_call_status", ""),
                "canonicalized_llm_call_status": canonical_story.get("llm_call_status", ""),
            }
        )
    write_csv(
        story_accuracy_csv_path,
        [
            "story_id",
            "raw_exact_rate_pct",
            "canonicalized_exact_rate_pct",
            "delta_exact_rate_pct",
            "raw_semantic_rate_pct",
            "canonicalized_semantic_rate_pct",
            "form_latency_ms",
            "raw_llm_latency_ms",
            "canonicalized_llm_latency_ms",
            "raw_llm_call_status",
            "canonicalized_llm_call_status",
        ],
        story_rows,
    )
    summarise_llm_timing(
        raw_story_metrics_csv_path,
        canonical_story_metrics_csv_path,
        timing_summary_csv_path,
    )


def summarise_llm_timing(
    raw_story_metrics_csv_path: Path = LLM_VS_FORM_STORY_METRICS_RAW_CSV,
    canonical_story_metrics_csv_path: Path = LLM_VS_FORM_STORY_METRICS_CSV,
    summary_csv_path: Path = LLM_VS_FORM_TIMING_SUMMARY_CSV,
) -> Path:
    rows: List[Dict[str, object]] = []
    for mode_name, path in (("raw", raw_story_metrics_csv_path), ("canonicalized", canonical_story_metrics_csv_path)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            items = list(csv.DictReader(handle))
        total = len(items)
        successful = [row for row in items if row.get("llm_call_status") == "ok"]
        form_latencies = [float(row["form_latency_ms"]) for row in successful if row.get("form_latency_ms")]
        llm_latencies = [float(row["llm_latency_ms"]) for row in successful if row.get("llm_latency_ms")]
        exact_rates = [float(row["exact_rate_pct"]) for row in successful if row.get("exact_rate_pct")]
        semantic_rates = [float(row["semantic_rate_pct"]) for row in successful if row.get("semantic_rate_pct")]
        llm_stats = extended_stats(llm_latencies)
        form_stats = basic_stats(form_latencies)
        rows.append(
            {
                "mode": mode_name,
                "total_stories": total,
                "successful_llm_calls": len(successful),
                "mean_form_latency_ms": form_stats["mean"],
                "mean_llm_latency_ms": llm_stats["mean"],
                "median_llm_latency_ms": llm_stats["median"],
                "p95_llm_latency_ms": llm_stats["p95"],
                "mean_exact_rate_pct": round(sum(exact_rates) / len(exact_rates), 2) if exact_rates else 0.0,
                "mean_semantic_rate_pct": round(sum(semantic_rates) / len(semantic_rates), 2) if semantic_rates else 0.0,
                "complete_story_exact_rate_pct": round(
                    (sum(1 for row in successful if float(row.get("exact_rate_pct", 0.0)) == 100.0) / len(successful)) * 100,
                    2,
                ) if successful else 0.0,
            }
        )
    write_csv(
        summary_csv_path,
        [
            "mode",
            "total_stories",
            "successful_llm_calls",
            "mean_form_latency_ms",
            "mean_llm_latency_ms",
            "median_llm_latency_ms",
            "p95_llm_latency_ms",
            "mean_exact_rate_pct",
            "mean_semantic_rate_pct",
            "complete_story_exact_rate_pct",
        ],
        rows,
    )
    return summary_csv_path


def summarise_llm_robustness(
    raw_csv_path: Path = LLM_ROBUSTNESS_RAW_CSV,
    canonical_csv_path: Path = LLM_ROBUSTNESS_CSV,
    summary_csv_path: Path = LLM_ROBUSTNESS_SUMMARY_CSV,
) -> Path:
    rows: List[Dict[str, object]] = []
    for mode_name, path in (("raw", raw_csv_path), ("canonicalized", canonical_csv_path)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            items = list(csv.DictReader(handle))
        total = len(items)
        valid_schema_rate = round((sum(int(row["valid_schema_flag"]) for row in items) / total) * 100, 2) if total else 0.0
        strict_label_match_rate = round(
            ((total - sum(int(row["obvious_misinterpretation_flag"]) for row in items)) / total) * 100,
            2,
        ) if total else 0.0
        rows.append(
            {
                "mode": mode_name,
                "total_prompts": total,
                "valid_schema_rate_pct": valid_schema_rate,
                "strict_label_match_rate_pct": strict_label_match_rate,
            }
        )

    write_csv(
        summary_csv_path,
        ["mode", "total_prompts", "valid_schema_rate_pct", "strict_label_match_rate_pct"],
        rows,
    )
    return summary_csv_path


def run_failure_mode_checks() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    invalid_key_client = TestClient(create_app())
    invalid_key_response = invalid_key_client.post(
        "/api/intake",
        json={
            "user_text": "I need a routine review next week.",
            "clarification_answers": {},
            "extractor": "llm",
            "api_key": "invalid-key-for-test",
        },
    )
    invalid_key_body = invalid_key_response.json()
    rows.append(
        {
            "test_case": "invalid_api_key",
            "status_code": invalid_key_response.status_code,
            "clean_error_flag": int(invalid_key_response.status_code == 400 and "LLM extraction failed" in invalid_key_body.get("detail", "")),
            "scheduling_attempted": 0,
            "detail": invalid_key_body.get("detail", ""),
        }
    )

    original_try_llm = ReceptionistAgent._try_llm

    def forced_network_error(self, *args, **kwargs):
        self._last_llm_error = "forced network error"
        return None

    try:
        ReceptionistAgent._try_llm = forced_network_error  # type: ignore[assignment]
        forced_error_client = TestClient(create_app())
        forced_error_response = forced_error_client.post(
            "/api/intake",
            json={
                "user_text": "I need a routine review next week.",
                "clarification_answers": {},
                "extractor": "llm",
                "api_key": "test-key",
            },
        )
        forced_error_body = forced_error_response.json()
        rows.append(
            {
                "test_case": "forced_network_error",
                "status_code": forced_error_response.status_code,
                "clean_error_flag": int(forced_error_response.status_code == 400 and "LLM extraction failed" in forced_error_body.get("detail", "")),
                "scheduling_attempted": 0,
                "detail": forced_error_body.get("detail", ""),
            }
        )
    finally:
        ReceptionistAgent._try_llm = original_try_llm  # type: ignore[assignment]

    write_csv(
        LLM_FAILURE_MODES_CSV,
        ["test_case", "status_code", "clean_error_flag", "scheduling_attempted", "detail"],
        rows,
    )
    return LLM_FAILURE_MODES_CSV


def main() -> None:
    run_golden_case_accuracy(canonicalize=False)
    run_golden_case_accuracy(canonicalize=True)
    build_llm_comparison_outputs()
    run_noisy_prompt_robustness(canonicalize=False)
    run_noisy_prompt_robustness(canonicalize=True)
    summarise_llm_robustness()
    run_failure_mode_checks()
    print(f"Wrote {LLM_VS_FORM_CSV}")
    print(f"Wrote {LLM_VS_FORM_SUMMARY_CSV}")
    print(f"Wrote {LLM_VS_FORM_RAW_CSV}")
    print(f"Wrote {LLM_VS_FORM_RAW_SUMMARY_CSV}")
    print(f"Wrote {LLM_VS_FORM_COMPARISON_CSV}")
    print(f"Wrote {LLM_STORY_ACCURACY_CSV}")
    print(f"Wrote {LLM_VS_FORM_STORY_METRICS_CSV}")
    print(f"Wrote {LLM_VS_FORM_STORY_METRICS_RAW_CSV}")
    print(f"Wrote {LLM_VS_FORM_TIMING_SUMMARY_CSV}")
    print(f"Wrote {LLM_ROBUSTNESS_CSV}")
    print(f"Wrote {LLM_ROBUSTNESS_RAW_CSV}")
    print(f"Wrote {LLM_ROBUSTNESS_SUMMARY_CSV}")
    print(f"Wrote {LLM_FAILURE_MODES_CSV}")


if __name__ == "__main__":
    main()
