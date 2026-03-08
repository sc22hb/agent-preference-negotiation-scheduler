from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    RESULTS_DIR,
    basic_stats,
    bootstrap_ci,
    cohens_d,
    ensure_results_dir,
    extended_stats,
    run_structured_workflow,
    utility_for_booking,
    write_csv,
)
from experiments.scenario_definitions import allocation_quality_profiles
from nhs_demo.orchestrator import DemoOrchestrator

ALLOCATION_QUALITY_CSV = RESULTS_DIR / "allocation_quality.csv"
ALLOCATION_QUALITY_SUMMARY_CSV = RESULTS_DIR / "allocation_quality_summary.csv"
FAIRNESS_SUMMARY_CSV = RESULTS_DIR / "fairness_summary.csv"
ALLOCATION_REASON_SUMMARY_CSV = RESULTS_DIR / "allocation_reason_summary.csv"
ALLOCATION_PRESSURE_SUMMARY_CSV = RESULTS_DIR / "allocation_pressure_summary.csv"
ALLOCATION_GROUP_REASON_SUMMARY_CSV = RESULTS_DIR / "allocation_group_reason_summary.csv"


def _num_preferred_days(profile: Dict[str, object]) -> int:
    preferences = profile["preferences"]
    preferred_days = set(preferences.get("preferred_days", []))
    preferred_day_periods = preferences.get("preferred_day_periods", [])
    preferred_days.update(item["day"] for item in preferred_day_periods)
    return len(preferred_days)


def _strong_modality_preference_flag(profile: Dict[str, object]) -> int:
    preferred_modalities = profile["preferences"].get("preferred_modalities", [])
    return 1 if len(preferred_modalities) == 1 else 0


def _constraint_pressure_score(profile: Dict[str, object]) -> int:
    preferences = profile["preferences"]
    preferred_day_periods = preferences.get("preferred_day_periods", [])
    excluded_day_periods = preferences.get("excluded_day_periods", [])
    score = 0
    score += len(preferences.get("excluded_days", []))
    score += len(preferences.get("excluded_periods", []))
    score += len(preferred_day_periods) * 2
    score += len(excluded_day_periods) * 2
    score += 1 if len(preferences.get("preferred_modalities", [])) == 1 else 0
    score += 2 if preferences.get("date_horizon_days", 0) <= 5 else 0
    score += 1 if preferences.get("date_horizon_days", 0) <= 8 else 0
    return score


def _pressure_band(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def run_allocation_quality_experiment() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    for profile in allocation_quality_profiles():
        orchestrator = DemoOrchestrator()
        result = run_structured_workflow(
            orchestrator,
            profile,
            use_routing_hint=False,
            relaxation_policy=None,
        )
        routing = result["routing_decision"]
        preferences = result["final_preferences"]
        offer = result["offer"]
        utility = utility_for_booking(
            orchestrator,
            routing,
            preferences,
            offer.booking,
        )
        rows.append(
            {
                "profile_id": profile["scenario_id"],
                "group_tag": profile["group"],
                "reason_code": profile["reason_code"],
                "service_type": routing.service_type,
                "success_flag": 1 if offer.booking else 0,
                "allocated_slot_id": offer.booking.slot_id if offer.booking else "None",
                "total_utility": utility,
                "num_preferred_days": _num_preferred_days(profile),
                "num_excluded_days": len(profile["preferences"].get("excluded_days", [])),
                "strong_modality_preference_flag": _strong_modality_preference_flag(profile),
                "preferred_period_count": len(profile["preferences"].get("preferred_periods", [])),
                "date_horizon_days": profile["preferences"]["date_horizon_days"],
                "constraint_pressure_score": _constraint_pressure_score(profile),
                "constraint_pressure_band": _pressure_band(_constraint_pressure_score(profile)),
            }
        )

    write_csv(
        ALLOCATION_QUALITY_CSV,
        [
            "profile_id",
            "group_tag",
            "reason_code",
            "service_type",
            "success_flag",
            "allocated_slot_id",
            "total_utility",
            "num_preferred_days",
            "num_excluded_days",
            "strong_modality_preference_flag",
            "preferred_period_count",
            "date_horizon_days",
            "constraint_pressure_score",
            "constraint_pressure_band",
        ],
        rows,
    )
    return ALLOCATION_QUALITY_CSV


def summarise_allocation_quality(
    allocation_csv_path: Path = ALLOCATION_QUALITY_CSV,
    summary_csv_path: Path = ALLOCATION_QUALITY_SUMMARY_CSV,
    fairness_csv_path: Path = FAIRNESS_SUMMARY_CSV,
) -> Dict[str, float]:
    rows: List[Dict[str, object]] = []
    with allocation_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_profiles = len(rows)
    successful_rows = [row for row in rows if row["success_flag"] == "1"]
    successful_utilities = [float(row["total_utility"]) for row in successful_rows if row["total_utility"]]
    stats = extended_stats(successful_utilities)
    success_rate = round((len(successful_rows) / total_profiles) * 100, 2) if total_profiles else 0.0

    success_flags = [1.0 if row["success_flag"] == "1" else 0.0 for row in rows]
    success_ci = bootstrap_ci(success_flags, statistic="proportion")
    utility_ci = bootstrap_ci(successful_utilities, statistic="mean") if successful_utilities else {"point": 0.0, "lower": 0.0, "upper": 0.0}

    summary_row = {
        "total_profiles": total_profiles,
        "successful_profiles": len(successful_rows),
        "success_rate_pct": success_rate,
        "success_rate_ci_lower": round(success_ci["lower"] * 100, 2),
        "success_rate_ci_upper": round(success_ci["upper"] * 100, 2),
        "mean_utility_successes": stats["mean"],
        "utility_ci_lower": utility_ci["lower"],
        "utility_ci_upper": utility_ci["upper"],
        "median_utility_successes": stats["median"],
        "stdev_utility_successes": stats["stdev"],
        "p25_utility_successes": stats["p25"],
        "p75_utility_successes": stats["p75"],
        "p95_utility_successes": stats["p95"],
    }
    write_csv(
        summary_csv_path,
        [
            "total_profiles",
            "successful_profiles",
            "success_rate_pct",
            "success_rate_ci_lower",
            "success_rate_ci_upper",
            "mean_utility_successes",
            "utility_ci_lower",
            "utility_ci_upper",
            "median_utility_successes",
            "stdev_utility_successes",
            "p25_utility_successes",
            "p75_utility_successes",
            "p95_utility_successes",
        ],
        [summary_row],
    )

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_tag"]].append(row)

    fairness_rows: List[Dict[str, object]] = []
    group_utility_lists: Dict[str, List[float]] = {}
    for group_tag in sorted(grouped):
        group_rows = grouped[group_tag]
        group_successes = [row for row in group_rows if row["success_flag"] == "1"]
        group_utilities = [float(row["total_utility"]) for row in group_successes if row["total_utility"]]
        group_utility_lists[group_tag] = group_utilities
        group_stats = extended_stats(group_utilities)
        group_success_flags = [1.0 if row["success_flag"] == "1" else 0.0 for row in group_rows]
        group_ci = bootstrap_ci(group_success_flags, statistic="proportion")
        group_success_rate = round((len(group_successes) / len(group_rows)) * 100, 2) if group_rows else 0.0
        fairness_rows.append(
            {
                "group_tag": group_tag,
                "total_profiles": len(group_rows),
                "successful_profiles": len(group_successes),
                "success_rate_pct": group_success_rate,
                "success_rate_ci_lower": round(group_ci["lower"] * 100, 2),
                "success_rate_ci_upper": round(group_ci["upper"] * 100, 2),
                "mean_utility_successes": group_stats["mean"],
                "median_utility_successes": group_stats["median"],
                "p75_utility_successes": group_stats["p75"],
            }
        )

    # Compute Cohen's d between group utility distributions
    group_tags = sorted(group_utility_lists.keys())
    if len(group_tags) >= 2:
        effect_d = cohens_d(group_utility_lists[group_tags[0]], group_utility_lists[group_tags[1]])
        for row in fairness_rows:
            row["cohens_d_utility"] = effect_d

    write_csv(
        fairness_csv_path,
        [
            "group_tag",
            "total_profiles",
            "successful_profiles",
            "success_rate_pct",
            "success_rate_ci_lower",
            "success_rate_ci_upper",
            "mean_utility_successes",
            "median_utility_successes",
            "p75_utility_successes",
            "cohens_d_utility",
        ],
        fairness_rows,
    )

    reason_rows: List[Dict[str, object]] = []
    group_reason_rows: List[Dict[str, object]] = []
    pressure_rows: List[Dict[str, object]] = []

    reason_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    group_reason_grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    pressure_grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        reason_grouped[row["reason_code"]].append(row)
        group_reason_grouped[(row["group_tag"], row["reason_code"])].append(row)
        pressure_grouped[row["constraint_pressure_band"]].append(row)

    for reason_code in sorted(reason_grouped):
        bucket = reason_grouped[reason_code]
        successes = [row for row in bucket if row["success_flag"] == "1"]
        utilities = [float(row["total_utility"]) for row in successes if row["total_utility"]]
        stats = extended_stats(utilities)
        reason_rows.append(
            {
                "reason_code": reason_code,
                "total_profiles": len(bucket),
                "successful_profiles": len(successes),
                "success_rate_pct": round((len(successes) / len(bucket)) * 100, 2) if bucket else 0.0,
                "mean_utility_successes": stats["mean"],
                "median_utility_successes": stats["median"],
            }
        )

    for (group_tag, reason_code), bucket in sorted(group_reason_grouped.items()):
        successes = [row for row in bucket if row["success_flag"] == "1"]
        utilities = [float(row["total_utility"]) for row in successes if row["total_utility"]]
        stats = basic_stats(utilities)
        group_reason_rows.append(
            {
                "group_tag": group_tag,
                "reason_code": reason_code,
                "total_profiles": len(bucket),
                "successful_profiles": len(successes),
                "success_rate_pct": round((len(successes) / len(bucket)) * 100, 2) if bucket else 0.0,
                "mean_utility_successes": stats["mean"],
            }
        )

    for pressure_band in ("low", "medium", "high"):
        bucket = pressure_grouped.get(pressure_band, [])
        successes = [row for row in bucket if row["success_flag"] == "1"]
        utilities = [float(row["total_utility"]) for row in successes if row["total_utility"]]
        stats = extended_stats(utilities)
        pressure_rows.append(
            {
                "constraint_pressure_band": pressure_band,
                "total_profiles": len(bucket),
                "successful_profiles": len(successes),
                "success_rate_pct": round((len(successes) / len(bucket)) * 100, 2) if bucket else 0.0,
                "mean_utility_successes": stats["mean"],
                "median_utility_successes": stats["median"],
            }
        )

    write_csv(
        ALLOCATION_REASON_SUMMARY_CSV,
        [
            "reason_code",
            "total_profiles",
            "successful_profiles",
            "success_rate_pct",
            "mean_utility_successes",
            "median_utility_successes",
        ],
        reason_rows,
    )
    write_csv(
        ALLOCATION_GROUP_REASON_SUMMARY_CSV,
        [
            "group_tag",
            "reason_code",
            "total_profiles",
            "successful_profiles",
            "success_rate_pct",
            "mean_utility_successes",
        ],
        group_reason_rows,
    )
    write_csv(
        ALLOCATION_PRESSURE_SUMMARY_CSV,
        [
            "constraint_pressure_band",
            "total_profiles",
            "successful_profiles",
            "success_rate_pct",
            "mean_utility_successes",
            "median_utility_successes",
        ],
        pressure_rows,
    )

    print(
        "Allocation quality summary:",
        f"n={summary_row['total_profiles']},",
        f"success_rate={summary_row['success_rate_pct']}%,",
        f"mean_utility={summary_row['mean_utility_successes']},",
        f"median_utility={summary_row['median_utility_successes']},",
        f"stdev={summary_row['stdev_utility_successes']}",
    )
    for fairness_row in fairness_rows:
        print(
            "Fairness summary:",
            f"group={fairness_row['group_tag']},",
            f"success_rate={fairness_row['success_rate_pct']}%,",
            f"mean_utility={fairness_row['mean_utility_successes']}",
        )

    return summary_row


def main() -> None:
    allocation_csv_path = run_allocation_quality_experiment()
    summarise_allocation_quality(allocation_csv_path)
    print(f"Wrote {allocation_csv_path}")
    print(f"Wrote {ALLOCATION_QUALITY_SUMMARY_CSV}")
    print(f"Wrote {FAIRNESS_SUMMARY_CSV}")
    print(f"Wrote {ALLOCATION_REASON_SUMMARY_CSV}")
    print(f"Wrote {ALLOCATION_PRESSURE_SUMMARY_CSV}")
    print(f"Wrote {ALLOCATION_GROUP_REASON_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
