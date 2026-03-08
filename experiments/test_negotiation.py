from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    RESULTS_DIR,
    accept_all_relaxations,
    accept_first_relaxation_only,
    basic_stats,
    bootstrap_ci,
    ensure_results_dir,
    extended_stats,
    list_to_pipe_string,
    reject_all_relaxations,
    run_structured_workflow,
    write_csv,
)
from experiments.scenario_definitions import negotiation_scenarios
from nhs_demo.orchestrator import DemoOrchestrator

NEGOTIATION_RUNS_CSV = RESULTS_DIR / "negotiation_runs.csv"
NEGOTIATION_SUMMARY_CSV = RESULTS_DIR / "negotiation_summary.csv"
NEGOTIATION_FAMILY_SUMMARY_CSV = RESULTS_DIR / "negotiation_family_summary.csv"
NEGOTIATION_DECISION_CSV = RESULTS_DIR / "negotiation_decision_sensitivity.csv"
NEGOTIATION_DECISION_SUMMARY_CSV = RESULTS_DIR / "negotiation_decision_sensitivity_summary.csv"

RELAXATION_KEYS = [
    "relax_excluded_periods",
    "relax_excluded_days",
    "relax_excluded_modalities",
    "extend_date_horizon",
]


def run_negotiation_experiment() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    for scenario in negotiation_scenarios():
        orchestrator = DemoOrchestrator()
        result = run_structured_workflow(
            orchestrator,
            scenario,
            use_routing_hint=False,
            relaxation_policy=accept_all_relaxations,
        )
        offer = result["offer"]
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "negotiation_family": scenario["negotiation_family"],
                "number_of_rounds": result["negotiation_rounds"],
                "final_outcome": offer.status,
                "final_slot_id": offer.booking.slot_id if offer.booking else "None",
                "constraints_relaxed": list_to_pipe_string(result["applied_relaxations"]),
            }
        )

    write_csv(
        NEGOTIATION_RUNS_CSV,
        [
            "scenario_id",
            "negotiation_family",
            "number_of_rounds",
            "final_outcome",
            "final_slot_id",
            "constraints_relaxed",
        ],
        rows,
    )
    return NEGOTIATION_RUNS_CSV


def summarise_negotiation_runs(
    runs_csv_path: Path = NEGOTIATION_RUNS_CSV,
    summary_csv_path: Path = NEGOTIATION_SUMMARY_CSV,
    family_summary_csv_path: Path = NEGOTIATION_FAMILY_SUMMARY_CSV,
) -> Path:
    rows: List[Dict[str, str]] = []
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_cases = len(rows)
    successful_rows = [row for row in rows if row["final_outcome"] == "booked"]
    success_rate = round((len(successful_rows) / total_cases) * 100, 2) if total_cases else 0.0
    round_values = [int(row["number_of_rounds"]) for row in rows]
    round_stats = basic_stats([float(value) for value in round_values])

    relaxation_counter: Counter[str] = Counter()
    for row in rows:
        relaxed = [item for item in row["constraints_relaxed"].split("|") if item]
        relaxation_counter.update(relaxed)

    success_flags = [1.0 if row["final_outcome"] == "booked" else 0.0 for row in rows]
    success_ci = bootstrap_ci(success_flags, statistic="proportion")

    summary_row = {
        "total_cases": total_cases,
        "successful_cases": len(successful_rows),
        "success_rate_pct": success_rate,
        "success_rate_ci_lower": round(success_ci["lower"] * 100, 2),
        "success_rate_ci_upper": round(success_ci["upper"] * 100, 2),
        "mean_rounds": round_stats["mean"],
        "relax_excluded_periods_count": relaxation_counter.get("relax_excluded_periods", 0),
        "relax_excluded_days_count": relaxation_counter.get("relax_excluded_days", 0),
        "relax_excluded_modalities_count": relaxation_counter.get("relax_excluded_modalities", 0),
        "extend_date_horizon_count": relaxation_counter.get("extend_date_horizon", 0),
    }
    write_csv(
        summary_csv_path,
        [
            "total_cases",
            "successful_cases",
            "success_rate_pct",
            "success_rate_ci_lower",
            "success_rate_ci_upper",
            "mean_rounds",
            "relax_excluded_periods_count",
            "relax_excluded_days_count",
            "relax_excluded_modalities_count",
            "extend_date_horizon_count",
        ],
        [summary_row],
    )

    family_rows: List[Dict[str, object]] = []
    family_grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        family_grouped.setdefault(row["negotiation_family"], []).append(row)

    for family in sorted(family_grouped):
        family_bucket = family_grouped[family]
        family_successes = [row for row in family_bucket if row["final_outcome"] == "booked"]
        family_rounds = [float(row["number_of_rounds"]) for row in family_bucket]
        family_stats = extended_stats(family_rounds)
        family_rows.append(
            {
                "negotiation_family": family,
                "total_cases": len(family_bucket),
                "successful_cases": len(family_successes),
                "success_rate_pct": round((len(family_successes) / len(family_bucket)) * 100, 2) if family_bucket else 0.0,
                "mean_rounds": family_stats["mean"],
                "median_rounds": family_stats["median"],
                "p95_rounds": family_stats["p95"],
            }
        )

    write_csv(
        family_summary_csv_path,
        [
            "negotiation_family",
            "total_cases",
            "successful_cases",
            "success_rate_pct",
            "mean_rounds",
            "median_rounds",
            "p95_rounds",
        ],
        family_rows,
    )
    return summary_csv_path


def run_decision_sensitivity_experiment() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []
    scenarios = negotiation_scenarios()

    for scenario in scenarios:
        accept_result = run_structured_workflow(
            DemoOrchestrator(),
            scenario,
            use_routing_hint=False,
            relaxation_policy=accept_all_relaxations,
        )
        accept_first_result = run_structured_workflow(
            DemoOrchestrator(),
            scenario,
            use_routing_hint=False,
            relaxation_policy=accept_first_relaxation_only,
        )
        reject_result = run_structured_workflow(
            DemoOrchestrator(),
            scenario,
            use_routing_hint=False,
            relaxation_policy=reject_all_relaxations,
        )
        accept_offer = accept_result["offer"]
        accept_first_offer = accept_first_result["offer"]
        reject_offer = reject_result["offer"]
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "negotiation_family": scenario["negotiation_family"],
                "accept_all_outcome": accept_offer.status,
                "accept_all_rounds": accept_result["negotiation_rounds"],
                "accept_first_outcome": accept_first_offer.status,
                "accept_first_rounds": accept_first_result["negotiation_rounds"],
                "reject_all_outcome": reject_offer.status,
                "reject_all_rounds": reject_result["negotiation_rounds"],
            }
        )

    write_csv(
        NEGOTIATION_DECISION_CSV,
        [
            "scenario_id",
            "negotiation_family",
            "accept_all_outcome",
            "accept_all_rounds",
            "accept_first_outcome",
            "accept_first_rounds",
            "reject_all_outcome",
            "reject_all_rounds",
        ],
        rows,
    )
    return NEGOTIATION_DECISION_CSV


def summarise_decision_sensitivity(
    runs_csv_path: Path = NEGOTIATION_DECISION_CSV,
    summary_csv_path: Path = NEGOTIATION_DECISION_SUMMARY_CSV,
) -> Path:
    rows: List[Dict[str, str]] = []
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_cases = len(rows)
    accept_successes = sum(1 for row in rows if row["accept_all_outcome"] == "booked")
    accept_first_successes = sum(1 for row in rows if row["accept_first_outcome"] == "booked")
    reject_successes = sum(1 for row in rows if row["reject_all_outcome"] == "booked")

    accept_rate = round((accept_successes / total_cases) * 100, 2) if total_cases else 0.0
    accept_first_rate = round((accept_first_successes / total_cases) * 100, 2) if total_cases else 0.0
    reject_rate = round((reject_successes / total_cases) * 100, 2) if total_cases else 0.0

    accept_flags = [1.0 if row["accept_all_outcome"] == "booked" else 0.0 for row in rows]
    reject_flags = [1.0 if row["reject_all_outcome"] == "booked" else 0.0 for row in rows]
    accept_ci = bootstrap_ci(accept_flags, statistic="proportion")
    reject_ci = bootstrap_ci(reject_flags, statistic="proportion")

    summary_row = {
        "total_cases": total_cases,
        "accept_all_success_rate_pct": accept_rate,
        "accept_all_ci_lower": round(accept_ci["lower"] * 100, 2),
        "accept_all_ci_upper": round(accept_ci["upper"] * 100, 2),
        "accept_first_success_rate_pct": accept_first_rate,
        "reject_all_success_rate_pct": reject_rate,
        "reject_all_ci_lower": round(reject_ci["lower"] * 100, 2),
        "reject_all_ci_upper": round(reject_ci["upper"] * 100, 2),
        "accept_all_vs_reject_gap_pct": round(accept_rate - reject_rate, 2),
        "accept_all_vs_accept_first_gap_pct": round(accept_rate - accept_first_rate, 2),
    }
    write_csv(
        summary_csv_path,
        [
            "total_cases",
            "accept_all_success_rate_pct",
            "accept_all_ci_lower",
            "accept_all_ci_upper",
            "accept_first_success_rate_pct",
            "reject_all_success_rate_pct",
            "reject_all_ci_lower",
            "reject_all_ci_upper",
            "accept_all_vs_reject_gap_pct",
            "accept_all_vs_accept_first_gap_pct",
        ],
        [summary_row],
    )
    return summary_csv_path


def main() -> None:
    runs_path = run_negotiation_experiment()
    summary_path = summarise_negotiation_runs(runs_path)
    decision_path = run_decision_sensitivity_experiment()
    decision_summary_path = summarise_decision_sensitivity(decision_path)

    print(f"Wrote {runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {NEGOTIATION_FAMILY_SUMMARY_CSV}")
    print(f"Wrote {decision_path}")
    print(f"Wrote {decision_summary_path}")


if __name__ == "__main__":
    main()
