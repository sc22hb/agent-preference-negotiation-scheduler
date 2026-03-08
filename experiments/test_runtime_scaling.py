from __future__ import annotations

import csv
import sys
import time
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
    build_intake_summary,
    ensure_results_dir,
    extended_stats,
    ols_fit,
    run_structured_workflow,
    write_csv,
)
from experiments.scenario_definitions import RUNTIME_STANDARD_SCENARIO, negotiation_scenarios
from nhs_demo.orchestrator import DemoOrchestrator
from nhs_demo.schemas import RouteRequest, ScheduleOfferRequest

RUNTIME_SCALING_CSV = RESULTS_DIR / "runtime_scaling.csv"
RUNTIME_SCALING_SUMMARY_CSV = RESULTS_DIR / "runtime_scaling_summary.csv"
RUNTIME_NEGOTIATION_CSV = RESULTS_DIR / "runtime_negotiation_rounds.csv"
RUNTIME_NEGOTIATION_SUMMARY_CSV = RESULTS_DIR / "runtime_negotiation_rounds_summary.csv"

SLOT_SIZES = [50, 100, 200, 400, 800]
RUNS_PER_SIZE = 50


def _prepare_runtime_run(orchestrator: DemoOrchestrator) -> tuple[str, object, object]:
    intake_summary = build_intake_summary(RUNTIME_STANDARD_SCENARIO)
    run_id = orchestrator.create_structured_run(intake_summary)
    route_response = orchestrator.route(
        RouteRequest(run_id=run_id, intake_summary=intake_summary.model_copy(update={"run_id": run_id}))
    )
    return run_id, route_response.routing_decision, intake_summary.preferences


def run_runtime_scaling_experiment() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    for total_slots in SLOT_SIZES:
        for run_index in range(1, RUNS_PER_SIZE + 1):
            orchestrator = DemoOrchestrator()
            run_id, routing_decision, preferences = _prepare_runtime_run(orchestrator)
            candidate_slots = orchestrator.rota_agent.generate_scaled_slots(routing_decision, total_slots)
            request = ScheduleOfferRequest(
                run_id=run_id,
                routing_decision=routing_decision,
                preferences=preferences,
            )

            start = time.perf_counter()
            offer = orchestrator.offer_with_candidate_slots(request, candidate_slots)
            runtime_ms = round((time.perf_counter() - start) * 1000, 3)

            rows.append(
                {
                    "N": total_slots,
                    "run_index": run_index,
                    "runtime_ms": runtime_ms,
                    "status": offer.status,
                }
            )

    write_csv(RUNTIME_SCALING_CSV, ["N", "run_index", "runtime_ms", "status"], rows)
    return RUNTIME_SCALING_CSV


def summarise_runtime_scaling(
    runs_csv_path: Path = RUNTIME_SCALING_CSV,
    summary_csv_path: Path = RUNTIME_SCALING_SUMMARY_CSV,
) -> Path:
    grouped: Dict[int, List[float]] = {size: [] for size in SLOT_SIZES}
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["N"])].append(float(row["runtime_ms"]))

    rows: List[Dict[str, object]] = []
    for total_slots in SLOT_SIZES:
        runtimes = grouped[total_slots]
        ci = bootstrap_ci(runtimes, statistic="mean") if runtimes else {"lower": 0.0, "upper": 0.0}
        rows.append(
            {
                "N": total_slots,
                "mean_runtime_ms": round(sum(runtimes) / len(runtimes), 3) if runtimes else 0.0,
                "mean_ci_lower": ci["lower"],
                "mean_ci_upper": ci["upper"],
                "median_runtime_ms": basic_stats(runtimes)["median"] if runtimes else 0.0,
                "stdev_runtime_ms": basic_stats(runtimes)["stdev"] if runtimes else 0.0,
                "p95_runtime_ms": extended_stats(runtimes)["p95"] if runtimes else 0.0,
                "max_runtime_ms": round(max(runtimes), 3) if runtimes else 0.0,
            }
        )

    # OLS regression on mean runtime vs N to characterise growth rate
    x_vals = [float(row["N"]) for row in rows]
    y_vals = [float(row["mean_runtime_ms"]) for row in rows]
    fit = ols_fit(x_vals, y_vals)
    rows.append(
        {
            "N": "OLS_FIT",
            "mean_runtime_ms": fit["slope"],
            "mean_ci_lower": fit["intercept"],
            "mean_ci_upper": fit["r_squared"],
            "median_runtime_ms": 0.0,
            "stdev_runtime_ms": 0.0,
            "p95_runtime_ms": 0.0,
            "max_runtime_ms": 0.0,
        }
    )

    write_csv(
        summary_csv_path,
        ["N", "mean_runtime_ms", "mean_ci_lower", "mean_ci_upper", "median_runtime_ms", "stdev_runtime_ms", "p95_runtime_ms", "max_runtime_ms"],
        rows,
    )
    return summary_csv_path


def _runtime_negotiation_cases() -> List[Dict[str, object]]:
    scenarios = negotiation_scenarios()
    one_round_case = next(item for item in scenarios if item["negotiation_family"] == "time_relax")
    two_round_case = next(item for item in scenarios if item["negotiation_family"] == "combined_multi_round")
    three_round_case = next(item for item in scenarios if item["negotiation_family"] == "combined_multi_round")
    return [
        {
            "case_id": "runtime_rounds_0",
            "scenario": RUNTIME_STANDARD_SCENARIO,
            "policy": None,
            "target_round_bucket": 0,
        },
        {
            "case_id": "runtime_rounds_1",
            "scenario": one_round_case,
            "policy": accept_all_relaxations,
            "target_round_bucket": 1,
        },
        {
            "case_id": "runtime_rounds_2",
            "scenario": two_round_case,
            "policy": accept_all_relaxations,
            "target_round_bucket": 2,
        },
        {
            "case_id": "runtime_rounds_3",
            "scenario": three_round_case,
            "policy": accept_first_relaxation_only,
            "target_round_bucket": 3,
        },
    ]


def run_runtime_by_negotiation_rounds() -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    for case in _runtime_negotiation_cases():
        for run_index in range(1, RUNS_PER_SIZE + 1):
            orchestrator = DemoOrchestrator()
            start = time.perf_counter()
            result = run_structured_workflow(
                orchestrator,
                case["scenario"],
                use_routing_hint=False,
                relaxation_policy=case["policy"],
            )
            runtime_ms = round((time.perf_counter() - start) * 1000, 3)
            offer = result["offer"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "target_round_bucket": case["target_round_bucket"],
                    "run_index": run_index,
                    "observed_negotiation_rounds": result["negotiation_rounds"],
                    "runtime_ms": runtime_ms,
                    "final_outcome": offer.status,
                }
            )

    write_csv(
        RUNTIME_NEGOTIATION_CSV,
        [
            "case_id",
            "target_round_bucket",
            "run_index",
            "observed_negotiation_rounds",
            "runtime_ms",
            "final_outcome",
        ],
        rows,
    )
    return RUNTIME_NEGOTIATION_CSV


def summarise_runtime_by_negotiation_rounds(
    runs_csv_path: Path = RUNTIME_NEGOTIATION_CSV,
    summary_csv_path: Path = RUNTIME_NEGOTIATION_SUMMARY_CSV,
) -> Path:
    grouped: Dict[int, List[Dict[str, str]]] = {0: [], 1: [], 2: [], 3: []}
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["target_round_bucket"])].append(row)

    rows: List[Dict[str, object]] = []
    for target_round_bucket in sorted(grouped):
        bucket_rows = grouped[target_round_bucket]
        runtimes = [float(row["runtime_ms"]) for row in bucket_rows]
        stats = extended_stats(runtimes)
        observed_rounds = sorted({row["observed_negotiation_rounds"] for row in bucket_rows})
        outcomes = sorted({row["final_outcome"] for row in bucket_rows})
        note = ""
        if target_round_bucket == 3:
            note = (
                "True third relaxation cycle is unreachable under MAX_NEGOTIATION_ROUNDS=3; "
                "this bucket represents runs that exhaust the three-offer-round budget."
            )
        rows.append(
            {
                "target_round_bucket": target_round_bucket,
                "observed_negotiation_rounds": "|".join(observed_rounds),
                "final_outcomes": "|".join(outcomes),
                "mean_runtime_ms": stats["mean"],
                "median_runtime_ms": stats["median"],
                "p95_runtime_ms": stats["p95"],
                "max_runtime_ms": round(max(runtimes), 3) if runtimes else 0.0,
                "note": note,
            }
        )

    write_csv(
        summary_csv_path,
        [
            "target_round_bucket",
            "observed_negotiation_rounds",
            "final_outcomes",
            "mean_runtime_ms",
            "median_runtime_ms",
            "p95_runtime_ms",
            "max_runtime_ms",
            "note",
        ],
        rows,
    )
    return summary_csv_path


def main() -> None:
    scaling_runs_path = run_runtime_scaling_experiment()
    scaling_summary_path = summarise_runtime_scaling(scaling_runs_path)
    negotiation_runs_path = run_runtime_by_negotiation_rounds()
    negotiation_summary_path = summarise_runtime_by_negotiation_rounds(negotiation_runs_path)

    print(f"Wrote {scaling_runs_path}")
    print(f"Wrote {scaling_summary_path}")
    print(f"Wrote {negotiation_runs_path}")
    print(f"Wrote {negotiation_summary_path}")


if __name__ == "__main__":
    main()
