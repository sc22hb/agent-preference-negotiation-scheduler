from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    RESULTS_DIR,
    audit_hash,
    ensure_results_dir,
    most_common_frequency,
    run_structured_workflow,
    write_csv,
)
from experiments.scenario_definitions import DETERMINISM_SCENARIOS, TIE_BREAK_SCENARIO
from nhs_demo.orchestrator import DemoOrchestrator
from nhs_demo.schemas import Slot

K_RUNS = 100
DETERMINISM_RUNS_CSV = RESULTS_DIR / "determinism_runs.csv"
DETERMINISM_SUMMARY_CSV = RESULTS_DIR / "determinism_summary.csv"
TIE_BREAK_RUNS_CSV = RESULTS_DIR / "tie_breaking_runs.csv"
TIE_BREAK_SUMMARY_CSV = RESULTS_DIR / "tie_breaking_summary.csv"


def run_replay_determinism(k_runs: int = K_RUNS) -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []

    for scenario in DETERMINISM_SCENARIOS:
        for run_index in range(1, k_runs + 1):
            orchestrator = DemoOrchestrator()
            result = run_structured_workflow(
                orchestrator,
                scenario,
                use_routing_hint=True,
                relaxation_policy=None,
            )
            offer = result["offer"]
            audit = result["audit"]
            rows.append(
                {
                    "scenario_name": scenario["scenario_id"],
                    "run_index": run_index,
                    "final_slot_id": offer.booking.slot_id if offer.booking else "None",
                    "audit_hash": audit_hash(audit),
                    "status": offer.status,
                }
            )

    write_csv(
        DETERMINISM_RUNS_CSV,
        ["scenario_name", "run_index", "final_slot_id", "audit_hash", "status"],
        rows,
    )
    return DETERMINISM_RUNS_CSV


def aggregate_determinism_results(
    runs_csv_path: Path = DETERMINISM_RUNS_CSV,
    summary_csv_path: Path = DETERMINISM_SUMMARY_CSV,
) -> Path:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"slots": [], "hashes": []})
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["scenario_name"]]["slots"].append(row["final_slot_id"])
            grouped[row["scenario_name"]]["hashes"].append(row["audit_hash"])

    summary_rows: List[Dict[str, object]] = []
    for scenario_name in sorted(grouped):
        slots = grouped[scenario_name]["slots"]
        hashes = grouped[scenario_name]["hashes"]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "num_runs": len(slots),
                "unique_slot_ids": len(set(slots)),
                "identical_slot_id_runs": most_common_frequency(slots),
                "unique_audit_hashes": len(set(hashes)),
                "identical_audit_hash_runs": most_common_frequency(hashes),
            }
        )

    if summary_rows:
        total_runs = sum(int(row["num_runs"]) for row in summary_rows)
        summary_rows.append(
            {
                "scenario_name": "OVERALL",
                "num_runs": total_runs,
                "unique_slot_ids": max(int(row["unique_slot_ids"]) for row in summary_rows),
                "identical_slot_id_runs": sum(int(row["identical_slot_id_runs"]) for row in summary_rows),
                "unique_audit_hashes": max(int(row["unique_audit_hashes"]) for row in summary_rows),
                "identical_audit_hash_runs": sum(int(row["identical_audit_hash_runs"]) for row in summary_rows),
            }
        )

    write_csv(
        summary_csv_path,
        [
            "scenario_name",
            "num_runs",
            "unique_slot_ids",
            "identical_slot_id_runs",
            "unique_audit_hashes",
            "identical_audit_hash_runs",
        ],
        summary_rows,
    )
    return summary_csv_path


def synthetic_tie_break_slots() -> List[Slot]:
    base_time = datetime(2026, 2, 9, 9, 0)
    return [
        Slot(
            slot_id="TIE-SLOT-C",
            service_type="GP",
            clinician_id="GP-C",
            site="Leeds General Infirmary",
            modality="phone",
            start_time=base_time + timedelta(hours=2),
            duration_minutes=15,
        ),
        Slot(
            slot_id="TIE-SLOT-A",
            service_type="GP",
            clinician_id="GP-A",
            site="Leeds General Infirmary",
            modality="phone",
            start_time=base_time,
            duration_minutes=15,
        ),
        Slot(
            slot_id="TIE-SLOT-B",
            service_type="GP",
            clinician_id="GP-B",
            site="Leeds General Infirmary",
            modality="phone",
            start_time=base_time + timedelta(hours=1),
            duration_minutes=15,
        ),
    ]


def run_tie_breaking_experiment(k_runs: int = K_RUNS) -> Path:
    ensure_results_dir()
    rows: List[Dict[str, object]] = []
    candidate_slots = synthetic_tie_break_slots()

    for run_index in range(1, k_runs + 1):
        orchestrator = DemoOrchestrator()
        result = run_structured_workflow(
            orchestrator,
            TIE_BREAK_SCENARIO,
            use_routing_hint=True,
            candidate_slots=candidate_slots,
        )
        booking = result["offer"].booking
        rows.append(
            {
                "run_index": run_index,
                "chosen_slot_id": booking.slot_id if booking else "None",
                "tie_break_reason": booking.tie_break_reason if booking else "None",
            }
        )

    write_csv(TIE_BREAK_RUNS_CSV, ["run_index", "chosen_slot_id", "tie_break_reason"], rows)
    return TIE_BREAK_RUNS_CSV


def summarise_tie_breaking(
    runs_csv_path: Path = TIE_BREAK_RUNS_CSV,
    summary_csv_path: Path = TIE_BREAK_SUMMARY_CSV,
) -> Path:
    chosen_slots: List[str] = []
    with runs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            chosen_slots.append(row["chosen_slot_id"])

    summary_row = {
        "num_runs": len(chosen_slots),
        "unique_chosen_slots": len(set(chosen_slots)),
        "identical_slot_choice_runs": most_common_frequency(chosen_slots),
        "chosen_slot_id": chosen_slots[0] if chosen_slots else "None",
    }
    write_csv(
        summary_csv_path,
        ["num_runs", "unique_chosen_slots", "identical_slot_choice_runs", "chosen_slot_id"],
        [summary_row],
    )
    return summary_csv_path


def main() -> None:
    runs_path = run_replay_determinism()
    summary_path = aggregate_determinism_results(runs_path)
    tie_runs_path = run_tie_breaking_experiment()
    tie_summary_path = summarise_tie_breaking(tie_runs_path)

    print(f"Wrote {runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tie_runs_path}")
    print(f"Wrote {tie_summary_path}")


if __name__ == "__main__":
    main()
