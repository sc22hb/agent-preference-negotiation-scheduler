from __future__ import annotations

from experiments.results_evaluation import (
    evaluate_ablation,
    evaluate_baselines,
    evaluate_constraint_satisfaction,
    evaluate_contention_and_fairness,
)


def parse_percent(value: str) -> float:
    return float(value.rstrip("%"))


def parse_mean_utility(cell: str) -> float:
    return float(cell.split(" ", 1)[0])


def test_constraint_satisfaction_has_zero_violations() -> None:
    results = evaluate_constraint_satisfaction()
    total_row = results[results["Constraint Type"] == "Total"].iloc[0]
    assert int(total_row["Violations"]) == 0


def test_scheduler_beats_simple_baselines_on_mean_utility() -> None:
    results = evaluate_baselines().set_index("Policy")
    scheduler = parse_mean_utility(results.loc["Your scheduler", "Mean Utility (95% CI)"])
    preferred_modality = parse_mean_utility(
        results.loc["Preferred modality earliest", "Mean Utility (95% CI)"]
    )
    earliest = parse_mean_utility(results.loc["Earliest feasible", "Mean Utility (95% CI)"])
    random = parse_mean_utility(results.loc["Random feasible", "Mean Utility (95% CI)"])

    assert scheduler > preferred_modality > earliest > random


def test_scarcity_first_improves_strict_group_booking_rate() -> None:
    results, _request_type_df, _figure_path = evaluate_contention_and_fairness()
    results = results.set_index("Metric")
    strict_fcfs = parse_percent(results.loc["Strict group booking rate", "FCFS"])
    strict_scarcity = parse_percent(results.loc["Strict group booking rate", "Scarcity-First"])
    assert strict_scarcity > strict_fcfs


def test_no_relaxation_reduces_booking_rate() -> None:
    results = evaluate_ablation().set_index("Variant")
    full_rate = parse_percent(results.loc["Full system", "Booking Rate"])
    no_relax_rate = parse_percent(results.loc["No relaxation", "Booking Rate"])
    assert no_relax_rate < full_rate
