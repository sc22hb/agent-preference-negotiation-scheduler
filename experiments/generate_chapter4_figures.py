from __future__ import annotations

import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Colourblind-safe Tol muted palette
C_TEAL = "#44AA99"
C_BLUE = "#4477AA"
C_RED = "#CC6677"
C_AMBER = "#DDCC77"
C_PURPLE = "#AA4499"
C_GREY = "#999999"
C_DARK = "#332288"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _setup_figure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
        }
    )


def _annotate_bars(ax, bars, fmt="{:.0f}%", fontsize=8, offset=1.5):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#333333",
            )


# ── Figure 1: Allocation quality and fairness (2 panels) ──────────────────

def generate_allocation_fairness_figure() -> Path:
    allocation_rows = _read_csv(RESULTS_DIR / "allocation_quality.csv")
    fairness_rows = _read_csv(RESULTS_DIR / "fairness_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Left: Utility distribution by group ──
    group_utilities: dict[str, list[float]] = defaultdict(list)
    for row in allocation_rows:
        if row["success_flag"] == "1" and row["total_utility"]:
            group_utilities[row["group_tag"]].append(float(row["total_utility"]))

    ax = axes[0]
    bins = np.arange(40, 155, 10)
    ax.hist(
        group_utilities.get("A", []),
        bins=bins,
        alpha=0.75,
        label="Group A (strict)",
        color=C_RED,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.hist(
        group_utilities.get("B", []),
        bins=bins,
        alpha=0.70,
        label="Group B (flexible)",
        color=C_TEAL,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("Utility score")
    ax.set_ylabel("Count")
    ax.set_title("(a) Utility distribution by availability group")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1)

    # ── Right: Fairness split with CI error bars ──
    ax = axes[1]
    group_labels = [f"Group {row['group_tag']}" for row in fairness_rows]
    success_rates = [float(row["success_rate_pct"]) for row in fairness_rows]
    ci_lower = [float(row.get("success_rate_ci_lower", row["success_rate_pct"])) for row in fairness_rows]
    ci_upper = [float(row.get("success_rate_ci_upper", row["success_rate_pct"])) for row in fairness_rows]
    yerr_lower = [s - lo for s, lo in zip(success_rates, ci_lower)]
    yerr_upper = [hi - s for s, hi in zip(success_rates, ci_upper)]

    bars = ax.bar(
        group_labels,
        success_rates,
        width=0.45,
        color=[C_RED, C_TEAL],
        edgecolor="white",
        linewidth=0.6,
        yerr=[yerr_lower, yerr_upper],
        capsize=6,
        error_kw={"linewidth": 1.4, "color": "#333333"},
    )
    ax.set_ylim(0, 118)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(b) Fairness split with 95% bootstrap CI")

    # Cohen's d annotation
    d_val = fairness_rows[0].get("cohens_d_utility", "")
    if d_val:
        ax.text(
            0.5, 0.93,
            f"Cohen's d (utility) = {d_val}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
            color="#444444",
        )
    _annotate_bars(ax, bars, offset=4)

    fig.tight_layout(w_pad=3.0)
    output_path = FIGURES_DIR / "allocation_fairness.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Figure 2: Negotiation (2 panels) ──────────────────────────────────────

def generate_negotiation_figure() -> Path:
    negotiation_rows = _read_csv(RESULTS_DIR / "negotiation_runs.csv")
    decision_rows = _read_csv(RESULTS_DIR / "negotiation_decision_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Left: Round distribution ──
    ax = axes[0]
    round_counter = Counter(int(row["number_of_rounds"]) for row in negotiation_rows)
    round_labels = sorted(round_counter)
    counts = [round_counter[r] for r in round_labels]
    bars = ax.bar(
        [str(r) for r in round_labels],
        counts,
        color=C_BLUE,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("Negotiation rounds required")
    ax.set_ylabel("Number of cases")
    ax.set_title(f"(a) Round distribution (n={len(negotiation_rows)})")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _annotate_bars(ax, bars, fmt="{:.0f}", offset=0.4)

    # ── Right: Decision-policy sensitivity ──
    ax = axes[1]
    policies = ["accept_all", "accept_first", "reject_all"]
    policy_labels = ["Accept all", "Accept first", "Reject all"]
    policy_colors = [C_TEAL, C_AMBER, C_RED]

    families = sorted({row["negotiation_family"] for row in decision_rows})
    x = np.arange(len(families))
    width = 0.24

    for idx, (policy, label, color) in enumerate(zip(policies, policy_labels, policy_colors)):
        rates = []
        for family in families:
            bucket = [row for row in decision_rows if row["negotiation_family"] == family]
            successes = sum(1 for row in bucket if row[f"{policy}_outcome"] == "booked")
            rates.append((successes / len(bucket)) * 100 if bucket else 0.0)
        ax.bar(
            x + (idx - 1) * width,
            rates,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("_", " ").title() for f in families],
        fontsize=7.5,
    )
    ax.set_ylim(0, 118)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(b) Decision-policy sensitivity by family")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1)

    fig.tight_layout(w_pad=3.0)
    output_path = FIGURES_DIR / "negotiation.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Figure 3: LLM intake evaluation (1 panel) ────────────────────────────

def generate_llm_figure() -> Path:
    comparison_rows = [
        row for row in _read_csv(RESULTS_DIR / "llm_vs_form_comparison.csv")
        if row["field_name"] != "overall"
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2))

    field_labels = [row["field_name"].replace("_", "\n") for row in comparison_rows]
    raw_acc = [float(row["raw_accuracy_pct"]) for row in comparison_rows]
    repaired_acc = [float(row["canonicalized_accuracy_pct"]) for row in comparison_rows]
    x = np.arange(len(field_labels))
    width = 0.36

    ax.bar(
        x - width / 2,
        raw_acc,
        width=width,
        label="Raw LLM output",
        color=C_RED,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        repaired_acc,
        width=width,
        label="After canonicalisation",
        color=C_BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(field_labels, fontsize=7.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("Field-level accuracy: raw LLM vs. after canonicalisation")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1)

    # Delta annotations for large improvements
    for i in range(len(comparison_rows)):
        delta = repaired_acc[i] - raw_acc[i]
        if delta >= 30:
            ax.annotate(
                f"+{delta:.0f} pp",
                xy=(x[i] + width / 2, repaired_acc[i] + 1.5),
                fontsize=7,
                ha="center",
                color=C_DARK,
                weight="bold",
            )

    fig.tight_layout()
    output_path = FIGURES_DIR / "llm_evaluation.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Figure 4: Runtime scaling (1 panel) ───────────────────────────────────

def generate_runtime_figure() -> Path:
    scaling_summary_rows = [
        r for r in _read_csv(RESULTS_DIR / "runtime_scaling_summary.csv")
        if r["N"] != "OLS_FIT"
    ]
    ols_row = next(
        (r for r in _read_csv(RESULTS_DIR / "runtime_scaling_summary.csv") if r["N"] == "OLS_FIT"),
        None,
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))

    slot_sizes = [int(row["N"]) for row in scaling_summary_rows]
    means = [float(row["mean_runtime_ms"]) for row in scaling_summary_rows]
    ci_lo = [float(row["mean_ci_lower"]) for row in scaling_summary_rows]
    ci_hi = [float(row["mean_ci_upper"]) for row in scaling_summary_rows]
    p95 = [float(row["p95_runtime_ms"]) for row in scaling_summary_rows]

    ax.fill_between(slot_sizes, ci_lo, ci_hi, alpha=0.20, color=C_BLUE, label="95% CI (mean)")
    ax.plot(slot_sizes, means, marker="o", linewidth=2, color=C_BLUE, markersize=5, label="Mean", zorder=3)
    ax.plot(slot_sizes, p95, marker="s", linewidth=1.5, color=C_RED, markersize=4, label="P95", zorder=3)

    # OLS regression line
    if ols_row:
        slope = float(ols_row["mean_runtime_ms"])
        intercept = float(ols_row["mean_ci_lower"])
        r_sq = float(ols_row["mean_ci_upper"])
        x_fit = np.linspace(min(slot_sizes), max(slot_sizes), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit, y_fit, "--",
            color=C_GREY,
            linewidth=1.3,
            label=f"OLS fit (R\u00b2 = {r_sq:.3f})",
        )

    ax.set_xlabel("Candidate slots (N)")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime scaling with linear regression fit")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1)

    fig.tight_layout()
    output_path = FIGURES_DIR / "runtime.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ── Figure 5: Constraint-interaction analysis (2 panels) ──────────────────

def generate_constraint_interaction_figure() -> Path:
    rows = _read_csv(RESULTS_DIR / "allocation_quality.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})

    # ── Left: Feasibility cliff scatter ──
    ax = axes[0]

    for row in rows:
        pressure = int(row["constraint_pressure_score"])
        success = row["success_flag"] == "1"
        utility = float(row["total_utility"]) if success else -3  # stack failures below axis
        group = row["group_tag"]
        colour = C_RED if group == "A" else C_TEAL
        marker = "o" if success else "X"
        alpha = 0.85 if success else 0.9
        size = 38 if success else 55
        zorder = 3 if not success else 2
        ax.scatter(
            pressure, utility,
            c=colour, marker=marker, s=size, alpha=alpha,
            edgecolors="white", linewidths=0.4, zorder=zorder,
        )

    # Feasibility cliff shading
    ax.axhspan(-10, 0, color="#fee0d2", alpha=0.35, zorder=0)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=1)
    ax.text(11.2, -5, "Infeasible", fontsize=7.5, color="#999999", style="italic", ha="right")

    # Legend proxies
    ax.scatter([], [], c=C_RED, marker="o", s=38, edgecolors="white", linewidths=0.4, label="Group A (strict)")
    ax.scatter([], [], c=C_TEAL, marker="o", s=38, edgecolors="white", linewidths=0.4, label="Group B (flexible)")
    ax.scatter([], [], c="#666666", marker="X", s=55, edgecolors="white", linewidths=0.4, label="Failed to schedule")

    ax.set_xlabel("Constraint-pressure score")
    ax.set_ylabel("Allocated utility")
    ax.set_title("(a) Feasibility cliff by constraint pressure")
    ax.set_ylim(-12, 145)
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1, fontsize=7.5, loc="upper left")

    # ── Right: Constraint-dimension heatmap ──
    ax = axes[1]

    # Normalise each dimension to 0–1 for the heatmap
    dimensions = [
        ("num_excluded_days", "Excluded\ndays"),
        ("num_preferred_days", "Preferred\ndays"),
        ("strong_modality_preference_flag", "Strong\nmodality"),
        ("preferred_period_count", "Preferred\nperiods"),
        ("date_horizon_days", "Date\nhorizon"),
        ("constraint_pressure_score", "Pressure\nscore"),
    ]
    dim_keys = [d[0] for d in dimensions]
    dim_labels = [d[1] for d in dimensions]

    # Sort rows: Group A first, then B; within group sort by pressure descending
    sorted_rows = sorted(rows, key=lambda r: (r["group_tag"], -int(r["constraint_pressure_score"])))

    # Build matrix
    raw_matrix = np.zeros((len(sorted_rows), len(dim_keys)))
    for i, row in enumerate(sorted_rows):
        for j, key in enumerate(dim_keys):
            raw_matrix[i, j] = float(row[key])

    # Normalise columns to 0–1
    col_min = raw_matrix.min(axis=0)
    col_max = raw_matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1  # avoid division by zero
    norm_matrix = (raw_matrix - col_min) / col_range

    # Add success column (1 = success, 0 = failure)
    success_col = np.array([1.0 if row["success_flag"] == "1" else 0.0 for row in sorted_rows])

    # Plot heatmap
    from matplotlib.colors import LinearSegmentedColormap
    cmap_constraints = LinearSegmentedColormap.from_list("constraint", ["#f7f7f7", C_BLUE], N=256)

    im = ax.imshow(norm_matrix, aspect="auto", cmap=cmap_constraints, interpolation="nearest")

    # Overlay success/failure markers on the right edge
    for i, s in enumerate(success_col):
        colour = C_TEAL if s == 1.0 else C_RED
        ax.plot(len(dim_keys) - 0.5 + 0.35, i, marker="s" if s == 1.0 else "X",
                color=colour, markersize=4.5, clip_on=False, zorder=5)

    # Group boundary line
    group_tags = [row["group_tag"] for row in sorted_rows]
    boundary = next(i for i, g in enumerate(group_tags) if g == "B")
    ax.axhline(boundary - 0.5, color="#333333", linewidth=0.8, linestyle="-")
    ax.text(-0.8, boundary / 2 - 0.5, "Group A", fontsize=7.5, ha="center", va="center",
            rotation=90, color=C_RED, weight="bold")
    ax.text(-0.8, boundary + (len(sorted_rows) - boundary) / 2 - 0.5, "Group B",
            fontsize=7.5, ha="center", va="center", rotation=90, color=C_TEAL, weight="bold")

    ax.set_xticks(range(len(dim_labels)))
    ax.set_xticklabels(dim_labels, fontsize=7, ha="center")
    ax.set_yticks([])
    ax.set_title("(b) Constraint dimensions across 72 test profiles")

    # Outcome legend
    ax.plot([], [], marker="s", color=C_TEAL, linestyle="None", markersize=5, label="Scheduled")
    ax.plot([], [], marker="X", color=C_RED, linestyle="None", markersize=5, label="Failed")
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", framealpha=1,
              fontsize=7, loc="lower right", title="Outcome", title_fontsize=7)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("Normalised intensity", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=2.5)
    output_path = FIGURES_DIR / "constraint_interaction.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_figure()

    outputs = [
        generate_allocation_fairness_figure(),
        generate_negotiation_figure(),
        generate_llm_figure(),
        generate_runtime_figure(),
        generate_constraint_interaction_figure(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
