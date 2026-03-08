from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Ensure Matplotlib uses a non-interactive backend for server-side rendering.
matplotlib.use("Agg")


@dataclass(frozen=True)
class DiagramConfig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    FIGURES_DIR: Path = PROJECT_ROOT / "output"
    OUTPUT_FILENAME: str = "architecture_dataflow.png"
    OUTPUT_PATH: Path = FIGURES_DIR / OUTPUT_FILENAME

    COLOR_PALETTE: Dict[str, str] = None
    DIAGRAM_DIMENSIONS: Dict[str, float] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "COLOR_PALETTE",
            {
                "background": "#ffffff",
                "edge": "#243b53",
                "text": "#102a43",
                "subtext": "#52606d",
                "panel_edge": "#d3d9e3",
                "dash_line": "#9fb3c8",
                "node_fill": "#ffffff",
                "panel_fill": "#f5f7fa",
            },
        )
        object.__setattr__(
            self,
            "DIAGRAM_DIMENSIONS",
            {
                # Bigger canvas = more room for text
                "figure_width": 10.5,
                "figure_height": 7.2,

                "x_lim_min": 0.0,
                "x_lim_max": 1.0,
                "y_lim_min": 0.0,
                "y_lim_max": 1.0,

                # Slightly wider nodes helps subtitles
                "node_width": 0.18,
                "node_height": 0.095,
                "shared_node_width": 0.19,

                "panel_padding": 0.010,
                "panel_rounding_size": 0.02,
                "node_rounding_size": 0.02,

                "arrow_mutation_scale": 12,
                "arrow_linewidth": 1.1,
                "panel_linewidth": 0.9,
                "node_linewidth": 1.15,

                # Text sizes
                "font_size_title": 9.6,
                "font_size_subtitle": 7.6,
                "font_size_panel_title": 9.0,

                "panel_title_offset_x": 0.012,
                "panel_title_offset_y": 0.032,

                # Title/subtitle vertical positioning within node
                "node_title_offset_y_ratio": 0.67,
                "node_subtitle_offset_y_ratio": 0.30,

                # Wrapping control (tune if you add longer labels)
                "wrap_chars_node": 22,
                "wrap_chars_shared": 20,
            },
        )


CFG = DiagramConfig()


def _draw_panel(ax: plt.Axes, xy: Tuple[float, float], width: float, height: float, title: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=(
            f"round,pad={CFG.DIAGRAM_DIMENSIONS['panel_padding']},"
            f"rounding_size={CFG.DIAGRAM_DIMENSIONS['panel_rounding_size']}"
        ),
        linewidth=CFG.DIAGRAM_DIMENSIONS["panel_linewidth"],
        edgecolor=CFG.COLOR_PALETTE["panel_edge"],
        facecolor=CFG.COLOR_PALETTE["panel_fill"],
    )
    ax.add_patch(patch)

    ax.text(
        xy[0] + CFG.DIAGRAM_DIMENSIONS["panel_title_offset_x"],
        xy[1] + height - CFG.DIAGRAM_DIMENSIONS["panel_title_offset_y"],
        title,
        ha="left",
        va="center",
        fontsize=CFG.DIAGRAM_DIMENSIONS["font_size_panel_title"],
        weight="bold",
        color=CFG.COLOR_PALETTE["subtext"],
    )


def _wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=True))


def _draw_node(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    wrap_chars: int = 22,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=(
            f"round,pad={CFG.DIAGRAM_DIMENSIONS['panel_padding']},"
            f"rounding_size={CFG.DIAGRAM_DIMENSIONS['node_rounding_size']}"
        ),
        linewidth=CFG.DIAGRAM_DIMENSIONS["node_linewidth"],
        edgecolor=CFG.COLOR_PALETTE["edge"],
        facecolor=CFG.COLOR_PALETTE["node_fill"],
    )
    ax.add_patch(patch)

    ax.text(
        xy[0] + width / 2,
        xy[1] + height * CFG.DIAGRAM_DIMENSIONS["node_title_offset_y_ratio"],
        title,
        ha="center",
        va="center",
        fontsize=CFG.DIAGRAM_DIMENSIONS["font_size_title"],
        weight="bold",
        color=CFG.COLOR_PALETTE["text"],
    )

    subtitle_wrapped = _wrap(subtitle, wrap_chars)
    if subtitle_wrapped:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height * CFG.DIAGRAM_DIMENSIONS["node_subtitle_offset_y_ratio"],
            subtitle_wrapped,
            ha="center",
            va="center",
            fontsize=CFG.DIAGRAM_DIMENSIONS["font_size_subtitle"],
            color=CFG.COLOR_PALETTE["subtext"],
            linespacing=1.15,
        )


def _draw_arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], linestyle: str = "-") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=CFG.DIAGRAM_DIMENSIONS["arrow_mutation_scale"],
        linewidth=CFG.DIAGRAM_DIMENSIONS["arrow_linewidth"],
        linestyle=linestyle,
        color=CFG.COLOR_PALETTE["edge"] if linestyle == "-" else CFG.COLOR_PALETTE["dash_line"],
    )
    ax.add_patch(arrow)


def generate_architecture_diagram() -> None:
    CFG.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(
        figsize=(CFG.DIAGRAM_DIMENSIONS["figure_width"], CFG.DIAGRAM_DIMENSIONS["figure_height"]),
        constrained_layout=True,
    )

    fig.patch.set_facecolor(CFG.COLOR_PALETTE["background"])
    ax.set_facecolor(CFG.COLOR_PALETTE["background"])
    ax.set_xlim(CFG.DIAGRAM_DIMENSIONS["x_lim_min"], CFG.DIAGRAM_DIMENSIONS["x_lim_max"])
    ax.set_ylim(CFG.DIAGRAM_DIMENSIONS["y_lim_min"], CFG.DIAGRAM_DIMENSIONS["y_lim_max"])
    ax.axis("off")

    # More vertical room between panels (no overlaps)
    _draw_panel(ax, (0.05, 0.76), 0.90, 0.18, "Interface and Orchestration")
    _draw_panel(ax, (0.05, 0.54), 0.90, 0.18, "Intake and Routing")
    _draw_panel(ax, (0.05, 0.30), 0.90, 0.20, "Deterministic Scheduling")
    _draw_panel(ax, (0.05, 0.08), 0.90, 0.18, "Shared Configuration and Artefacts")

    node_w = CFG.DIAGRAM_DIMENSIONS["node_width"]
    node_h = CFG.DIAGRAM_DIMENSIONS["node_height"]
    shared_node_w = CFG.DIAGRAM_DIMENSIONS["shared_node_width"]

    # Top row
    y_top = 0.80
    browser_node = (0.07, y_top, node_w, node_h)
    api_node = (0.30, y_top, node_w, node_h)
    orchestrator_node = (0.53, y_top, node_w, node_h)
    outputs_node = (0.76, y_top, node_w, node_h)

    _draw_node(ax, browser_node[:2], node_w, node_h, "Browser UI", "HTML forms and audit view",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, api_node[:2], node_w, node_h, "FastAPI Service", "Typed HTTP endpoints",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, orchestrator_node[:2], node_w, node_h, "DemoOrchestrator", "Run state and agent calls",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, outputs_node[:2], node_w, node_h, "User / Admin", "Booking and audit output",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])

    _draw_arrow(ax, (browser_node[0] + node_w, y_top + node_h / 2), (api_node[0], y_top + node_h / 2))
    _draw_arrow(ax, (api_node[0] + node_w, y_top + node_h / 2), (orchestrator_node[0], y_top + node_h / 2))
    _draw_arrow(ax, (orchestrator_node[0] + node_w, y_top + node_h / 2), (outputs_node[0], y_top + node_h / 2))

    # Middle row
    y_mid = 0.58
    safety_node = (0.13, y_mid, node_w, node_h)
    receptionist_node = (0.42, y_mid, node_w, node_h)
    triage_node = (0.71, y_mid, node_w, node_h)

    _draw_node(ax, safety_node[:2], node_w, node_h, "SafetyGateAgent", "Red-flag escalation",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, receptionist_node[:2], node_w, node_h, "ReceptionistAgent", "Form/LLM intake → PatientPreferences",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, triage_node[:2], node_w, node_h, "TriageRoutingAgent", "Complaint → service via rules",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])

    _draw_arrow(ax, (orchestrator_node[0] + node_w / 2, y_top), (receptionist_node[0] + node_w / 2, y_mid + node_h))
    _draw_arrow(ax, (receptionist_node[0] + node_w / 2, y_mid + node_h), (safety_node[0] + node_w / 2, y_mid + node_h))
    _draw_arrow(ax, (safety_node[0] + node_w, y_mid + node_h / 2), (receptionist_node[0], y_mid + node_h / 2))
    _draw_arrow(ax, (receptionist_node[0] + node_w, y_mid + node_h / 2), (triage_node[0], y_mid + node_h / 2))
    _draw_arrow(ax, (triage_node[0] + node_w / 2, y_mid), (orchestrator_node[0] + node_w / 2, y_top))

    # Bottom row
    y_bot = 0.37
    rota_node = (0.13, y_bot, node_w, node_h)
    patient_node = (0.42, y_bot, node_w, node_h)
    allocator_node = (0.71, y_bot, node_w, node_h)

    _draw_node(ax, rota_node[:2], node_w, node_h, "RotaAgent", "Deterministic slot inventory",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, patient_node[:2], node_w, node_h, "PatientAgent", "Utility scoring and relaxations",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])
    _draw_node(ax, allocator_node[:2], node_w, node_h, "MasterAllocatorAgent", "Hard constraints and final choice",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_node"])

    _draw_arrow(ax, (orchestrator_node[0] + node_w / 2, y_top), (patient_node[0] + node_w / 2, y_bot + node_h))
    _draw_arrow(ax, (rota_node[0] + node_w, y_bot + node_h / 2), (patient_node[0], y_bot + node_h / 2))
    _draw_arrow(ax, (patient_node[0] + node_w, y_bot + node_h / 2), (allocator_node[0], y_bot + node_h / 2))
    _draw_arrow(ax, (allocator_node[0] + node_w / 2, y_bot), (orchestrator_node[0] + node_w / 2, y_top))

    # Shared artefacts row
    y_shared = 0.13
    schemas_node = (0.07, y_shared, shared_node_w, node_h)
    rules_node = (0.30, y_shared, shared_node_w, node_h)
    config_node = (0.53, y_shared, shared_node_w, node_h)
    tests_node = (0.76, y_shared, shared_node_w, node_h)

    _draw_node(ax, schemas_node[:2], shared_node_w, node_h, "schemas.py", "Pydantic models",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_shared"])
    _draw_node(ax, rules_node[:2], shared_node_w, node_h, "routing_rules.json", "Routing policy",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_shared"])
    _draw_node(ax, config_node[:2], shared_node_w, node_h, "config.py", "Weights, horizons, caps",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_shared"])
    _draw_node(ax, tests_node[:2], shared_node_w, node_h, "experiments/ + tests/", "Validation and regression",
               wrap_chars=CFG.DIAGRAM_DIMENSIONS["wrap_chars_shared"])

    # Dashed vertical connectors
    _draw_arrow(ax, (schemas_node[0] + shared_node_w / 2, y_shared + node_h), (api_node[0] + node_w / 2, y_top),
               linestyle="--")
    _draw_arrow(ax, (rules_node[0] + shared_node_w / 2, y_shared + node_h), (triage_node[0] + node_w / 2, y_mid),
               linestyle="--")
    _draw_arrow(ax, (config_node[0] + shared_node_w / 2, y_shared + node_h), (patient_node[0] + node_w / 2, y_bot),
               linestyle="--")
    _draw_arrow(ax, (tests_node[0] + shared_node_w / 2, y_shared + node_h), (orchestrator_node[0] + node_w / 2, y_top),
               linestyle="--")

    fig.savefig(CFG.OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=220)
    plt.close(fig)
    print(f"Diagram saved to: {CFG.OUTPUT_PATH}")


if __name__ == "__main__":
    generate_architecture_diagram()