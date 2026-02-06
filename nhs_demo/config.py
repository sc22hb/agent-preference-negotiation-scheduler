from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
ROUTING_RULES_PATH = DATA_DIR / "routing_rules.json"

MAX_NEGOTIATION_ROUNDS = 3
DEFAULT_DATE_HORIZON_DAYS = 10
DATE_RANGE_EXTENSION_DAYS = 3

# Fixed synthetic base keeps tests and demos deterministic.
SYNTHETIC_BASE_DATETIME = datetime(2026, 2, 9, 8, 0)

UTILITY_WEIGHTS: Dict[str, int] = {
    "preferred_modality": 30,
    "preferred_day": 20,
    "preferred_period": 18,
    "soonest": 20,
}

BLOCKER_SEVERITY: Dict[str, int] = {
    "modality_not_allowed": 5,
    "excluded_modality": 4,
    "excluded_day": 3,
    "excluded_period": 3,
    "outside_horizon": 2,
}

RELAXATION_ORDER = [
    "relax_excluded_periods",
    "relax_excluded_days",
    "relax_excluded_modalities",
    "extend_date_horizon",
]
