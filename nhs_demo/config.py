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
MAX_PREFERENCE_HORIZON_DAYS = 30
SLOT_DATABASE_HORIZON_DAYS = MAX_PREFERENCE_HORIZON_DAYS + DATE_RANGE_EXTENSION_DAYS

# Fixed synthetic base keeps tests and demos deterministic.
SYNTHETIC_BASE_DATETIME = datetime(2026, 2, 9, 8, 0)

UTILITY_WEIGHTS: Dict[str, int] = {
    "preferred_modality": 30,
    "preferred_day": 20,
    "preferred_period": 18,
    "preferred_day_period": 14,
    "adjacent_preferred_day": 6,
    "soonest": 20,
}

MISMATCH_PENALTIES: Dict[str, float] = {
    "modality": 0.35,
    "day": 0.30,
    "period": 0.28,
}

BLOCKER_SEVERITY: Dict[str, int] = {
    "modality_not_allowed": 5,
    "excluded_day_period": 5,
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
