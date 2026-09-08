"""Explicit local defaults; existing cloud credentials never enable a provider implicitly."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DATA_MODE = os.getenv("STOCKASSIST_DATA_MODE", "demo")
if DATA_MODE not in {"demo", "massive", "ibkr"}:
    raise ValueError("STOCKASSIST_DATA_MODE must be demo, massive, or ibkr")
DATABASE_PATH = os.getenv(
    "STOCKASSIST_DB",
    str(Path(__file__).resolve().parents[1] / "data" / f"{DATA_MODE}.sqlite3"),
)
