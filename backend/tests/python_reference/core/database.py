import sqlite3
from contextlib import contextmanager
from pathlib import Path
from core.config import DATABASE_PATH


@contextmanager
def connect():
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE_PATH, timeout=20)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK(id=1), cash_cents INTEGER NOT NULL);
        INSERT OR IGNORE INTO account VALUES (1, 10000000);
        CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, quantity INTEGER NOT NULL, cost_cents INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL, price_cents INTEGER NOT NULL, created_at TEXT NOT NULL, price_as_of TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, direction TEXT NOT NULL, threshold REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1, triggered_at TEXT, created_at TEXT NOT NULL);
        """)
