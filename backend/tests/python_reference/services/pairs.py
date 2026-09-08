"""Pair statistics and persistent, one-shot or edge-triggered pair alerts."""

import json
import math
from datetime import date, datetime, timezone
from statistics import mean, pstdev
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from core.database import connect
from services import market


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS pair_alerts (
            id TEXT PRIMARY KEY, config_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
            version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            last_checked_at TEXT, last_as_of TEXT, last_matched INTEGER NOT NULL DEFAULT 0,
            latest_json TEXT, last_error TEXT, triggered_at TEXT);
        CREATE TABLE IF NOT EXISTS pair_events (
            id TEXT PRIMARY KEY, alert_id TEXT NOT NULL, version INTEGER NOT NULL,
            as_of TEXT NOT NULL, created_at TEXT NOT NULL, event_json TEXT NOT NULL,
            UNIQUE(alert_id, version, as_of));
        """)


def now():
    return datetime.now(timezone.utc).isoformat()


def calculate(config, history_a, history_b, today=None):
    """Score the latest common close against PRECEDING aligned observations."""
    a = {bar["time"]: bar["close"] for bar in history_a}
    b = {bar["time"]: bar["close"] for bar in history_b}
    if not a or not b:
        raise HTTPException(422, "Both stocks need completed daily price history.")
    if max(a) != max(b):
        raise HTTPException(
            409,
            f"Latest bar dates differ: {config['ticker_a']} {max(a)}, {config['ticker_b']} {max(b)}. Waiting for aligned data.",
        )
    dates = sorted(a.keys() & b.keys())
    current = today or datetime.now(ZoneInfo("America/New_York")).date()
    as_of = date.fromisoformat(dates[-1])
    if as_of >= current or (current - as_of).days > 7:
        raise HTTPException(
            409,
            "Pair data must contain completed closes no more than seven calendar days old. Waiting for fresh history.",
        )
    if any(not math.isfinite(p) or p <= 0 for d in dates for p in (a[d], b[d])):
        raise HTTPException(502, "Pair history contains invalid prices.")
    lookback = config["lookback"]
    metric = config["metric"]
    required = lookback + 1 if metric == "zscore" else 2
    if len(dates) < required:
        raise HTTPException(
            422,
            f"Need {required} aligned trading dates; only {len(dates)} are available.",
        )
    ratios = [a[d] / b[d] for d in dates]
    spreads = [math.log(a[d]) - config["hedge_ratio"] * math.log(b[d]) for d in dates]
    spread_mean = spread_std = z_score = None
    points = []
    if metric == "zscore":
        baseline = spreads[-lookback - 1 : -1]
        spread_mean, spread_std = mean(baseline), pstdev(baseline)
        if spread_std <= 1e-12:
            raise HTTPException(
                422,
                "The spread has no measurable variation in this window; its z-score is undefined. Try a different pair or lookback.",
            )
        z_score = (spreads[-1] - spread_mean) / spread_std
        for index in range(max(lookback, len(dates) - 90), len(dates)):
            training = spreads[index - lookback : index]
            deviation = pstdev(training)
            if deviation > 1e-12:
                points.append(
                    {
                        "time": dates[index],
                        "value": (spreads[index] - mean(training)) / deviation,
                    }
                )
        value = z_score
    else:
        points = [
            {"time": d, "value": ratios[index]}
            for index, d in enumerate(dates)
            if index >= len(dates) - 90
        ]
        value = ratios[-1]
    if not math.isfinite(value):
        raise HTTPException(502, "Pair metric is not finite.")
    return {
        "ticker_a": config["ticker_a"],
        "ticker_b": config["ticker_b"],
        "metric": metric,
        "value": value,
        "ratio": ratios[-1],
        "spread": spreads[-1],
        "z_score": z_score,
        "mean": spread_mean,
        "std": spread_std,
        "hedge_ratio": config["hedge_ratio"],
        "lookback": lookback,
        "as_of": dates[-1],
        "price_a": a[dates[-1]],
        "price_b": b[dates[-1]],
        "aligned_bars": len(dates),
        "baseline_start": dates[-lookback - 1] if metric == "zscore" else None,
        "baseline_end": dates[-2] if metric == "zscore" else None,
        "matched": matches(value, config),
        "points": points,
    }


def matches(value, config):
    condition, threshold = config["condition"], config["threshold"]
    if condition == "above":
        return value >= threshold
    if condition == "below":
        return value <= threshold
    if condition == "outside":
        return abs(value) >= threshold
    return abs(value) <= threshold


def analyze(config, cache=None):
    cache = {} if cache is None else cache
    histories = []
    sources = []
    for key in ("ticker_a", "ticker_b"):
        symbol = config[key]
        if symbol not in cache:
            try:
                cache[symbol] = market.history(symbol, 500)
            except Exception as exc:
                cache[symbol] = exc
        data = cache[symbol]
        if isinstance(data, Exception):
            raise data
        histories.append(data["bars"])
        sources.append(data["source"])
    if sources[0] != sources[1]:
        raise HTTPException(409, "Both legs must use the same market-data source.")
    result = calculate(config, *histories)
    result["source"] = sources[0]
    return result


def decode(row):
    row = dict(row)
    config = json.loads(row.pop("config_json"))
    row["latest"] = json.loads(row["latest_json"]) if row["latest_json"] else None
    row.pop("latest_json")
    return {**row, **config}


def list_alerts():
    with connect() as db:
        return [
            decode(row)
            for row in db.execute("SELECT * FROM pair_alerts ORDER BY created_at DESC")
        ]


def get(alert_id):
    with connect() as db:
        row = db.execute("SELECT * FROM pair_alerts WHERE id=?", (alert_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Pair alert not found.")
    return decode(row)


def save(config, alert_id=None):
    if alert_id:
        get(alert_id)
    snapshot = analyze(config)
    stamp = now()
    with connect() as db:
        if alert_id:
            cursor = db.execute(
                """UPDATE pair_alerts SET config_json=?, active=1, version=version+1,
                updated_at=?, last_checked_at=NULL, last_as_of=NULL, last_matched=0,
                latest_json=?, last_error=NULL, triggered_at=NULL WHERE id=?""",
                (json.dumps(config), stamp, json.dumps(snapshot), alert_id),
            )
            if not cursor.rowcount:
                raise HTTPException(404, "Pair alert not found.")
        else:
            alert_id = str(uuid4())
            db.execute(
                """INSERT INTO pair_alerts (id,config_json,created_at,updated_at,latest_json)
                VALUES (?,?,?,?,?)""",
                (alert_id, json.dumps(config), stamp, stamp, json.dumps(snapshot)),
            )
    return get(alert_id)


def set_active(alert_id, active):
    with connect() as db:
        cursor = db.execute(
            """UPDATE pair_alerts SET active=?, version=version+1, updated_at=?,
            last_as_of=NULL, last_matched=0, last_error=NULL, triggered_at=NULL WHERE id=?""",
            (int(active), now(), alert_id),
        )
        if not cursor.rowcount:
            raise HTTPException(404, "Pair alert not found.")
    return get(alert_id)


def delete(alert_id):
    with connect() as db:
        if not db.execute("DELETE FROM pair_alerts WHERE id=?", (alert_id,)).rowcount:
            raise HTTPException(404, "Pair alert not found.")


def events():
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM pair_events ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return [
        {
            **{key: row[key] for key in ("id", "alert_id", "created_at", "as_of")},
            **json.loads(row["event_json"]),
        }
        for row in rows
    ]


def check_pair_alerts():
    with connect() as db:
        alerts = [
            dict(row) for row in db.execute("SELECT * FROM pair_alerts WHERE active=1")
        ]
    cache = {}
    for row in alerts:
        config = json.loads(row["config_json"])
        stamp = now()
        try:
            snapshot = analyze(config, cache)
        except Exception as exc:
            message = (
                exc.detail
                if isinstance(exc, HTTPException)
                else "Pair data could not be evaluated. Check the market-data connection."
            )
            with connect() as db:
                db.execute(
                    "UPDATE pair_alerts SET last_checked_at=?, last_error=? WHERE id=? AND version=? AND active=1",
                    (stamp, str(message)[:1000], row["id"], row["version"]),
                )
            continue
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT * FROM pair_alerts WHERE id=? AND version=? AND active=1",
                (row["id"], row["version"]),
            ).fetchone()
            if not current:
                continue
            if current["last_as_of"] and snapshot["as_of"] < current["last_as_of"]:
                db.execute(
                    "UPDATE pair_alerts SET last_checked_at=?, last_error=? WHERE id=?",
                    (
                        stamp,
                        "Pair data moved backward in time. Waiting for newer closes.",
                        row["id"],
                    ),
                )
                continue
            new_bar = current["last_as_of"] != snapshot["as_of"]
            fire = new_bar and snapshot["matched"] and not current["last_matched"]
            if fire:
                event = {
                    "config": config,
                    "snapshot": {
                        key: value for key, value in snapshot.items() if key != "points"
                    },
                }
                db.execute(
                    "INSERT OR IGNORE INTO pair_events VALUES (?,?,?,?,?,?)",
                    (
                        str(uuid4()),
                        row["id"],
                        row["version"],
                        snapshot["as_of"],
                        stamp,
                        json.dumps(event),
                    ),
                )
            active = 0 if fire and not config["repeat"] else 1
            db.execute(
                """UPDATE pair_alerts SET active=?, last_checked_at=?, last_as_of=?, last_matched=?,
                latest_json=?, last_error=NULL, triggered_at=CASE WHEN ? THEN ? ELSE triggered_at END WHERE id=? AND version=?""",
                (
                    active,
                    stamp,
                    snapshot["as_of"],
                    int(snapshot["matched"]) if new_bar else current["last_matched"],
                    json.dumps(snapshot),
                    int(fire),
                    stamp,
                    row["id"],
                    row["version"],
                ),
            )
