"""Persistent, one-shot price alerts checked while the API is running."""

import logging
from datetime import datetime, timezone
from core.database import connect
from services import market

logger = logging.getLogger(__name__)


def check_alerts():
    with connect() as db:
        alerts = [
            dict(row) for row in db.execute("SELECT * FROM alerts WHERE active=1")
        ]
    quotes = {}
    for alert in alerts:
        symbol = alert["ticker"]
        if symbol not in quotes:
            try:
                quotes[symbol] = market.history(symbol, 2)["price"]
            except Exception:
                logger.warning("Could not evaluate alerts for %s", symbol)
                quotes[symbol] = None
        price = quotes[symbol]
        if price is None:
            continue
        matched = (
            price >= alert["threshold"]
            if alert["direction"] == "above"
            else price <= alert["threshold"]
        )
        if matched:
            with connect() as db:
                db.execute(
                    "UPDATE alerts SET active=0, triggered_at=? WHERE id=? AND active=1",
                    (datetime.now(timezone.utc).isoformat(), alert["id"]),
                )
