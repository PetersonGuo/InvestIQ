"""Atomic paper fills at the last available daily closing price."""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from uuid import uuid4
from fastapi import HTTPException
from core.database import connect
from services import market


def place_order(query):
    quote = market.history(query.ticker, 2)
    price = int(
        (Decimal(str(quote["price"])) * 100).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    if price <= 0:
        raise HTTPException(422, "Price must be positive.")
    total = price * query.quantity
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        cash = db.execute("SELECT cash_cents FROM account WHERE id=1").fetchone()[0]
        position = db.execute(
            "SELECT * FROM positions WHERE ticker=?", (query.ticker,)
        ).fetchone()
        held = position["quantity"] if position else 0
        cost = position["cost_cents"] if position else 0
        if query.side == "buy":
            if total > cash:
                raise HTTPException(409, "Insufficient paper cash.")
            cash -= total
            held += query.quantity
            cost += total
        else:
            if query.quantity > held:
                raise HTTPException(
                    409, "Insufficient shares. Short selling is not supported."
                )
            cash += total
            cost = round(cost * (held - query.quantity) / held)
            held -= query.quantity
        db.execute("UPDATE account SET cash_cents=? WHERE id=1", (cash,))
        if held:
            db.execute(
                "INSERT OR REPLACE INTO positions VALUES (?, ?, ?)",
                (query.ticker, held, cost),
            )
        else:
            db.execute("DELETE FROM positions WHERE ticker=?", (query.ticker,))
        order = {
            "id": str(uuid4()),
            "ticker": query.ticker,
            "side": query.side,
            "quantity": query.quantity,
            "price_cents": price,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "price_as_of": quote["as_of"],
        }
        db.execute(
            "INSERT INTO orders VALUES (:id,:ticker,:side,:quantity,:price_cents,:created_at,:price_as_of)",
            order,
        )
    return order


def portfolio():
    with connect() as db:
        cash = db.execute("SELECT cash_cents FROM account WHERE id=1").fetchone()[0]
        positions = [
            dict(row) for row in db.execute("SELECT * FROM positions ORDER BY ticker")
        ]
        orders = [
            dict(row)
            for row in db.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT 50"
            )
        ]
    return {"cash_cents": cash, "positions": positions, "orders": orders}
