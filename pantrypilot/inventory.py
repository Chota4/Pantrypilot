"""Inventory store (SQLite) plus pure functions for lot validity, urgency and allocation."""
from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

from .knowledge import INGREDIENTS

GRACE_DAYS = 3  # best-before items may be used (with a quality check) up to 3 days late


def lot_status(lot: dict, on_date: date) -> str:
    """ok | quality_check | discard.  USE-BY past expiry is NEVER usable (safety rule)."""
    days = (lot["expiry"] - on_date).days
    if days >= 0:
        return "ok"
    if lot["etype"] == "use_by":
        return "discard"
    return "quality_check" if days >= -GRACE_DAYS else "discard"


def lot_urgency(lot: dict, on_date: date) -> float:
    s = lot_status(lot, on_date)
    if s == "discard":
        return 0.0
    if s == "quality_check":
        return 1.0
    return math.exp(-max((lot["expiry"] - on_date).days, 0) / 3.0)


def urgency_by_name(lots: list, on_date: date) -> dict:
    out: dict = {}
    for l in lots:
        if l["qty"] > 1e-9:
            u = lot_urgency(l, on_date)
            if u > 0:
                out[l["name"]] = max(out.get(l["name"], 0.0), u)
    return out


def allocate(needs: dict, lots: list, on_date: date, commit: bool = False):
    """Greedy FIFO allocation of `needs` from `lots`. Returns (uses, missing)."""
    uses, missing = [], {}
    for name, need in needs.items():
        remaining = need
        usable = sorted((l for l in lots if l["name"] == name and l["qty"] > 1e-9
                         and lot_status(l, on_date) != "discard"), key=lambda l: l["expiry"])
        for l in usable:
            take = min(l["qty"], remaining)
            if take > 0:
                uses.append(dict(name=name, qty=take, expiry=l["expiry"], etype=l["etype"],
                                 status=lot_status(l, on_date), days_left=(l["expiry"] - on_date).days))
                if commit:
                    l["qty"] -= take
                remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            missing[name] = remaining
    return uses, missing


class Inventory:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS lots(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                        " name TEXT, qty REAL, expiry TEXT, etype TEXT)")

    def add(self, name: str, qty: float, expiry: date, etype: str | None = None) -> int:
        if name not in INGREDIENTS:
            raise ValueError(f"unknown ingredient: {name!r}")
        if not (0 < qty <= 100000):
            raise ValueError("quantity out of range")
        etype = etype or INGREDIENTS[name]["etype"]
        cur = self.db.execute("INSERT INTO lots(name,qty,expiry,etype) VALUES (?,?,?,?)",
                              (name, float(qty), expiry.isoformat(), etype))
        self.db.commit()
        return cur.lastrowid

    def lots(self) -> list:
        rows = self.db.execute("SELECT * FROM lots WHERE qty > 1e-9 ORDER BY expiry").fetchall()
        return [dict(id=r["id"], name=r["name"], qty=r["qty"], etype=r["etype"],
                     expiry=date.fromisoformat(r["expiry"])) for r in rows]

    def consume(self, needs: dict, on_date: date):
        lots = self.lots()
        uses, missing = allocate(needs, lots, on_date, commit=True)
        for l in lots:
            self.db.execute("UPDATE lots SET qty=? WHERE id=?", (l["qty"], l["id"]))
        self.db.commit()
        return uses, missing

    def clear(self):
        self.db.execute("DELETE FROM lots")
        self.db.commit()
