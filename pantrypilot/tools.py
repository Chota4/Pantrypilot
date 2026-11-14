"""Tool layer: every external operation the agent can perform.

Each tool is called through ToolBox.call(), which provides
  * tracing  (tool, latency, ok, fallback_used)  -> observability / evaluation
  * retry    (one retry on failure)
  * fallback (simpler local implementation when a tool keeps failing)
  * optional fault injection to test robustness
"""
from __future__ import annotations

import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

from .inventory import Inventory, lot_status, lot_urgency
from .knowledge import (DIET_RANK, INGREDIENTS, RECIPE_BY_ID, RECIPES, nutrition_of, price_of)

TOOL_NAMES = ["inventory_query", "inventory_add", "expiry_ranker", "recipe_retriever",
              "cost_calculator", "nutrition_lookup", "shopping_list_builder"]


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    data: object = None
    error: str | None = None
    fallback: bool = False
    latency_ms: float = 0.0


def _tokens(text: str) -> list:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.findall(r"[a-z]+", text.lower())]


class TfIdfIndex:
    """Tiny dependency-free TF-IDF retriever over the recipe base (the 'RAG' component)."""

    def __init__(self, docs: dict):
        self.tf = {k: Counter(_tokens(v)) for k, v in docs.items()}
        n = len(docs)
        df = Counter(t for c in self.tf.values() for t in c)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.norm = {k: math.sqrt(sum((c[t] * self.idf[t]) ** 2 for t in c)) or 1.0 for k, c in self.tf.items()}

    def query(self, text: str) -> dict:
        q = Counter(_tokens(text))
        qn = math.sqrt(sum((q[t] * self.idf.get(t, 0)) ** 2 for t in q)) or 1.0
        out = {}
        for k, c in self.tf.items():
            dot = sum(q[t] * self.idf.get(t, 0) * c[t] * self.idf[t] for t in q if t in c)
            out[k] = dot / (qn * self.norm[k])
        return out


def _doc(r: dict) -> str:
    return " ".join([r["name"], *r["ingredients"].keys(), *r["tags"]]).replace("_", " ")


class ToolBox:
    def __init__(self, inventory: Inventory, today: date, fail_rate: float = 0.0, seed: int = 0):
        self.inv = inventory
        self.today = today
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)
        self.trace: list = []
        self.index = TfIdfIndex({r["id"]: _doc(r) for r in RECIPES})

    # ------------------------------------------------------------------ dispatch
    def call(self, tool: str, /, **kw) -> ToolResult:
        name = tool
        fn = getattr(self, "_t_" + name)
        t0 = time.perf_counter()
        ok, data, err, fb = False, None, None, False
        for _ in range(2):  # 1 retry
            try:
                if self.fail_rate and self.rng.random() < self.fail_rate:
                    raise ToolError(f"{name}: injected outage")
                data, ok = fn(**kw), True
                break
            except Exception as e:  # noqa: BLE001 - tool boundary: never crash the agent
                err = f"{type(e).__name__}: {e}"
        if not ok and hasattr(self, "_fb_" + name):
            try:
                data, ok, fb = getattr(self, "_fb_" + name)(**kw), True, True
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        self.trace.append(dict(tool=name, ok=ok, fallback=fb, ms=ms, error=None if ok else err))
        return ToolResult(ok, data, None if ok else err, fb, ms)

    # ------------------------------------------------------------------ tools
    def _t_inventory_query(self):
        return self.inv.lots()

    def _t_inventory_add(self, name: str, qty: float, days_to_expiry: int):
        return self.inv.add(name, qty, self.today + timedelta(days=int(days_to_expiry)))

    def _t_expiry_ranker(self, as_of: date | None = None):
        d = as_of or self.today
        rows = []
        for l in self.inv.lots():
            rows.append(dict(name=l["name"], qty=l["qty"], expiry=l["expiry"], etype=l["etype"],
                             days_left=(l["expiry"] - d).days, status=lot_status(l, d),
                             urgency=round(lot_urgency(l, d), 3)))
        return sorted(rows, key=lambda r: (-r["urgency"], r["days_left"]))

    def _fb_expiry_ranker(self, as_of: date | None = None):  # degraded: date sort only
        d = as_of or self.today
        return [dict(name=l["name"], qty=l["qty"], expiry=l["expiry"], etype=l["etype"],
                     days_left=(l["expiry"] - d).days, status=lot_status(l, d),
                     urgency=0.5 if l["expiry"] >= d else 0.0) for l in self.inv.lots()]

    def _filter(self, r, diet, allergens, exclude, max_minutes):
        if r["id"] in exclude:
            return False
        if DIET_RANK[r["diet"]] > DIET_RANK.get(diet or "omnivore", 2):
            return False
        if set(r["declared_allergens"]) & set(allergens or []):  # trusts DECLARED labels only
            return False
        return not (max_minutes and r["minutes"] > max_minutes)

    def _t_recipe_retriever(self, query: str, diet: str = "omnivore", declared_allergens=(),
                            exclude_ids=(), max_minutes: int | None = None, k: int = 12):
        scores = self.index.query(query or "dinner")
        pool = [r for r in RECIPES if self._filter(r, diet, declared_allergens, set(exclude_ids), max_minutes)]
        pool.sort(key=lambda r: -scores[r["id"]])
        return [dict(r, score=round(scores[r["id"]], 4)) for r in pool[:k]]

    def _fb_recipe_retriever(self, query: str, diet: str = "omnivore", declared_allergens=(),
                             exclude_ids=(), max_minutes: int | None = None, k: int = 12):
        q = set(_tokens(query))  # degraded: plain keyword overlap
        pool = [r for r in RECIPES if self._filter(r, diet, declared_allergens, set(exclude_ids), max_minutes)]
        pool.sort(key=lambda r: -len(q & set(_tokens(_doc(r)))))
        return [dict(r, score=0.0) for r in pool[:k]]

    def _t_cost_calculator(self, items: dict):
        return dict(total=price_of(items), breakdown={n: price_of({n: q}) for n, q in items.items()})

    def _fb_cost_calculator(self, items: dict):  # degraded: flat conservative estimate
        return dict(total=round(0.5 * len(items), 2), breakdown={}, estimated=True)

    def _t_nutrition_lookup(self, items: dict):
        return nutrition_of(items)

    def _t_shopping_list_builder(self, meals: list):
        agg: dict = {}
        for m in meals:
            for n, q in m["missing"].items():
                agg[n] = agg.get(n, 0) + q
        return dict(items=agg, total=price_of(agg))
