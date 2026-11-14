"""PantryPilot agent: perceive -> plan -> decide/act (tools) -> critic -> (replan) -> respond -> learn.

Architecture: goal-based, learning agent with a generate-and-verify loop.
  Planner  : day-by-day greedy construction scored on expiry urgency, stock coverage, cost,
             learned preference (UCB bandit + tag weights) and variety.
  Critic   : independent ground-truth audit (allergen, diet, use-by safety, budget, duplicates);
             violations trigger targeted replanning (exclude offender / raise cost weight).
  Memory   : short-term (session context), episodic + preference (persistent, reward-driven).
"""
from __future__ import annotations

import copy
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta

from .critic import Critic
from .inventory import Inventory, allocate, urgency_by_name
from .knowledge import ALLERGENS, DIET_RANK, INGREDIENTS, RECIPE_BY_ID, price_of
from .llm import make_llm
from .memory import Memory
from .tools import ToolBox

PRIMARY_TOOL = {"plan_meals": "inventory_query", "add_item": "inventory_add",
                "show_expiring": "expiry_ranker", "shopping_list": "shopping_list_builder", "unknown": None}


@dataclass
class Weights:
    urgency: float = 1.2
    have: float = 0.6
    cost: float = 0.8
    pref: float = 0.7
    variety: float = 0.25
    relevance: float = 0.2


@dataclass
class Plan:
    meals: list
    ctx: dict
    total_cost: float = 0.0
    shopping: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    replans: int = 0
    discard: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return dict(budget=self.ctx.get("budget"), total_cost=self.total_cost, shopping=self.shopping,
                    violations=self.violations, replans=self.replans, discard=self.discard, notes=self.notes,
                    meals=[dict(day=i + 1, name=m["name"], recipe_id=m["recipe_id"], minutes=m["minutes"],
                                kcal=(m["nutrition"] or {}).get("kcal", "?"), missing_cost=m["missing_cost"],
                                rescues=m["rescues"]) for i, m in enumerate(self.meals)])


class PantryPilot:
    def __init__(self, inventory: Inventory | None = None, today: date | None = None, memory: Memory | None = None,
                 llm=None, use_memory: bool = True, use_critic: bool = True, max_replans: int = 5,
                 fail_rate: float = 0.0, seed: int = 0):
        self.inv = inventory or Inventory()
        self.today = today or date.today()
        self.memory = memory or Memory(enabled=use_memory)
        self.memory.enabled = use_memory
        self.llm = llm or make_llm()
        self.use_memory, self.use_critic, self.max_replans = use_memory, use_critic, max_replans
        self.tools = ToolBox(self.inv, self.today, fail_rate=fail_rate, seed=seed)
        self.critic = Critic()
        self.weights = Weights()
        self.history = deque(maxlen=12)       # short-term memory (recent turns)
        self.session = {}                     # short-term context (last n / budget / keywords)
        self.last_plan: Plan | None = None

    def set_today(self, d: date):
        self.today = d
        self.tools.today = d

    # =================================================================== PERCEIVE
    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^\x20-\x7E]", " ", text or "")[:500]

    def _validate(self, ent: dict) -> dict:
        out = {}
        if "n" in ent:
            out["n"] = max(1, min(7, int(ent["n"])))
        if "budget" in ent:
            out["budget"] = max(0.0, min(500.0, float(ent["budget"])))
        if ent.get("diet") in DIET_RANK:
            out["diet"] = ent["diet"]
        if ent.get("allergies"):
            out["allergies"] = sorted({a for a in ent["allergies"] if a in ALLERGENS})
        if "max_minutes" in ent:
            out["max_minutes"] = max(5, min(240, int(ent["max_minutes"])))
        if ent.get("name") in INGREDIENTS:
            out["name"] = ent["name"]
            out["qty"] = max(1.0, min(10000.0, float(ent.get("qty", 100))))
            out["days"] = max(-30, min(365, int(ent.get("days", 7))))
        return out

    def _perceive(self, n, budget, diet, allergies, max_minutes, keywords) -> dict:
        prof = self.memory.profile
        # SAFETY RULE: user text may ADD allergen constraints / tighten diet, never relax them.
        allergens = sorted(set(prof["allergens"]) | set(allergies or []))
        d = prof["diet"]
        if diet and DIET_RANK[diet] < DIET_RANK[d]:
            d = diet
        prof["allergens"], prof["diet"] = allergens, d
        return dict(n=n, budget=budget if budget is not None else prof.get("budget"), diet=d, allergens=allergens,
                    max_minutes=max_minutes or prof.get("max_minutes"), keywords=keywords or "")

    # =================================================================== PLAN / ACT
    def plan_meals(self, n: int = 3, budget: float | None = None, diet: str | None = None, allergies=None,
                   max_minutes: int | None = None, keywords: str = "") -> Plan:
        import time
        t0 = time.perf_counter()
        ctx = self._perceive(n, budget, diet, allergies, max_minutes, keywords)
        notes = []
        inv = self.tools.call("inventory_query")
        lots = inv.data if inv.ok else []
        if not inv.ok:
            notes.append("Inventory unavailable - planned as if the pantry were empty.")
        rank = self.tools.call("expiry_ranker")
        discard = [f"{r['name']} ({r['qty']:.0f})" for r in (rank.data or []) if r["status"] == "discard"]
        if rank.fallback:
            notes.append("Expiry ranking ran in degraded mode.")

        excluded: set = set()
        w = copy.copy(self.weights)
        attempts, viol = 0, []
        while True:
            meals, build_notes = self._build(ctx, lots, excluded, w)
            viol = self.critic.audit(meals, ctx) if self.use_critic else []
            if not viol or attempts >= self.max_replans:
                break
            for v in viol:                        # targeted adaptation, not blind retry
                if v["recipe_id"]:
                    excluded.add(v["recipe_id"])
                if v["type"] == "budget":
                    w.cost *= 2
                    if attempts >= 1 and meals:
                        excluded.add(max(meals, key=lambda m: m["missing_cost"])["recipe_id"])
            attempts += 1
        notes += build_notes
        sl = self.tools.call("shopping_list_builder", meals=meals)
        shopping = sl.data["items"] if sl.ok else {}
        total = round(sum(m["missing_cost"] for m in meals), 2)
        plan = Plan(meals, ctx, total, shopping, viol, attempts, discard, notes,
                    (time.perf_counter() - t0) * 1000)
        self.last_plan = plan
        self.session.update({k: ctx[k] for k in ("n", "budget") if ctx.get(k) is not None})
        return plan

    def _build(self, ctx, lots0, excluded, w):
        lots = [dict(l) for l in lots0]
        meals, chosen, notes = [], set(), []
        per_meal = ctx["budget"] / ctx["n"] if ctx["budget"] else None
        for d in range(ctx["n"]):
            day = self.today + timedelta(days=d)
            urg = urgency_by_name(lots, day)
            urgent = [k for k, v in sorted(urg.items(), key=lambda kv: -kv[1]) if v >= 0.3][:6]
            res = self.tools.call("recipe_retriever", query=" ".join(urgent + [ctx["keywords"]]), diet=ctx["diet"],
                                  declared_allergens=ctx["allergens"], exclude_ids=sorted(excluded | chosen),
                                  max_minutes=ctx["max_minutes"], k=14)
            best = None
            for r in (res.data or []):
                uses, missing = allocate(r["ingredients"], lots, day)
                n_ing = len(r["ingredients"])
                have = sum(min(1.0, sum(u["qty"] for u in uses if u["name"] == nm) / need)
                           for nm, need in r["ingredients"].items()) / n_ing
                use_sc = sum(urg.get(nm, 0) * min(1.0, sum(u["qty"] for u in uses if u["name"] == nm) / need)
                             for nm, need in r["ingredients"].items()) / n_ing
                cost_pen = min(2.0, price_of(missing) / (per_meal or 6.0))
                variety = w.variety * sum(1 for m in meals if m["tags"][0] == r["tags"][0])
                pref = self.memory.pref(r) if self.use_memory else 0.0
                score = (w.urgency * use_sc + w.have * have - w.cost * cost_pen + w.pref * pref
                         - variety + w.relevance * r["score"])
                if best is None or score > best[0]:
                    best = (score, r)
            if best is None:
                notes.append(f"No suitable recipe found for day {d + 1}.")
                continue
            r = best[1]
            uses, missing = allocate(r["ingredients"], lots, day, commit=True)
            cost = self.tools.call("cost_calculator", items=missing)
            nut = self.tools.call("nutrition_lookup", items=r["ingredients"])
            rescues = sorted({f"{u['name']} (exp. in {max(u['days_left'], 0)}d)" for u in uses if u["days_left"] <= 3 and u["etype"] is not None})
            if any(u["status"] == "quality_check" for u in uses):
                notes.append(f"Day {d + 1}: check {', '.join(sorted({u['name'] for u in uses if u['status'] == 'quality_check'}))} "
                             "for quality before use (past best-before).")
            meals.append(dict(date=day, recipe_id=r["id"], name=r["name"], minutes=r["minutes"], tags=r["tags"],
                              uses=uses, missing=missing, missing_cost=(cost.data or {}).get("total", 0.0),
                              nutrition=nut.data if nut.ok else None, rescues=rescues, score=round(best[0], 3)))
            chosen.add(r["id"])
        return meals, notes

    # =================================================================== LEARN
    def feedback(self, plan: Plan, outcomes: list, wasted: list | None = None) -> list:
        """outcomes[i] = {"cooked": bool, "rating": 1-5 | None}. Returns per-meal rewards."""
        rewards, ep_out = [], []
        for m, o in zip(plan.meals, outcomes):
            rating = o.get("rating")
            if not o["cooked"]:
                r = -1.0
            else:
                r = 1.0 + (0.5 if rating and rating >= 4 else -0.5 if rating and rating <= 2 else 0.0)
            rewards.append(r)
            self.memory.update(RECIPE_BY_ID[m["recipe_id"]], r)
            ep_out.append(dict(recipe_id=m["recipe_id"], name=m["name"], cooked=o["cooked"], rating=rating, reward=r))
        episode = dict(date=self.today.isoformat(), outcomes=ep_out, wasted=wasted or [], total_cost=plan.total_cost)
        self.memory.add_episode(episode, self.llm.reflect(episode))
        self.memory.save()
        return rewards

    # =================================================================== INTERFACE
    def handle(self, text: str) -> str:
        text = self._clean(text)
        self.history.append(("user", text))
        parsed = self.llm.parse_intent(text)
        intent = parsed.get("intent") if parsed.get("intent") in PRIMARY_TOOL else "unknown"
        ent = self._validate(parsed.get("entities", {}) or {})
        if intent == "add_item":
            if "name" not in ent:
                reply = "I could not recognise that ingredient."
            else:
                r = self.tools.call("inventory_add", name=ent["name"], qty=ent["qty"], days_to_expiry=ent["days"])
                reply = (f"Added {ent['qty']:.0f} of {ent['name'].replace('_', ' ')} (expires in {ent['days']} days)."
                         if r.ok else f"Could not add the item: {r.error}")
        elif intent == "show_expiring":
            r = self.tools.call("expiry_ranker")
            rows = [x for x in (r.data or []) if x["urgency"] > 0.3 or x["status"] == "discard"][:10]
            reply = "\n".join(f"  {x['name']:<14}{x['qty']:>6.0f}  {x['days_left']:>3}d  {x['status']}" for x in rows) or "Nothing urgent."
        elif intent == "shopping_list":
            if not self.last_plan:
                reply = "Ask me to plan some meals first, then I can build the shopping list."
            else:
                r = self.tools.call("shopping_list_builder", meals=self.last_plan.meals)
                reply = ("Shopping list:\n" + "\n".join(f"  {k.replace('_', ' ')}: {v:.0f}" for k, v in r.data["items"].items())
                         + f"\n  Estimated cost: {r.data['total']:.2f}") if r.ok else "Shopping-list tool failed."
        elif intent == "plan_meals":
            plan = self.plan_meals(n=ent.get("n", self.session.get("n", 3)), budget=ent.get("budget", self.session.get("budget")),
                                   diet=ent.get("diet"), allergies=ent.get("allergies"), max_minutes=ent.get("max_minutes"))
            reply = self.llm.explain(plan.to_dict())
        else:
            reply = "I can plan meals, add groceries, show what is expiring, or build a shopping list."
        self.history.append(("agent", reply))
        return reply
