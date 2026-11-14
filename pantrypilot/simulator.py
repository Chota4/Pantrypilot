"""Simulation harness: synthetic pantries + a simulated user with HIDDEN tag preferences.
Used for the learning curve, ablations and fault-injection experiments. All planners are scored
by the same ground-truth evaluator, so comparisons are fair."""
from __future__ import annotations

import random
import time
from datetime import date, timedelta

from .agent import PantryPilot, Plan
from .critic import true_allergens
from .inventory import Inventory, allocate
from .knowledge import ALL_TAGS, DIET_RANK, INGREDIENTS, RECIPE_BY_ID, RECIPES, price_of
from .llm import RuleLLM
from .memory import Memory

TODAY = date(2026, 1, 5)
HORIZON = 7  # days over which waste is judged


class SimUser:
    def __init__(self, seed: int):
        r = random.Random(seed)
        tags = r.sample(ALL_TAGS, 4)
        self.liked, self.disliked = set(tags[:2]), set(tags[2:])
        self.allergens = r.choice([[], [], ["peanut"], ["dairy"], ["gluten"], ["egg"], ["shellfish"], ["soy"]])
        self.diet = r.choice(["omnivore"] * 3 + ["vegetarian", "vegan"])
        self.rng = random.Random(seed + 1)

    def decide(self, recipe: dict):
        t = set(recipe["tags"])
        p = min(0.95, max(0.05, 0.55 + 0.3 * len(t & self.liked) - 0.35 * len(t & self.disliked)))
        cooked = self.rng.random() < p
        rating = None if not cooked else (5 if t & self.liked else 2 if t & self.disliked else 4)
        return cooked, rating


def make_pantry(rng: random.Random):
    names = rng.sample(sorted(INGREDIENTS), rng.randint(10, 15))
    items = []
    for n in names:
        info = INGREDIENTS[n]
        qty = rng.choice([1, 2, 3, 6]) if info["base"] == 1 else rng.choice([100, 200, 300, 500])
        days = rng.randint(1, min(info["shelf"], 10)) if info["shelf"] <= 10 else rng.choice([2, 4, 6, 9, 20, 60])
        if info["etype"] == "use_by" and rng.random() < 0.10:
            days = -1  # already past use-by: must never be used
        items.append((n, qty, days))
    return items


class NaiveBaseline:
    """Stateless 'LLM-only' stand-in: sees recipe titles/labels only. Honours diet and DECLARED
    allergen labels, but has no inventory, expiry, budget, critic or memory."""
    name = "baseline"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def plan(self, inv, user, n, budget, today):
        pool = [r for r in RECIPES if DIET_RANK[r["diet"]] <= DIET_RANK[user.diet]
                and not set(r["declared_allergens"]) & set(user.allergens)]
        meals = [dict(date=today + timedelta(days=i), recipe_id=r["id"], name=r["name"], tags=r["tags"], uses=[])
                 for i, r in enumerate(self.rng.sample(pool, min(n, len(pool))))]
        return meals, {}

    def learn(self, *a, **k):
        pass


class AgentWrapper:
    def __init__(self, name, **kw):
        self.name, self.kw, self.agent, self.latencies = name, kw, None, []

    def start(self, user):
        mem = Memory(enabled=self.kw.get("use_memory", True))
        mem.profile.update(allergens=list(user.allergens), diet=user.diet)
        self.agent = PantryPilot(llm=RuleLLM(), memory=mem, today=TODAY, **self.kw)

    def plan(self, inv, user, n, budget, today):
        a = self.agent
        a.inv.clear()
        for name, qty, days in inv:
            a.inv.add(name, qty, today + timedelta(days=days))
        a.set_today(today)
        p = a.plan_meals(n=n, budget=budget)
        self.latencies.append(p.latency_ms)
        self._plan = p
        return p.meals, dict(replans=p.replans, plan=p)

    def learn(self, meals, outcomes, wasted, extra):
        self.agent.feedback(extra["plan"], outcomes, wasted)


def run_session(planner, user, rng, n=4, budget=None):
    """Run one planning session and score it with ground truth. Returns a metrics dict."""
    pantry = make_pantry(rng)
    budget = budget or rng.choice([6, 8, 10, 12])
    inv = Inventory()
    for name, qty, days in pantry:
        inv.add(name, qty, TODAY + timedelta(days=days))
    initial = {l["id"]: dict(l) for l in inv.lots()}
    meals, extra = planner.plan(pantry, user, n, budget, TODAY)

    # ---- ground-truth plan checks (identical for every planner)
    sim_lots = [dict(l) for l in inv.lots()]
    planned_spend, violations = 0.0, 0
    for m in meals:
        r = RECIPE_BY_ID[m["recipe_id"]]
        _, missing = allocate(r["ingredients"], sim_lots, m["date"], commit=True)
        planned_spend += price_of(missing)
        if true_allergens(r["id"]) & set(user.allergens) or DIET_RANK[r["diet"]] > DIET_RANK[user.diet]:
            violations += 1
    over_budget = planned_spend > budget + 1e-9

    # ---- the simulated user cooks (consumes REAL stock; missing items assumed bought)
    outcomes = []
    for m in meals:
        r = RECIPE_BY_ID[m["recipe_id"]]
        cooked, rating = user.decide(r)
        if cooked:
            inv.consume(r["ingredients"], m["date"])
        outcomes.append(dict(cooked=cooked, rating=rating))

    # ---- waste: value of expiring-in-horizon stock that was never used
    left = {l["id"]: l["qty"] for l in inv.lots()}
    tot_value, wasted_value, wasted_items, wasted_names = 0.0, 0.0, 0, []
    for lid, l in initial.items():
        d = (l["expiry"] - TODAY).days
        if 0 <= d <= HORIZON - 1:
            val = price_of({l["name"]: l["qty"]})
            rem = left.get(lid, 0.0)
            tot_value += val
            wasted_value += val * rem / l["qty"]
            if rem / l["qty"] > 0.5:
                wasted_items += 1
                wasted_names.append(l["name"])
    planner.learn(meals, outcomes, wasted_names, extra)

    rewards = [(-1.0 if not o["cooked"] else 1.0 + (0.5 if o["rating"] == 5 else -0.5 if o["rating"] == 2 else 0.0))
               for o in outcomes]
    return dict(waste_rate=(wasted_value / tot_value) if tot_value else 0.0, wasted_items=wasted_items,
                violations=violations, plan_violation=int(violations > 0), over_budget=int(over_budget),
                spend=planned_spend, acceptance=sum(o["cooked"] for o in outcomes) / max(1, len(outcomes)),
                reward=sum(rewards) / max(1, len(rewards)), n_meals=len(meals), replans=extra.get("replans", 0))
