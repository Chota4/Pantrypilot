"""Critic / guardrail layer.  'Trust but verify': it re-checks every plan against ground-truth
ingredient data rather than trusting recipe labels or the planner."""
from __future__ import annotations

from .knowledge import DIET_RANK, INGREDIENTS, RECIPE_BY_ID


def true_allergens(recipe_id: str) -> set:
    return {a for i in RECIPE_BY_ID[recipe_id]["ingredients"] for a in INGREDIENTS[i]["allergens"]}


class Critic:
    def audit(self, meals: list, ctx: dict) -> list:
        v = []
        seen = set()
        for idx, m in enumerate(meals):
            rid = m["recipe_id"]
            hit = true_allergens(rid) & set(ctx["allergens"])
            if hit:
                v.append(dict(type="allergen", meal=idx, recipe_id=rid, detail=f"contains {sorted(hit)}"))
            if DIET_RANK[RECIPE_BY_ID[rid]["diet"]] > DIET_RANK[ctx["diet"]]:
                v.append(dict(type="diet", meal=idx, recipe_id=rid, detail=f"not {ctx['diet']}"))
            for u in m.get("uses", []):
                if u["etype"] == "use_by" and u["expiry"] < m["date"]:
                    v.append(dict(type="expiry", meal=idx, recipe_id=rid, detail=f"{u['name']} past use-by"))
            if rid in seen:
                v.append(dict(type="duplicate", meal=idx, recipe_id=rid, detail="repeated recipe"))
            seen.add(rid)
        if ctx.get("budget") is not None:
            total = round(sum(m["missing_cost"] for m in meals), 2)
            if total > ctx["budget"] + 1e-9:
                v.append(dict(type="budget", meal=None, recipe_id=None, detail=f"{total} > {ctx['budget']}"))
        return v
