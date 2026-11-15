"""Static domain knowledge: ingredient facts and a small recipe base.

Ingredient quantities are expressed in the ingredient's own base unit
(grams, millilitres, or a count for eggs). `price`, `kcal` and `protein`
are per `base` units.
"""
from __future__ import annotations

ALLERGENS = {"peanut", "dairy", "egg", "gluten", "soy", "shellfish", "fish"}
DIET_RANK = {"vegan": 0, "vegetarian": 1, "omnivore": 2}
CURRENCY = "$"


def _i(price, base, kcal, protein, allergens=(), diet="vegan", shelf=7, etype="best_before"):
    return dict(price=price, base=base, kcal=kcal, protein=protein,
                allergens=tuple(allergens), diet=diet, shelf=shelf, etype=etype)


INGREDIENTS = {
    # fresh produce (best-before: quality, not safety)
    "tomato": _i(0.30, 100, 18, 0.9, shelf=6),
    "onion": _i(0.15, 100, 40, 1.1, shelf=20),
    "garlic": _i(0.60, 100, 149, 6.4, shelf=30),
    "spinach": _i(0.50, 100, 23, 2.9, shelf=4),
    "carrot": _i(0.15, 100, 41, 0.9, shelf=15),
    "potato": _i(0.12, 100, 77, 2.0, shelf=25),
    "bell_pepper": _i(0.50, 100, 31, 1.0, shelf=8),
    "broccoli": _i(0.40, 100, 34, 2.8, shelf=6),
    "cabbage": _i(0.12, 100, 25, 1.3, shelf=14),
    "mushroom": _i(0.60, 100, 22, 3.1, shelf=5),
    "lemon": _i(0.30, 100, 29, 1.1, shelf=14),
    "chili": _i(0.80, 100, 40, 1.9, shelf=8),
    # chilled protein: USE-BY => safety critical
    "chicken": _i(0.80, 100, 165, 31, diet="omnivore", shelf=4, etype="use_by"),
    "beef_mince": _i(1.20, 100, 250, 26, diet="omnivore", shelf=3, etype="use_by"),
    "fish": _i(1.50, 100, 130, 22, ("fish",), "omnivore", 2, "use_by"),
    "shrimp": _i(1.80, 100, 99, 24, ("shellfish",), "omnivore", 2, "use_by"),
    "tofu": _i(0.45, 100, 76, 8, ("soy",), "vegan", 7, "use_by"),
    # dairy / eggs
    "egg": _i(0.25, 1, 72, 6.3, ("egg",), "vegetarian", 21),
    "milk": _i(0.10, 100, 42, 3.4, ("dairy",), "vegetarian", 7, "use_by"),
    "cheese": _i(0.90, 100, 402, 25, ("dairy",), "vegetarian", 21),
    "yogurt": _i(0.30, 100, 61, 3.5, ("dairy",), "vegetarian", 10, "use_by"),
    "butter": _i(0.80, 100, 717, 0.9, ("dairy",), "vegetarian", 30),
    # grains / bakery
    "rice": _i(0.10, 100, 360, 7, shelf=365),
    "pasta": _i(0.20, 100, 371, 13, ("gluten",), shelf=365),
    "noodles": _i(0.25, 100, 360, 12, ("gluten",), shelf=365),
    "bread": _i(0.25, 100, 265, 9, ("gluten",), shelf=5),
    # pantry
    "lentils": _i(0.25, 100, 353, 25, shelf=365),
    "chickpeas": _i(0.25, 100, 364, 19, shelf=365),
    "soy_sauce": _i(0.40, 100, 53, 8, ("soy", "gluten"), shelf=365),
    "peanut_butter": _i(0.60, 100, 588, 25, ("peanut",), shelf=180),
    "peanuts": _i(0.60, 100, 567, 26, ("peanut",), shelf=180),
    "coconut_milk": _i(0.35, 100, 197, 2, shelf=365),
    "olive_oil": _i(1.00, 100, 884, 0, shelf=365),
    "fish_sauce": _i(0.50, 100, 35, 5, ("fish",), "omnivore", 365),
}

# (id, name, ingredients, minutes, tags, DECLARED allergens)
# NOTE: declared labels are deliberately incomplete for some recipes (e.g. soy sauce
# contains gluten; pad thai contains peanuts). This mimics messy real-world metadata and
# is why the Critic verifies allergens at ingredient level instead of trusting labels.
_R = [
    ("tomato_egg_stirfry", "Tomato & Egg Stir-fry", {"egg": 3, "tomato": 200, "onion": 50, "rice": 100}, 20, ["stirfry", "egg", "quick"], ["egg"]),
    ("veg_fried_rice", "Vegetable Fried Rice", {"rice": 120, "egg": 2, "carrot": 80, "onion": 50, "soy_sauce": 20, "cabbage": 80}, 25, ["rice", "stirfry", "comfort"], ["egg", "soy"]),
    ("spinach_omelette", "Spinach & Cheese Omelette", {"egg": 3, "spinach": 100, "cheese": 40, "butter": 10}, 15, ["egg", "quick", "healthy"], ["egg", "dairy"]),
    ("chicken_curry", "Chicken Coconut Curry", {"chicken": 250, "onion": 80, "tomato": 150, "coconut_milk": 100, "rice": 100, "garlic": 10}, 40, ["curry", "spicy", "comfort"], []),
    ("lentil_soup", "Hearty Lentil Soup", {"lentils": 100, "carrot": 100, "onion": 60, "tomato": 100, "garlic": 10}, 35, ["soup", "healthy", "veggie"], []),
    ("mushroom_pasta", "Creamy Mushroom Pasta", {"pasta": 120, "mushroom": 150, "garlic": 10, "cheese": 40, "milk": 50}, 25, ["pasta", "comfort", "veggie"], ["gluten", "dairy"]),
    ("tofu_stirfry", "Tofu & Broccoli Stir-fry", {"tofu": 250, "broccoli": 150, "bell_pepper": 100, "soy_sauce": 20, "rice": 100}, 25, ["stirfry", "healthy", "veggie"], ["soy"]),
    ("baked_fish_potato", "Lemon Baked Fish & Potato", {"fish": 250, "potato": 300, "lemon": 30, "butter": 15}, 35, ["fish", "healthy", "comfort"], ["fish", "dairy"]),
    ("chickpea_curry", "Chickpea Spinach Curry", {"chickpeas": 100, "tomato": 200, "onion": 80, "coconut_milk": 100, "spinach": 80, "rice": 100}, 35, ["curry", "veggie", "spicy"], []),
    ("beef_chili", "Beef & Lentil Chili", {"beef_mince": 250, "tomato": 250, "onion": 80, "chili": 10, "lentils": 50, "rice": 100}, 45, ["spicy", "comfort", "soup"], []),
    ("shrimp_noodles", "Garlic Shrimp Noodles", {"shrimp": 200, "noodles": 120, "bell_pepper": 80, "soy_sauce": 20, "garlic": 10, "fish_sauce": 10}, 20, ["noodles", "quick", "stirfry"], ["shellfish", "gluten"]),
    ("pad_thai_style", "Pad Thai-style Noodles", {"noodles": 120, "egg": 2, "tofu": 100, "peanuts": 40, "cabbage": 80, "fish_sauce": 15}, 30, ["noodles", "egg", "stirfry"], ["egg", "gluten"]),
    ("potato_cheese_bake", "Cheesy Potato Bake", {"potato": 400, "cheese": 80, "milk": 100, "onion": 50}, 50, ["comfort", "veggie"], ["dairy"]),
    ("veggie_stew", "Root Vegetable Stew", {"potato": 250, "carrot": 150, "cabbage": 100, "onion": 60, "tomato": 100, "lentils": 50}, 45, ["soup", "veggie", "healthy"], []),
    ("chicken_veg_stirfry", "Chicken & Veg Stir-fry", {"chicken": 250, "broccoli": 150, "bell_pepper": 100, "soy_sauce": 20, "garlic": 10, "rice": 100}, 25, ["stirfry", "quick", "healthy"], ["soy"]),
    ("chicken_yogurt_wrap", "Chicken Yogurt Flatbread", {"chicken": 200, "yogurt": 100, "bread": 100, "cabbage": 60, "tomato": 80}, 20, ["quick", "comfort"], ["dairy", "gluten"]),
    ("pasta_pomodoro", "Pasta Pomodoro", {"pasta": 120, "tomato": 300, "garlic": 10, "onion": 50}, 20, ["pasta", "quick", "veggie"], ["gluten"]),
    ("peanut_noodles", "Spicy Peanut Noodles", {"noodles": 120, "peanut_butter": 40, "soy_sauce": 20, "carrot": 60, "cabbage": 60, "chili": 5}, 20, ["noodles", "spicy", "quick"], ["peanut", "gluten"]),
    ("broccoli_cheese_soup", "Broccoli Cheddar Soup", {"broccoli": 250, "cheese": 60, "milk": 200, "potato": 150, "onion": 50}, 35, ["soup", "comfort"], ["dairy"]),
    ("chickpea_salad", "Lemony Chickpea Salad", {"chickpeas": 100, "tomato": 150, "onion": 40, "lemon": 30, "olive_oil": 15, "bread": 60}, 15, ["healthy", "quick", "veggie"], ["gluten"]),
    ("mushroom_rice_bowl", "Garlic Mushroom Rice Bowl", {"rice": 100, "mushroom": 200, "garlic": 10, "soy_sauce": 15, "spinach": 60}, 25, ["rice", "veggie", "healthy"], ["soy"]),
    ("fish_curry", "Coconut Fish Curry", {"fish": 250, "coconut_milk": 150, "tomato": 150, "onion": 60, "chili": 10, "rice": 100}, 40, ["curry", "fish", "spicy"], ["fish"]),
    ("scrambled_tofu_toast", "Scrambled Tofu on Toast", {"tofu": 250, "spinach": 80, "tomato": 100, "onion": 50, "bread": 80}, 20, ["quick", "veggie", "healthy"], ["soy", "gluten"]),
    ("beef_potato_stew", "Beef & Potato Stew", {"beef_mince": 200, "potato": 300, "carrot": 100, "onion": 60, "tomato": 80}, 50, ["soup", "comfort"], []),
]


def _build_recipes():
    out = []
    for rid, name, ings, minutes, tags, declared in _R:
        true_all = sorted({a for i in ings for a in INGREDIENTS[i]["allergens"]})
        diet = max((INGREDIENTS[i]["diet"] for i in ings), key=lambda d: DIET_RANK[d])
        out.append(dict(id=rid, name=name, ingredients=ings, minutes=minutes, tags=tags,
                        declared_allergens=declared, true_allergens=true_all, diet=diet))
    return out


RECIPES = _build_recipes()
RECIPE_BY_ID = {r["id"]: r for r in RECIPES}
ALL_TAGS = sorted({t for r in RECIPES for t in r["tags"]})


def price_of(needs: dict) -> float:
    return round(sum(INGREDIENTS[n]["price"] * q / INGREDIENTS[n]["base"] for n, q in needs.items()), 2)


def nutrition_of(needs: dict) -> dict:
    kcal = sum(INGREDIENTS[n]["kcal"] * q / INGREDIENTS[n]["base"] for n, q in needs.items())
    prot = sum(INGREDIENTS[n]["protein"] * q / INGREDIENTS[n]["base"] for n, q in needs.items())
    return {"kcal": round(kcal), "protein_g": round(prot, 1)}
