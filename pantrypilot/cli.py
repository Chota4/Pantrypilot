"""Interactive CLI.   python -m pantrypilot.cli [--demo] [--memory memory.json]"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from .agent import PantryPilot
from .inventory import Inventory
from .llm import make_llm
from .memory import Memory

DEMO_PANTRY = [("chicken", 500, 2), ("spinach", 200, 1), ("tomato", 400, 3), ("egg", 6, 10), ("rice", 1000, 300),
               ("onion", 400, 12), ("coconut_milk", 400, 200), ("milk", 500, 1), ("cheese", 200, 15),
               ("peanuts", 200, 1), ("noodles", 500, 200), ("tofu", 300, 3)]


def seed_demo(agent):
    for n, q, d in DEMO_PANTRY:
        agent.inv.add(n, q, agent.today + timedelta(days=d))
    agent.inv.add("beef_mince", 300, agent.today - timedelta(days=1))   # past use-by: must be discarded


def rate(agent):
    p = agent.last_plan
    if not p or not p.meals:
        print("No plan to rate yet."); return
    outcomes = []
    for m in p.meals:
        a = input(f"  Did you cook '{m['name']}'? [y/n] ").strip().lower() == "y"
        r = None
        if a:
            try:
                r = int(input("    rating 1-5: "))
            except ValueError:
                r = None
        outcomes.append(dict(cooked=a, rating=r))
    print("  rewards:", agent.feedback(p, outcomes))
    print("  learned dislikes:", agent.memory.disliked_tags() or "none yet")


def demo(agent):
    scenarios = [("1) NORMAL", "plan 3 dinners"),
                 ("2) MULTI-STEP / TIGHT BUDGET", "plan 4 dinners under $3 budget"),
                 ("3) SAFETY: peanut allergy + 50% tool failures (peanuts are expiring!)", "plan 3 dinners, I have a peanut allergy")]
    for title, text in scenarios:
        print(f"\n=== {title}\n> {text}")
        if title.startswith("3"):
            agent.tools.fail_rate = 0.5
        print(agent.handle(text))
    agent.tools.fail_rate = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--memory", default=None, help="path of persistent memory JSON")
    args = ap.parse_args()
    agent = PantryPilot(Inventory(), date.today(), Memory(args.memory), make_llm())
    print(f"PantryPilot ready (LLM: {agent.llm.name}). Commands: /rate  /trace  /memory  /quit")
    if args.demo:
        seed_demo(agent); demo(agent); return
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if text in ("/quit", "/q"):
            break
        elif text == "/rate":
            rate(agent)
        elif text == "/trace":
            for t in agent.tools.trace[-15:]:
                print(f"  {t['tool']:<22} ok={t['ok']} fallback={t['fallback']} {t['ms']:.2f}ms")
        elif text == "/memory":
            print("profile:", agent.memory.profile); print("tag weights:", {k: round(v, 2) for k, v in agent.memory.tag_w.items()})
            print("reflections:", agent.memory.reflections[-3:])
        elif text:
            print(agent.handle(text))


if __name__ == "__main__":
    main()
