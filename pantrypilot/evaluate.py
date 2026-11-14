"""Systematic evaluation: ablations, learning curve, tool-selection accuracy, fault injection.
Run:  python -m pantrypilot.evaluate        (writes results/metrics.json + PNG charts)"""
from __future__ import annotations

import json
import random
import statistics as st
from collections import Counter
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agent import PRIMARY_TOOL, PantryPilot
from .inventory import Inventory
from .llm import RuleLLM
from .simulator import TODAY, AgentWrapper, NaiveBaseline, SimUser, run_session

OUT = Path(__file__).resolve().parent.parent / "results"
N_USERS, N_SESSIONS = 10, 20


def make_planner(cond: str, seed: int):
    if cond == "baseline":
        return NaiveBaseline(seed)
    kw = dict(full={}, no_memory=dict(use_memory=False), no_critic=dict(use_critic=False))[cond]
    return AgentWrapper(cond, **kw)


def ablation_experiment():
    conds = ["baseline", "no_critic", "no_memory", "full"]
    rows = {c: [] for c in conds}
    curves = {c: [[] for _ in range(N_SESSIONS)] for c in conds}
    tool_counts, lat = Counter(), []
    for c in conds:
        for u in range(N_USERS):
            user = SimUser(1000 + u)
            planner = make_planner(c, u)
            if hasattr(planner, "start"):
                planner.start(user)
            rng = random.Random(5000 + u)       # same pantries for every condition
            for s in range(N_SESSIONS):
                m = run_session(planner, user, rng)
                rows[c].append(m)
                curves[c][s].append(m["reward"])
            if hasattr(planner, "agent"):
                tool_counts.update(t["tool"] for t in planner.agent.tools.trace)
                lat += [t["ms"] for t in planner.agent.tools.trace]
    summary = {}
    for c in conds:
        r = rows[c]
        summary[c] = dict(
            waste_rate=st.mean(x["waste_rate"] for x in r), wasted_items=st.mean(x["wasted_items"] for x in r),
            plan_violation_rate=st.mean(x["plan_violation"] for x in r), over_budget_rate=st.mean(x["over_budget"] for x in r),
            acceptance=st.mean(x["acceptance"] for x in r), reward=st.mean(x["reward"] for x in r),
            avg_replans=st.mean(x["replans"] for x in r), sessions=len(r))
    curve_mean = {c: [st.mean(v) for v in curves[c]] for c in conds}
    return summary, curve_mean, tool_counts, lat


TOOL_TESTS = [  # (utterance, expected first tool)
    ("add 500g chicken expiring in 2 days", "inventory_add"), ("I bought 6 eggs", "inventory_add"),
    ("add 300g spinach in 1 days", "inventory_add"), ("got 2 kg potato", "inventory_add"),
    ("what is expiring soon?", "expiry_ranker"), ("what should I use up first", "expiry_ranker"),
    ("anything about to go bad?", "expiry_ranker"), ("plan 3 dinners", "inventory_query"),
    ("what can I cook tonight", "inventory_query"), ("plan 4 meals under $10 budget", "inventory_query"),
    ("make me a vegetarian menu for 2 days", "inventory_query"), ("show my shopping list", "shopping_list_builder"),
    ("what do I need to buy?", "shopping_list_builder"), ("tell me a joke", None), ("what's the weather", None),
]


def tool_selection_experiment():
    a = PantryPilot(today=TODAY, llm=RuleLLM())
    for n, q, d in [("chicken", 400, 2), ("spinach", 200, 1), ("tomato", 400, 3), ("rice", 500, 300), ("egg", 6, 10), ("onion", 300, 10)]:
        a.inv.add(n, q, TODAY + timedelta(days=d))
    a.plan_meals(n=3)
    ok, detail = 0, []
    for text, exp in TOOL_TESTS:
        a.tools.trace.clear()
        a.handle(text)
        got = a.tools.trace[0]["tool"] if a.tools.trace else None
        ok += got == exp
        detail.append(dict(text=text, expected=exp, got=got))
    return ok / len(TOOL_TESTS), detail


def fault_experiment():
    res = {}
    for fr in [0.0, 0.1, 0.3, 0.5, 1.0]:
        succ, fb, n, wr = 0, 0, 30, []
        for i in range(n):
            user = SimUser(2000 + i)
            w = AgentWrapper("fault", fail_rate=fr, seed=i)
            w.start(user)
            try:
                m = run_session(w, user, random.Random(9000 + i))
                succ += int(m["n_meals"] >= 1 and m["violations"] == 0)
                wr.append(m["waste_rate"])
            except Exception:  # noqa: BLE001
                pass
            fb += sum(t["fallback"] for t in w.agent.tools.trace)
        res[fr] = dict(success_rate=succ / n, fallbacks_per_session=fb / n, waste_rate=st.mean(wr) if wr else None)
    return res


def charts(summary, curve, tools, lat, fault):
    OUT.mkdir(exist_ok=True)
    conds = list(summary)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, key, title in zip(ax, ["waste_rate", "plan_violation_rate", "over_budget_rate"],
                             ["Food-waste rate (value)", "Plans with safety violation", "Plans over budget"]):
        a.bar(conds, [summary[c][key] for c in conds], color=["#9aa0a6", "#e8710a", "#f9ab00", "#188038"])
        a.set_title(title); a.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(OUT / "ablation_bars.png", dpi=150); plt.close()

    plt.figure(figsize=(6.5, 3.8))
    for c in conds:
        y = curve[c]; k = 3
        sm = [st.mean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))]
        plt.plot(range(1, len(y) + 1), sm, label=c, marker="o", ms=3)
    plt.xlabel("session"); plt.ylabel("mean meal reward (3-session avg)"); plt.title("Learning curve"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "learning_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6.5, 3.6))
    ks = sorted(tools, key=tools.get, reverse=True)
    plt.bar(ks, [tools[k] for k in ks], color="#1a73e8"); plt.xticks(rotation=30, ha="right")
    plt.title("Tool-usage frequency (all agent runs)"); plt.tight_layout(); plt.savefig(OUT / "tool_usage.png", dpi=150); plt.close()

    plt.figure(figsize=(5.5, 3.6))
    plt.boxplot(lat, vert=True, showfliers=False); plt.ylabel("ms per tool call"); plt.title("Tool latency")
    plt.tight_layout(); plt.savefig(OUT / "latency.png", dpi=150); plt.close()

    plt.figure(figsize=(5.5, 3.6))
    xs = list(fault)
    plt.plot(xs, [fault[x]["success_rate"] for x in xs], marker="o")
    plt.xlabel("injected tool-failure rate"); plt.ylabel("safe-plan success rate"); plt.ylim(0, 1.05)
    plt.title("Robustness to tool failures"); plt.tight_layout(); plt.savefig(OUT / "fault_tolerance.png", dpi=150); plt.close()


def main():
    OUT.mkdir(exist_ok=True)
    summary, curve, tools, lat = ablation_experiment()
    acc, detail = tool_selection_experiment()
    fault = fault_experiment()
    charts(summary, curve, tools, lat, fault)
    result = dict(ablation=summary, tool_selection_accuracy=acc, tool_selection_detail=detail,
                  fault_injection={str(k): v for k, v in fault.items()}, tool_calls=dict(tools),
                  latency_ms=dict(mean=st.mean(lat), p95=sorted(lat)[int(0.95 * len(lat))]))
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"{'condition':<11}{'waste':>7}{'viol%':>7}{'overbud%':>9}{'accept':>8}{'reward':>8}{'replans':>8}")
    for c, s in summary.items():
        print(f"{c:<11}{s['waste_rate']:>7.3f}{100*s['plan_violation_rate']:>7.1f}{100*s['over_budget_rate']:>9.1f}"
              f"{s['acceptance']:>8.3f}{s['reward']:>8.3f}{s['avg_replans']:>8.2f}")
    print(f"\nTool-selection accuracy: {acc:.2%}")
    print("Fault injection (failure rate -> safe-plan success / waste rate):")
    for k, v in fault.items():
        print(f"  {k:>4}: {v['success_rate']:.2f} / {v['waste_rate']:.3f}")
    print(f"Latency per tool call: mean {result['latency_ms']['mean']:.2f} ms, p95 {result['latency_ms']['p95']:.2f} ms")


if __name__ == "__main__":
    main()
