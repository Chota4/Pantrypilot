"""PantryPilot dashboard.   streamlit run dashboard.py"""
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pantrypilot.agent import PantryPilot
from pantrypilot.cli import seed_demo
from pantrypilot.inventory import Inventory
from pantrypilot.knowledge import ALLERGENS, CURRENCY, INGREDIENTS, RECIPE_BY_ID
from pantrypilot.llm import make_llm
from pantrypilot.memory import Memory

RESULTS = Path(__file__).parent / "results"
st.set_page_config(page_title="PantryPilot", page_icon="🥘", layout="wide")


def new_agent():
    return PantryPilot(Inventory(), date.today(), Memory(), make_llm())


if "agent" not in st.session_state:
    st.session_state.agent = new_agent()
agent: PantryPilot = st.session_state.agent

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🥘 PantryPilot")
    st.caption(f"LLM backend: **{agent.llm.name}**")
    if st.button("Load demo pantry", use_container_width=True):
        seed_demo(agent)
        st.rerun()
    if st.button("Reset everything", use_container_width=True):
        st.session_state.agent = new_agent()
        st.rerun()
    st.divider()
    agent.tools.fail_rate = st.slider("Inject tool failures", 0.0, 1.0, agent.tools.fail_rate, 0.1,
                                      help="Robustness demo: each tool call fails with this probability.")
    agent.use_critic = st.toggle("Safety critic", agent.use_critic)
    agent.use_memory = agent.memory.enabled = st.toggle("Learning memory", agent.use_memory)
    st.caption("Safety profile (can only be tightened):")
    st.json(agent.memory.profile, expanded=False)
    if st.button("Clear allergens / diet", use_container_width=True):
        agent.memory.profile.update(allergens=[], diet="omnivore")
        st.rerun()

tab_pantry, tab_plan, tab_fb, tab_mem, tab_eval, tab_trace = st.tabs(
    ["Pantry", "Plan", "Feedback", "Memory", "Evaluation", "Tool trace"])

# ------------------------------------------------------------------ pantry
with tab_pantry:
    rows = agent.tools.call("expiry_ranker").data or []
    c1, c2, c3 = st.columns(3)
    c1.metric("Items", len(rows))
    c2.metric("Use soon (≤2 days)", sum(1 for r in rows if r["status"] != "discard" and r["days_left"] <= 2))
    c3.metric("Discard (unsafe)", sum(1 for r in rows if r["status"] == "discard"))
    if rows:
        df = pd.DataFrame([dict(item=r["name"], qty=round(r["qty"]), expires=str(r["expiry"]), days_left=r["days_left"],
                                type=r["etype"], status=r["status"], urgency=r["urgency"]) for r in rows])
        color = {"discard": "background-color:#fce8e6", "quality_check": "background-color:#fef7e0"}
        st.dataframe(df.style.apply(lambda r: [color.get(r["status"], "")] * len(r), axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("Pantry is empty. Add items below or load the demo pantry from the sidebar.")
    with st.form("add_item", clear_on_submit=True):
        a, b, c, d = st.columns([2, 1, 1, 1])
        name = a.selectbox("Ingredient", sorted(INGREDIENTS))
        qty = b.number_input("Quantity (g/ml/count)", 1, 10000, 200)
        days = c.number_input("Expires in (days)", -10, 365, 5)
        d.write("")
        if d.form_submit_button("Add"):
            r = agent.tools.call("inventory_add", name=name, qty=float(qty), days_to_expiry=int(days))
            st.success("Added") if r.ok else st.error(r.error)
            st.rerun()
    st.caption("Red = past use-by or too old (never used). Yellow = past best-before, quality check needed.")

# ------------------------------------------------------------------ plan
with tab_plan:
    with st.form("plan_form"):
        a, b, c, d = st.columns(4)
        n = a.slider("Dinners", 1, 7, 3)
        budget = b.number_input(f"Budget ({CURRENCY}, 0 = none)", 0.0, 500.0, 0.0, 1.0)
        diet = c.selectbox("Diet", ["omnivore", "vegetarian", "vegan"])
        allergies = d.multiselect("Allergies", sorted(ALLERGENS))
        go = st.form_submit_button("Plan meals", type="primary")
    if go:
        agent.plan_meals(n=n, budget=budget or None, diet=None if diet == "omnivore" else diet, allergies=allergies)
    plan = agent.last_plan
    if plan:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Extra cost", f"{CURRENCY}{plan.total_cost:.2f}", None if plan.ctx["budget"] is None else f"budget {plan.ctx['budget']:.0f}")
        c2.metric("Critic replans", plan.replans)
        c3.metric("Unresolved violations", len(plan.violations))
        c4.metric("Planning time", f"{plan.latency_ms:.1f} ms")
        for note in plan.notes:
            st.warning(note)
        for i, m in enumerate(plan.meals, 1):
            with st.container(border=True):
                l, r = st.columns([3, 2])
                l.subheader(f"Day {i} · {m['name']}")
                l.caption(f"{m['date']} · {m['minutes']} min · tags: {', '.join(m['tags'])}")
                if m["rescues"]:
                    l.success("Rescues: " + ", ".join(m["rescues"]))
                nut = m["nutrition"] or {}
                r.write(f"**{nut.get('kcal', '?')} kcal · {nut.get('protein_g', '?')} g protein**")
                r.write(f"To buy: {CURRENCY}{m['missing_cost']:.2f}")
                r.caption("Missing: " + (", ".join(f"{k} {v:.0f}" for k, v in m["missing"].items()) or "nothing"))
        left, right = st.columns(2)
        with left:
            st.markdown("**Shopping list**")
            if plan.shopping:
                st.dataframe(pd.DataFrame([dict(item=k, qty=round(v)) for k, v in plan.shopping.items()]), hide_index=True, use_container_width=True)
            else:
                st.caption("Nothing to buy.")
        with right:
            st.markdown("**Discarded for safety**")
            st.write(", ".join(plan.discard) or "none")
            for v in plan.violations:
                st.error(f"{v['type']}: {v['detail']}")
        with st.expander("Explanation (LLM)"):
            st.text(agent.llm.explain(plan.to_dict()))

# ------------------------------------------------------------------ feedback
with tab_fb:
    plan = agent.last_plan
    if not plan or not plan.meals:
        st.info("Create a plan first.")
    else:
        st.write("Tell the agent what actually happened. This becomes the reward signal.")
        with st.form("fb"):
            outs = []
            for i, m in enumerate(plan.meals):
                a, b = st.columns([2, 2])
                cooked = a.checkbox(f"Cooked: {m['name']}", key=f"c{i}")
                rating = b.slider("Rating", 1, 5, 4, key=f"r{i}")
                outs.append(dict(cooked=cooked, rating=rating if cooked else None))
            if st.form_submit_button("Submit feedback", type="primary"):
                rewards = agent.feedback(plan, outs)
                st.success("Learned. Rewards: " + ", ".join(f"{r:+.1f}" for r in rewards))

# ------------------------------------------------------------------ memory
with tab_mem:
    mem = agent.memory
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Learned tag preferences** (reward EMA)")
        if mem.tag_w:
            st.bar_chart(pd.Series(mem.tag_w).sort_values())
        else:
            st.caption("No feedback yet.")
        st.markdown("**Recipe statistics**")
        if mem.recipe_stats:
            st.dataframe(pd.DataFrame([dict(recipe=RECIPE_BY_ID[k]["name"], times=v["n"], mean_reward=round(v["sum"] / v["n"], 2))
                                       for k, v in mem.recipe_stats.items()]).sort_values("mean_reward"), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Reflections (episodic memory)**")
        for r in reversed(mem.reflections[-8:]):
            st.write("• " + r)
        st.markdown("**Short-term context**")
        st.json(agent.session)
    st.divider()
    st.markdown("**Chat with the agent**")
    msg = st.text_input("e.g. add 300g spinach in 1 days · what is expiring? · plan 3 dinners under $8 budget")
    if msg:
        st.code(agent.handle(msg))

# ------------------------------------------------------------------ evaluation
with tab_eval:
    f = RESULTS / "metrics.json"
    if not f.exists():
        st.info("Run `python -m pantrypilot.evaluate` first.")
    else:
        res = json.loads(f.read_text())
        ab = pd.DataFrame(res["ablation"]).T
        st.markdown("**Ablation study** (same pantries and users for every condition)")
        st.dataframe(ab.style.format("{:.3f}"), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tool-selection accuracy", f"{res['tool_selection_accuracy']:.0%}")
        c2.metric("Mean tool latency", f"{res['latency_ms']['mean']:.2f} ms")
        c3.metric("Test utterances", len(res["tool_selection_detail"]))
        st.markdown("**Fault injection**")
        st.dataframe(pd.DataFrame(res["fault_injection"]).T, use_container_width=True)
        imgs = ["ablation_bars", "learning_curve", "tool_usage", "latency", "fault_tolerance"]
        cols = st.columns(2)
        for i, name in enumerate(imgs):
            if (RESULTS / f"{name}.png").exists():
                cols[i % 2].image(str(RESULTS / f"{name}.png"), caption=name.replace("_", " "))

# ------------------------------------------------------------------ trace
with tab_trace:
    tr = agent.tools.trace
    if not tr:
        st.info("No tool calls yet.")
    else:
        df = pd.DataFrame(tr)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tool calls", len(df))
        c2.metric("Failures", int((~df.ok).sum()))
        c3.metric("Fallbacks used", int(df.fallback.sum()))
        st.bar_chart(df.tool.value_counts())
        st.dataframe(df.tail(60).iloc[::-1], use_container_width=True, hide_index=True)
