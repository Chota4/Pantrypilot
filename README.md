# PantryPilot — expiry-aware, budget-aware, self-improving meal-planning agent
DAT404 Assignment D. Plans multi-day dinners from what is in your pantry, rescues food about to expire,
respects allergens/diet/budget, verifies its own plans, and learns from what you actually cook.

## Quick start
```bash
pip install -r requirements.txt          # matplotlib + pytest (anthropic is optional)
python -m pantrypilot.cli --demo         # scripted demo: normal / tight budget / allergy + tool failures
python -m pantrypilot.cli                # interactive (/rate /trace /memory /quit)
streamlit run dashboard.py               # web dashboard (see below)
pytest -q                                # 15 tests (safety, robustness, learning, dashboard smoke test)
python -m pantrypilot.evaluate           # experiments -> results/metrics.json + PNG charts
```
No API key needed: a deterministic rule-based `RuleLLM` is used offline (reproducible experiments).
Set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) and Claude is used for intent parsing,
explanations and reflections. LLM output is validated and treated as data; any failure falls back to rules.

Try: `add 500g chicken expiring in 2 days` · `what is expiring?` · `plan 4 dinners under $10 budget, I'm allergic to peanuts` · `shopping list`

## Dashboard (Streamlit)
`streamlit run dashboard.py` opens six tabs: **Pantry** (colour-coded expiry table, add items), **Plan** (dinners, budget, diet, allergies -> meal cards, shopping list, critic result),
**Feedback** (cooked/rating -> reward), **Memory** (learned tag preferences, reflections, chat box), **Evaluation** (ablation table and charts from `results/`),
**Tool trace** (calls, failures, fallbacks). The sidebar toggles the critic and memory and injects tool failures live, which makes ablations easy to show in the demo.

## Architecture (goal-based learning agent, generate-and-verify loop)
```
User -> PERCEIVE (intent+entities, validated; profile merge: text can only TIGHTEN safety)
     -> PLAN (day-by-day construction)  <-- Memory: bandit/tag preferences, short-term context
          tools: inventory_query, expiry_ranker, recipe_retriever (TF-IDF RAG), cost_calculator,
                 nutrition_lookup, shopping_list_builder, inventory_add
     -> CRITIC (ground-truth audit: allergen, diet, use-by safety, budget, duplicates)
          violation -> targeted REPLAN (exclude offender / double cost weight), max 5
     -> RESPOND (LLM explanation grounded in the plan JSON)
     -> FEEDBACK (cooked/skipped + rating -> reward) -> REFLECT -> update memory
```
| File | Role |
|---|---|
| `agent.py` | agent loop, planner scoring, replanning, feedback |
| `tools.py` | tool layer with tracing, retry, fallbacks, fault injection; TF-IDF retriever |
| `critic.py` | independent safety/constraint verifier |
| `memory.py` | profile, episodic memory, UCB recipe bandit, tag preferences, JSON persistence |
| `llm.py` | `RuleLLM` (offline) and `AnthropicLLM`, same interface |
| `inventory.py` | SQLite lots, use-by vs best-before rules, urgency, FIFO allocation |
| `simulator.py` / `evaluate.py` | synthetic pantries, simulated user with hidden tastes, experiments, charts |

Key design decisions (for the defence)
* **Use-by vs best-before**: use-by past expiry is *never* used (hard safety rule); best-before gets a 3-day grace with a quality-check warning.
* **Trust but verify**: recipe allergen labels are deliberately incomplete (e.g. soy sauce -> gluten, pad thai -> peanut). Retrieval filters on labels; the critic checks real ingredients.
* **Safety cannot be relaxed by text**: user messages can add allergens or tighten diet, never remove them.
* **Learning**: reward = +1 cooked, +0.5 rated >=4, -0.5 rated <=2, -1 skipped. Score uses `0.6*recipe mean + 0.4*tag mean + UCB bonus`.
* **Failure handling**: each tool retries once, then a degraded fallback; if inventory is unreachable the agent plans from an empty pantry and says so.

## Results (10 simulated users x 20 sessions per condition, identical pantries, `results/`)
| Condition | Waste rate | Plans w/ safety violation | Over budget | Acceptance | Mean reward |
|---|---|---|---|---|---|
| baseline (no tools/memory/critic) | 0.911 | 21.0% | 21.5% | 0.603 | 0.351 |
| no_critic | 0.797 | 14.0% | 0.0% | 0.706 | 0.651 |
| no_memory | 0.791 | 0.0% | 0.0% | 0.632 | 0.450 |
| **full** | 0.795 | **0.0%** | **0.0%** | **0.718** | **0.676** |

Tool-selection accuracy 15/15 (100%). Under injected tool failures (0 -> 100%), safe-plan rate stayed 100% but
waste rose 0.894 -> 0.966 (graceful degradation). Charts: `ablation_bars.png`, `learning_curve.png`, `tool_usage.png`, `latency.png`, `fault_tolerance.png`.

## Honest limitations (put these in the report)
* Waste improvement is modest (0.91 -> 0.80): 4 meals physically cannot consume ~12 pantry items. Raising the urgency weight to 6 only reached 0.77 and cost reward.
* Memory improves acceptance (0.63 -> 0.72) mainly via tag preferences; cold-start sessions still explore.
* Results come from a simulator with synthetic users, 24 recipes and made-up prices; no real user study.
* The offline LLM is regex-based; real NLU quality/latency is only exercised with the Anthropic backend (not benchmarked here). Local tool latency (~0.04 ms) is negligible; an LLM call would dominate.
* Quantities are approximate; portions/servings, leftovers and real price APIs are not modelled.
* Not medical/dietary advice; allergen data is a simplified model and must not replace label reading.

## Milestone mapping
W2 PEAS/architecture · W4 tools+planner · W6 memory+reward+bandit · W8 optional planner/critic split (`critic.py`) ·
W11 LLM + TF-IDF retrieval · W13-14 `evaluate.py`, traces (`tools.trace`), `tests/` · W15 this repo.
Commit incrementally to show progressive development (assignment requirement).

## Ideas for extension
Receipt-photo parsing (multimodal), real price/recipe APIs or MCP, Streamlit UI, true Planner/Critic multi-agent with LLM debate, SHAP-style score-component explanations.
