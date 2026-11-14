"""Agent memory: profile, episodic memory, recipe-level bandit statistics, tag preferences."""
from __future__ import annotations

import json
import math
from pathlib import Path


class Memory:
    def __init__(self, path: str | None = None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled
        self.profile = dict(allergens=[], diet="omnivore", budget=None, max_minutes=None)
        self.recipe_stats: dict = {}      # id -> {"n": int, "sum": float}
        self.tag_w: dict = {}             # tag -> EMA reward
        self.episodes: list = []
        self.reflections: list = []
        self.total_n = 0
        if self.path and self.path.exists():
            self.load()

    # ---- preference signal used by the planner --------------------------------
    def pref(self, recipe: dict, c: float = 0.3) -> float:
        """0.6*mean recipe reward + 0.4*mean tag preference + UCB exploration bonus."""
        if not self.enabled:
            return 0.0
        s = self.recipe_stats.get(recipe["id"], {"n": 0, "sum": 0.0})
        mean = s["sum"] / s["n"] if s["n"] else 0.0
        tags = [self.tag_w.get(t, 0.0) for t in recipe["tags"]]
        tag_avg = sum(tags) / len(tags) if tags else 0.0
        ucb = c * math.sqrt(math.log(self.total_n + 2) / (s["n"] + 1))
        return 0.6 * mean + 0.4 * tag_avg + ucb

    def update(self, recipe: dict, reward: float, lr: float = 0.3):
        if not self.enabled:
            return
        s = self.recipe_stats.setdefault(recipe["id"], {"n": 0, "sum": 0.0})
        s["n"] += 1
        s["sum"] += reward
        self.total_n += 1
        r = max(-1.0, min(1.5, reward))
        for t in recipe["tags"]:
            self.tag_w[t] = self.tag_w.get(t, 0.0) + lr * (r - self.tag_w.get(t, 0.0))

    def add_episode(self, ep: dict, reflection: str):
        if not self.enabled:
            return
        self.episodes.append(ep)
        self.reflections = (self.reflections + [reflection])[-20:]

    def disliked_tags(self, thr: float = -0.4) -> list:
        return sorted(t for t, w in self.tag_w.items() if w < thr)

    # ---- persistence ------------------------------------------------------------
    def save(self):
        if self.path:
            self.path.write_text(json.dumps(dict(profile=self.profile, recipe_stats=self.recipe_stats,
                                                 tag_w=self.tag_w, episodes=self.episodes[-200:],
                                                 reflections=self.reflections, total_n=self.total_n), indent=1))

    def load(self):
        d = json.loads(self.path.read_text())
        self.profile = d.get("profile", self.profile)
        self.recipe_stats = d.get("recipe_stats", {})
        self.tag_w = d.get("tag_w", {})
        self.episodes = d.get("episodes", [])
        self.reflections = d.get("reflections", [])
        self.total_n = d.get("total_n", 0)
