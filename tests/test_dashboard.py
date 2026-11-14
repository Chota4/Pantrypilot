"""Headless smoke test of the Streamlit dashboard (skipped if streamlit is not installed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "dashboard.py")


def test_dashboard_runs_and_plans():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    next(b for b in at.sidebar.button if b.label == "Load demo pantry").click().run()
    assert not at.exception
    next(b for b in at.button if b.label == "Plan meals").click().run()
    assert not at.exception
    assert any("Day 1" in s.value for s in at.subheader)
