"""Streamlit execution checks with synthetic records and a mocked menu selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from machine_learning import user_interface_app as ui

ROOT = Path(__file__).resolve().parents[1]


def synthetic_records():
    rng = np.random.default_rng(14)
    size = 50
    los = rng.integers(30, 350, size=size)
    return pd.DataFrame({
        "ID": [f"DEMO-{i}" for i in range(size)],
        "Age": rng.integers(0, 100, size=size),
        "Day": rng.integers(1, 29, size=size),
        "DayofWeek": [ui.aed.DAY_ORDER[i % 7] for i in range(size)],
        "Period": rng.integers(0, 24, size=size),
        "LoS": los,
        "Breachornot": np.where(los > 240, "breach", "non-breach"),
        "HRG": ["Demo-A" if i % 2 else "Demo-B" for i in range(size)],
        "noofinvestigation": rng.integers(0, 5, size=size),
        "nooftreatment": rng.integers(0, 5, size=size),
        "noofpatients": rng.integers(10, 80, size=size),
    })


class DashboardTests(unittest.TestCase):
    def test_scheduling_page_does_not_require_patient_data(self):
        with patch.object(ui, "option_menu", return_value="Tasks 1-3: Optimisation"), \
             patch.object(ui.aed, "load_data", side_effect=AssertionError("Data should not load")):
            app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button[0].label, "Solve scheduling optimisation")

    def test_scheduling_button_returns_a_solution(self):
        with patch.object(ui, "option_menu", return_value="Tasks 1-3: Optimisation"):
            app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1,784.00")

    def test_data_sections_execute_with_synthetic_records(self):
        sections = [
            "Task 4a: Descriptive Statistics (Single Variables)",
            "Task 4b: Relationships Between Variables",
            "Task 5: Factors influencing Length of Stay and Breach",
            "Raw Data Access",
        ]
        for section in sections:
            with self.subTest(section=section), \
                 patch.object(ui, "option_menu", return_value=section), \
                 patch.object(ui.aed, "load_data", side_effect=lambda: synthetic_records()):
                app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertGreater(len(app.dataframe), 0)
                if section == "Raw Data Access":
                    # The string categorical columns must remain selectable under pandas 3.
                    categories = app.multiselect(key="ui_cat_cols").options
                    self.assertIn("HRG", categories)
                    next(button for button in app.button if button.label == "Apply filters").click().run()
                    self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
