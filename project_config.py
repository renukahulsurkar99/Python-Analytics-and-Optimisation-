"""Paths and runtime helpers for the Group K coursework."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("AED_DATA_PATH", PROJECT_ROOT / "data" / "AED4weeks.csv"))
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "local_outputs"
QUICK_TEST = os.environ.get("AED_QUICK_TEST") == "1"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_OUTPUT_DIR / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


def choose_threshold(y_true, probabilities, minimum_recall):
    """Maximise validation precision subject to a required positive-class recall.

    Include zero so that an all-positive fallback is available when the original
    0.01--0.99 threshold grid has no feasible value. Ties keep the first threshold.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must be nonempty and equally sized.")
    if not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
        raise ValueError("Threshold selection needs binary labels and positive examples.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not 0 <= minimum_recall <= 1:
        raise ValueError("Required recall must lie between zero and one.")
    best = None
    for threshold in np.linspace(0.01, 0.99, 200):
        predicted = scores >= threshold
        recall = recall_score(labels, predicted, zero_division=0)
        precision = precision_score(labels, predicted, zero_division=0)
        if recall >= minimum_recall and (best is None or precision > best[1]):
            best = (float(threshold), float(precision), float(recall))
    if best is None:
        return (0.0, float((labels == 1).mean()), 1.0)
    return best


def capture_plots(script_path):
    """Save batch-script plots locally; they may describe private patient data."""
    import matplotlib.pyplot as plt

    directory = LOCAL_OUTPUT_DIR / "figures" / Path(script_path).stem
    directory.mkdir(parents=True, exist_ok=True)
    counter = 0

    def save_and_close(*args, **kwargs):
        nonlocal counter
        for number in list(plt.get_fignums()):
            counter += 1
            figure = plt.figure(number)
            figure.savefig(directory / f"figure_{counter:02}.png", dpi=120, bbox_inches="tight")
            plt.close(figure)

    plt.show = save_and_close


def save_model_summary(name, y_true, scores, predictions, threshold, parameters, quick_test):
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_OUTPUT_DIR / "model_summaries.json"
    summaries = json.loads(path.read_text()) if path.exists() else {}
    summaries[name] = {
        "quick_test": bool(quick_test),
        "test_size": int(len(y_true)),
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "parameters": parameters,
    }
    path.write_text(json.dumps(summaries, indent=2))
