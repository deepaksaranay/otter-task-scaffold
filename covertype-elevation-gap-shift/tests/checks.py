"""Deterministic checks for otter/covertype-elevation-gap-shift."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(os.environ.get("OTTER_APP_DIR", "/app"))
TESTS_DIR = Path(__file__).resolve().parent
TRAIN = APP / "task_inputs" / "train.csv"
TEST_FEATS = APP / "task_inputs" / "test_features.csv"
OUT = APP / "outputs"
LABELS = TESTS_DIR / "test_labels.csv"  # hidden; lives only in tests/
CLASSES = list(range(1, 8))
CHECK_ERRORS = (AssertionError, OSError, ValueError, KeyError, IndexError, TypeError)


def _macro_f1(y_true, y_pred):
    """Macro-F1 over classes that occur in y_true."""
    scores = []
    for c in CLASSES:
        support = int((y_true == c).sum())
        if support == 0:
            continue
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = support - tp
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def _min_class_recall(y_true, y_pred, min_support=50):
    recalls = [
        float(((y_true == c) & (y_pred == c)).sum()) / int((y_true == c).sum())
        for c in CLASSES
        if int((y_true == c).sum()) >= min_support
    ]
    assert recalls, "no class has enough support"
    return min(recalls)


def _load_pred(path, expected_ids):
    assert path.exists(), f"missing {path.name}"
    df = pd.read_csv(path)
    assert list(df.columns) == ["Id", "Cover_Type"], f"bad columns in {path.name}"
    assert not df.isna().any().any(), "missing values"
    assert df.Id.is_unique, "duplicate ids"
    assert set(df.Id) == set(expected_ids), "ids do not match the expected set"
    assert (df.Cover_Type == df.Cover_Type.round()).all(), "labels must be integers"
    assert df.Cover_Type.isin(CLASSES).all(), "labels must be in 1-7"
    return df.set_index("Id").Cover_Type.astype(int)


def _val_ids():
    assert (OUT / "val_ids.csv").exists(), "missing val_ids.csv"
    df = pd.read_csv(OUT / "val_ids.csv")
    assert list(df.columns) == ["Id"], "val_ids.csv must have exactly the column Id"
    return df.Id


def check_val_ids_valid():
    """Validation ids are unique, exist in train.csv and cover >= 5% of rows."""
    tr, ids = pd.read_csv(TRAIN), _val_ids()
    assert ids.is_unique, "duplicate validation ids"
    assert ids.isin(tr.Id).all(), "validation ids not in train.csv"
    assert len(ids) >= 0.05 * len(tr), "validation set smaller than 5% of train"


def check_val_band_is_elevation_gap():
    """Validation rows are exactly the training rows inside one elevation range, with data on both sides."""
    tr, ids = pd.read_csv(TRAIN), _val_ids()
    lo = tr.loc[tr.Id.isin(ids), "Elevation"].min()
    hi = tr.loc[tr.Id.isin(ids), "Elevation"].max()
    inside = tr.Elevation.between(lo, hi)
    assert set(tr.Id[inside]) == set(ids), "training rows inside the validation elevation range were not held out"
    assert (tr.Elevation < lo).mean() >= 0.01, "too few training rows below the validation band"
    assert (tr.Elevation > hi).mean() >= 0.01, "too few training rows above the validation band"


def check_val_predictions_valid():
    """val_predictions.csv is well formed and covers exactly the validation ids."""
    _load_pred(OUT / "val_predictions.csv", _val_ids())


def check_val_macro_f1_sanity():
    """Macro-F1 on the validation band, recomputed from train.csv labels, clears a low sanity bar."""
    tr = pd.read_csv(TRAIN).set_index("Id")
    pred = _load_pred(OUT / "val_predictions.csv", _val_ids())
    truth = tr.loc[pred.index, "Cover_Type"]
    assert _macro_f1(truth.values, pred.values) >= 0.40, "validation macro-F1 below 0.40"


def check_test_predictions_valid():
    """test_predictions.csv is well formed and covers exactly the test ids."""
    _load_pred(OUT / "test_predictions.csv", pd.read_csv(TEST_FEATS).Id)


def check_test_predictions_not_degenerate():
    """No class holds more than 90% of the test predictions."""
    pred = _load_pred(OUT / "test_predictions.csv", pd.read_csv(TEST_FEATS).Id)
    assert pred.value_counts(normalize=True).max() <= 0.90, "degenerate prediction distribution"


CHECKS = [
    ("val-ids-valid", "shift-aware-validation", check_val_ids_valid),
    ("val-band-is-elevation-gap", "shift-aware-validation", check_val_band_is_elevation_gap),
    ("val-predictions-valid", "validation-modeling", check_val_predictions_valid),
    ("val-macro-f1-sanity", "validation-modeling", check_val_macro_f1_sanity),
    ("test-predictions-valid", "final-test-predictions", check_test_predictions_valid),
    ("test-predictions-not-degenerate", "final-test-predictions", check_test_predictions_not_degenerate),
]


def _test_metrics():
    pred = _load_pred(OUT / "test_predictions.csv", pd.read_csv(TEST_FEATS).Id)
    lab = pd.read_csv(LABELS).set_index("Id").Cover_Type
    y_true, y_pred = lab.loc[pred.index].values, pred.values
    return {
        "test_macro_f1": _macro_f1(y_true, y_pred),
        "min_class_recall": _min_class_recall(y_true, y_pred),
    }


def evaluate(context=None):
    """Run every check once; return per-criterion results and measured metrics."""
    criteria = []
    for cid, mid, fn in CHECKS:
        try:
            fn()
            criteria.append({"id": cid, "milestone_id": mid, "passed": True, "detail": ""})
        except CHECK_ERRORS as exc:  # a check fails by raising
            criteria.append({"id": cid, "milestone_id": mid, "passed": False, "detail": str(exc)})
    try:
        metrics = _test_metrics()
    except CHECK_ERRORS:
        metrics = {"test_macro_f1": 0.0, "min_class_recall": 0.0}
    return {"criteria": criteria, "metrics": metrics}
