import joblib
import numpy as np
import pytest

from src.evaluate import compute_metrics
from src.preprocess import load_data, make_split
from src.train import load_config


@pytest.fixture(scope="module")
def model_and_data():
    cfg = load_config()
    model = joblib.load("models/best_model.joblib")
    df = load_data(cfg["data"]["path"])
    _, X_test, _, y_test = make_split(df, cfg["data"]["test_size"], cfg["data"]["random_state"])
    return model, X_test, y_test


def test_predictions_have_correct_type_and_shape(model_and_data):
    model, X_test, _ = model_and_data
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probas.shape == (len(X_test), 2)


def test_model_meets_minimum_performance(model_and_data):
    model, X_test, y_test = model_and_data
    metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    assert metrics["accuracy"] >= 0.75
    assert metrics["roc_auc"] >= 0.80