"""Use mlflow.search_runs() to rank runs, pick the best by F1, and save it for the app."""
import json
import os

import joblib
import mlflow
import mlflow.sklearn

from src.train import load_config


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["experiment_name"]],
        order_by=["metrics.f1 DESC"],
    )
    columns = [
        "tags.mlflow.runName", "params.model_type",
        "metrics.accuracy", "metrics.precision", "metrics.recall",
        "metrics.f1", "metrics.roc_auc",
    ]
    print(runs[columns].to_string(index=False))

    best = runs.iloc[0]
    print("\nBEST RUN:", best["tags.mlflow.runName"], "| F1 =", round(best["metrics.f1"], 4))

    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.joblib")

    info = {
        "run_id": best["run_id"],
        "run_name": best["tags.mlflow.runName"],
        "model_type": best["params.model_type"],
        "accuracy": best["metrics.accuracy"],
        "precision": best["metrics.precision"],
        "recall": best["metrics.recall"],
        "f1": best["metrics.f1"],
        "roc_auc": best["metrics.roc_auc"],
    }
    with open("models/best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("Saved models/best_model.joblib")


if __name__ == "__main__":
    main()