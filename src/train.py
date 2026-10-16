"""Train every configuration in configs/config.yaml and log each run to MLflow."""
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics
from src.preprocess import build_preprocessor, load_data, make_split

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = load_data(cfg["data"]["path"])
    X_train, X_test, y_train, y_test = make_split(
        df, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )

    for run_cfg in cfg["runs"]:
        model = MODEL_CLASSES[run_cfg["model_type"]](**run_cfg["params"])
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        with mlflow.start_run(run_name=run_cfg["name"]):
            mlflow.log_param("model_type", run_cfg["model_type"])
            mlflow.log_params(run_cfg["params"])
            mlflow.log_param("data_description", cfg["data"]["description"])
            mlflow.log_param("data_test_size", cfg["data"]["test_size"])
            mlflow.log_param("data_random_state", cfg["data"]["random_state"])
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_test_rows", len(X_test))

            pipeline.fit(X_train, y_train)
            metrics = compute_metrics(
                y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")
            print(run_cfg["name"], metrics)


if __name__ == "__main__":
    main()