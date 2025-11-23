import pandas as pd
import mlflow
import xgboost as xgb
import joblib
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

def load_data():
    df_hist = pd.read_csv("")

    audit_url = "http://audit-app:8001/audit/labels"
    df_audit = pd.DataFrame(requests.get(audit_url).json())

    df = pd.concat([df_hist, df_audit], axis=0)
    df.to_csv("combined.csv", index=False)

    return "combined.csv"

def preprocess(path):
    df = pd.read_csv(path)

    y = df["label"]
    X = df.drop(columns=["label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    pd.DataFrame(X_scaled).to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)

    return "X.csv", "y.csv"

def train_model():
    params = yaml.safe_load(open("params.yaml"))["model"]

    X = pd.read_csv("X.csv")
    y = pd.read_csv("y.csv")

    model = xgb.XGBClassifier(**params)

    mlflow.start_run()

    mlflow.log_params(params)

    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("scaler.pkl")

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    return run_id

def evaluate(run_id):
    X = pd.read_csv("X.csv")
    y = pd.read_csv("y.csv").values.flatten()

    model = joblib.load("model.pkl")
    preds = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds)
    }

    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.end_run()

    return metrics

def register_model(run_id):
    client = mlflow.tracking.MlflowClient()

    name = "fraud_detection_model"
    model_uri = f"runs:/{run_id}/model.pkl"

    result = mlflow.register_model(model_uri, name)

    client.transition_model_version_stage(
        name=name,
        version=result.version,
        # change to alias "Production"
        stage="Production",
        archive_existing_versions=True
    )

    return result.version

# def trigger_reload():
#     url = "http://model-serving:8000/load_model"
#     r = requests.post(url, json={"model_name": "fraud_detection_model"})
#     return r.json()

def retrain_pipeline():
    print("Loading data...")
    path = load_data()

    print("Preprocessing...")
    X, y = preprocess(path)

    print("Training...")
    run_id = train_model()

    print("Evaluating...")
    evaluate(run_id)

    print("Registering...")
    register_model(run_id)

    # print("Triggering reload...")
    # trigger_reload()

    print("Retraining pipeline completed.")


if __name__ == "__main__":
    retrain_pipeline()
