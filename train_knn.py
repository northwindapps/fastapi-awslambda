import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def infer_task(y):
    # simple heuristic
    try:
        nunique = y.nunique()
    except Exception:
        return "regression"
    if y.dtype.kind in 'O' or nunique <= 20:
        return "classification"
    return "regression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--output", default="knn_model.joblib", help="Output model filename")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in CSV")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    task = infer_task(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("knn-mlflow-sample")

    with mlflow.start_run():
        if task == "classification":
            model = KNeighborsClassifier(n_neighbors=args.n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=args.n_neighbors)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "classification":
            metric = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", float(metric))
        else:
            metric = mean_squared_error(y_test, preds)
            mlflow.log_metric("mse", float(metric))

        mlflow.log_param("n_neighbors", args.n_neighbors)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("task", task)

        # save locally and log as artifact/model
        joblib.dump(model, args.output)
        mlflow.sklearn.log_model(model, "model")

        print("Saved model to:", args.output)
        print("MLflow run id:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
