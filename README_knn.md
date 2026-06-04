KNN training with MLflow

Train and log a KNN model from a CSV file using `train_knn.py`.

Usage:

```powershell
python train_knn.py --csv sample_knn.csv --target target
```

Options:
- `--n-neighbors` default 3
- `--test-size` default 0.2
- `--tracking-uri` default `sqlite:///mlflow.db`
- `--output` default `knn_model.joblib`

Example run:

```powershell
python train_knn.py --csv sample_knn.csv --target target --n-neighbors 3
```

View MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`.
