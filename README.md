# FastAPI Sample

## Run locally

1. Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
uvicorn main:app --reload
```

4. Open your browser:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/items/1?q=hello
- http://127.0.0.1:8000/docs (interactive API docs)

## AWS Lambda example

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Deploy `lambda_main.py` to AWS Lambda.

3. Set the Lambda handler to:

```
lambda_main.handler
```

4. Use API Gateway or Lambda Function URLs to invoke the app.

## AWS SAM deployment example

This project includes `template.yaml` for AWS SAM.

1. Install AWS SAM CLI and configure AWS credentials.

2. Build the application:

```powershell
sam build
```

3. Deploy with guided mode:

```powershell
sam deploy --guided
```

4. After deployment, SAM will provide the API endpoint.

> Make sure your Lambda handler is still `lambda_main.handler` and your dependencies are installed in the deployment package.

## GitHub Actions auto-deploy

A GitHub Actions workflow is included at `.github/workflows/aws-sam-deploy.yml`.
It deploys automatically when you push to `main` or `master`.

### Configure repository secrets

In your GitHub repo settings, add:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (for example `ap-northeast-1`)

### Workflow behavior

On push to `main` or `master`, the workflow:

1. checks out the repo
2. configures AWS credentials
3. installs Python and AWS SAM CLI
4. installs dependencies
5. runs `sam build`
6. runs `sam deploy --resolve-s3 --no-confirm-changeset`

If you want, you can also change the workflow to use a specific S3 bucket or a different stack name.

## Simple MLflow sample

This project includes `mlflow_sample.py`, a minimal example that:

- trains a `LinearRegression` model on the sklearn diabetes dataset
- logs parameters, metrics, and the model to a local MLflow tracking folder

### Run the sample

```powershell
pip install -r requirements.txt
python mlflow_sample.py
```

### View the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` in your browser to inspect the run.
