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
