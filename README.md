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
```

4. Open your browser:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/items/1?q=hello
- http://127.0.0.1:8000/docs (interactive API docs)
