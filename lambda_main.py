from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI on AWS Lambda!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

handler = Mangum(app)
