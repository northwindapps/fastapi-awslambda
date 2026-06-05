from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from recommender import load_model, get_recommendations

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts.update(load_model())
    yield
    artifacts.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.get("/recommend/{anime_name}")
def recommend(anime_name: str, n: int = Query(default=10, ge=1, le=50)):
    results = get_recommendations(anime_name, artifacts, n)
    if not results:
        raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found in dataset.")
    return {"anime": anime_name, "recommendations": results}
