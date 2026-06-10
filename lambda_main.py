from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, HTTPException, Query
from mangum import Mangum
from recommender import load_model, get_recommendations

artifacts = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    artifacts.update(load_model())
    yield
    artifacts.clear()


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/api/v1")


@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI on AWS Lambda!"}


@router.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@router.get("/recommend/{anime_name}")
async def recommend(anime_name: str, n: int = Query(default=10, ge=1, le=50), model: str = Query(default="knn", pattern="^(knn|svd)$")):
    results = get_recommendations(anime_name, artifacts, n, model)
    if not results:
        raise HTTPException(status_code=404, detail=f"Anime '{anime_name}' not found in dataset.")
    return {"anime": anime_name, "recommendations": results}


app.include_router(router)
handler = Mangum(app)
