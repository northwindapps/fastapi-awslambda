import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

S3_BUCKET = os.getenv("MODEL_BUCKET", "")
S3_PREFIX = os.getenv("MODEL_PREFIX", "models")

KNN_MODEL_PATH = os.getenv("KNN_MODEL_PATH", "knnModel/knn_anime_recommender_model.pkl")
ANIME_FEATURES_PATH = os.getenv("ANIME_FEATURES_PATH", "knnModel/anime_features_matrix.pkl")
ANIME_TO_IDX_PATH = os.getenv("ANIME_TO_IDX_PATH", "knnModel/anime_to_idx_map.pkl")
IDX_TO_ANIME_PATH = os.getenv("IDX_TO_ANIME_PATH", "knnModel/idx_to_anime_map.pkl")

SVD_ITEM_FACTORS_PATH = os.getenv("SVD_ITEM_FACTORS_PATH", "svdModel/svd_item_factors.npy")
SVD_NAME_TO_ID_PATH = os.getenv("SVD_NAME_TO_ID_PATH", "svdModel/svd_anime_name_to_original_id_map.pkl")
SVD_ID_TO_INNER_PATH = os.getenv("SVD_ID_TO_INNER_PATH", "svdModel/svd_anime_id_to_inner_id_map.pkl")
SVD_INNER_TO_NAME_PATH = os.getenv("SVD_INNER_TO_NAME_PATH", "svdModel/svd_inner_id_to_anime_name_map.pkl")


def _resolve(relative_path: str) -> str:
    """Download from S3 to /tmp if MODEL_BUCKET is set, otherwise use local path."""
    if not S3_BUCKET:
        return relative_path
    import boto3
    local = f"/tmp/{relative_path.replace('/', '_')}"
    if os.path.exists(local):
        return local

    s3_key = f"{S3_PREFIX}/{relative_path}" if S3_PREFIX else relative_path
    s3 = boto3.client("s3")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            s3.download_file(S3_BUCKET, s3_key, local)
            return local
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{s3_key} after {max_retries} attempts: {e}")
            import time
            time.sleep(2 ** attempt)  # exponential backoff


def load_model():
    return {
        "knn": joblib.load(_resolve(KNN_MODEL_PATH)),
        "anime_features": joblib.load(_resolve(ANIME_FEATURES_PATH)),
        "anime_to_idx": joblib.load(_resolve(ANIME_TO_IDX_PATH)),
        "idx_to_anime": joblib.load(_resolve(IDX_TO_ANIME_PATH)),
        "svd_item_factors": np.load(_resolve(SVD_ITEM_FACTORS_PATH)),
        "svd_name_to_id": joblib.load(_resolve(SVD_NAME_TO_ID_PATH)),
        "svd_id_to_inner": joblib.load(_resolve(SVD_ID_TO_INNER_PATH)),
        "svd_inner_to_name": joblib.load(_resolve(SVD_INNER_TO_NAME_PATH)),
    }


def _recommend_knn(anime_name: str, artifacts: dict, n: int) -> list[dict]:
    anime_to_idx = artifacts["anime_to_idx"]
    idx_to_anime = artifacts["idx_to_anime"]
    knn = artifacts["knn"]
    anime_features = artifacts["anime_features"]

    if anime_name not in anime_to_idx:
        return []

    idx = anime_to_idx[anime_name]
    distances, indices = knn.kneighbors(anime_features[idx], n_neighbors=n + 1)
    distances = distances.flatten()
    indices = indices.flatten()

    results = []
    for i in range(1, len(indices)):
        results.append({
            "anime": idx_to_anime[indices[i]],
            "similarity": round(float(1 - distances[i]), 4),
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def _recommend_svd(anime_name: str, artifacts: dict, n: int) -> list[dict]:
    name_to_id = artifacts["svd_name_to_id"]
    id_to_inner = artifacts["svd_id_to_inner"]
    inner_to_name = artifacts["svd_inner_to_name"]
    item_factors = artifacts["svd_item_factors"]

    original_id = name_to_id.get(anime_name)
    if original_id is None:
        return []
    inner_id = id_to_inner.get(original_id)
    if inner_id is None:
        return []

    target_vec = item_factors[inner_id].reshape(1, -1)
    similarities = cosine_similarity(target_vec, item_factors)[0]
    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        if int(idx) == inner_id:
            continue
        name = inner_to_name.get(int(idx))
        if name is None:
            continue
        results.append({"anime": name, "similarity": round(float(similarities[idx]), 4)})
        if len(results) >= n:
            break

    return results


def get_recommendations(anime_name: str, artifacts: dict, n: int = 10, model: str = "knn") -> list[dict]:
    if model == "svd":
        return _recommend_svd(anime_name, artifacts, n)
    return _recommend_knn(anime_name, artifacts, n)
