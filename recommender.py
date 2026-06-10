import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

KNN_MODEL_PATH = os.getenv("KNN_MODEL_PATH", "knnModel/knn_anime_recommender_model.pkl")
ANIME_FEATURES_PATH = os.getenv("ANIME_FEATURES_PATH", "knnModel/anime_features_matrix.pkl")
ANIME_TO_IDX_PATH = os.getenv("ANIME_TO_IDX_PATH", "knnModel/anime_to_idx_map.pkl")
IDX_TO_ANIME_PATH = os.getenv("IDX_TO_ANIME_PATH", "knnModel/idx_to_anime_map.pkl")

SVD_MODEL_PATH = os.getenv("SVD_MODEL_PATH", "svdModel/svd_anime_recommender_model.pkl")
SVD_NAME_TO_ID_PATH = os.getenv("SVD_NAME_TO_ID_PATH", "svdModel/svd_anime_name_to_original_id_map.pkl")
SVD_ID_TO_INNER_PATH = os.getenv("SVD_ID_TO_INNER_PATH", "svdModel/svd_anime_id_to_inner_id_map.pkl")
SVD_INNER_TO_NAME_PATH = os.getenv("SVD_INNER_TO_NAME_PATH", "svdModel/svd_inner_id_to_anime_name_map.pkl")


def load_model():
    svd_model = joblib.load(SVD_MODEL_PATH)
    return {
        "knn": joblib.load(KNN_MODEL_PATH),
        "anime_features": joblib.load(ANIME_FEATURES_PATH),
        "anime_to_idx": joblib.load(ANIME_TO_IDX_PATH),
        "idx_to_anime": joblib.load(IDX_TO_ANIME_PATH),
        "svd_model": svd_model,
        "svd_item_factors": svd_model.qi,
        "svd_name_to_id": joblib.load(SVD_NAME_TO_ID_PATH),
        "svd_id_to_inner": joblib.load(SVD_ID_TO_INNER_PATH),
        "svd_inner_to_name": joblib.load(SVD_INNER_TO_NAME_PATH),
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
