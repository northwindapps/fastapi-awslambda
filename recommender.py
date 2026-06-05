import os
import joblib

KNN_MODEL_PATH = os.getenv("KNN_MODEL_PATH", "knn_anime_recommender_model.pkl")
ANIME_FEATURES_PATH = os.getenv("ANIME_FEATURES_PATH", "anime_features_matrix.pkl")
ANIME_TO_IDX_PATH = os.getenv("ANIME_TO_IDX_PATH", "anime_to_idx_map.pkl")
IDX_TO_ANIME_PATH = os.getenv("IDX_TO_ANIME_PATH", "idx_to_anime_map.pkl")


def load_model():
    return {
        "knn": joblib.load(KNN_MODEL_PATH),
        "anime_features": joblib.load(ANIME_FEATURES_PATH),
        "anime_to_idx": joblib.load(ANIME_TO_IDX_PATH),
        "idx_to_anime": joblib.load(IDX_TO_ANIME_PATH),
    }


def get_recommendations(anime_name: str, artifacts: dict, n: int = 10) -> list[dict]:
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
    for i in range(1, len(indices)):  # skip index 0 (the query anime itself)
        results.append({
            "anime": idx_to_anime[indices[i]],
            "similarity": round(float(1 - distances[i]), 4),
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)
