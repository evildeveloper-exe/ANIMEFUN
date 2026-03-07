"""
============================================================
  AnimeSensei – Local ML Model Trainer
  ============================================================
  Trains a local recommendation model from anime_data.json.
  NO API CALLS. NO INTERNET. Runs 100% on your machine.

  Algorithms used:
  ─────────────────────────────────────────────────────────
  1. TF-IDF + Cosine Similarity  → Content-based filtering
  2. KNN (K-Nearest Neighbors)   → Feature-space similarity
  3. SVD (Truncated SVD)         → Latent feature discovery
  4. Random Forest Classifier    → Quiz answer → anime type
  5. Collaborative Filtering     → User watchlist patterns
  6. Hybrid Scorer               → Combines all models

  Output: saves trained models to /models/ folder
  ─────────────────────────────────────────────────────────

  Install deps (one time):
      pip install numpy scikit-learn flask joblib

  Run trainer:
      python model_trainer.py

  Then run server:
      python local_server.py
============================================================
"""

import json
import os
import math
import random
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

DATA_FILE   = "anime_data.json"
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# STEP 1 – Load & parse dataset
# ──────────────────────────────────────────────
def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)
    anime = raw["anime"]
    print(f"[LOAD] {len(anime)} anime loaded from {filepath}")
    return anime


# ──────────────────────────────────────────────
# STEP 2 – Feature engineering
# ──────────────────────────────────────────────
def build_feature_matrix(anime: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Converts each anime into a numeric feature vector:
      [rating_norm, popularity_norm, episodes_norm, year_norm,
       genre_onehot × N, mood_onehot × M, hidden_gem, seasonal, trending]
    Returns: (matrix, feature_names)
    """
    ALL_GENRES = sorted({g for a in anime for g in a.get("genres", [])})
    ALL_MOODS  = sorted({m for a in anime for m in a.get("mood", [])})

    feature_names = (
        ["rating", "popularity", "episodes_log", "year_norm"]
        + [f"genre_{g}" for g in ALL_GENRES]
        + [f"mood_{m}"  for m in ALL_MOODS]
        + ["hidden_gem", "seasonal", "trending"]
    )

    rows = []
    for a in anime:
        row = [
            a.get("rating", 0) / 10.0,
            a.get("popularity", 0) / 100.0,
            math.log1p(a.get("episodes", 1)) / math.log1p(1000),
            (a.get("year", 2000) - 1980) / 50.0,
        ]
        row += [1 if g in a.get("genres", []) else 0 for g in ALL_GENRES]
        row += [1 if m in a.get("mood",   []) else 0 for m in ALL_MOODS]
        row += [
            int(a.get("hidden_gem", False)),
            int(a.get("seasonal", False)),
            int(a.get("trending", False)),
        ]
        rows.append(row)

    matrix = np.array(rows, dtype=np.float32)
    print(f"[FEATURES] Matrix shape: {matrix.shape}  features: {len(feature_names)}")
    return matrix, feature_names, ALL_GENRES, ALL_MOODS


# ──────────────────────────────────────────────
# STEP 3 – TF-IDF on text (synopsis + tags + genres)
# ──────────────────────────────────────────────
def build_tfidf(anime: list[dict]) -> tuple:
    """
    Builds a TF-IDF matrix from combined text fields.
    Used for content-based similarity ("describe what you want").
    """
    corpus = []
    for a in anime:
        text = " ".join([
            a.get("title", ""),
            a.get("synopsis", ""),
            " ".join(a.get("genres", [])),
            " ".join(a.get("tags", [])),
            " ".join(a.get("mood", [])),
            a.get("studio", ""),
        ])
        corpus.append(text.lower())

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"[TFIDF] Matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# ──────────────────────────────────────────────
# STEP 4 – KNN model (feature-based nearest neighbours)
# ──────────────────────────────────────────────
def train_knn(feature_matrix: np.ndarray, n_neighbors: int = 11) -> NearestNeighbors:
    """
    Fits a KNN model on the numeric feature matrix.
    Used to find anime most similar in feature space.
    n_neighbors = 11 so we can drop self and return top 10.
    """
    k = min(n_neighbors, len(feature_matrix))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(feature_matrix)
    print(f"[KNN] Trained with k={k}, metric=cosine")
    return knn


# ──────────────────────────────────────────────
# STEP 5 – SVD latent features
# ──────────────────────────────────────────────
def train_svd(feature_matrix: np.ndarray, n_components: int = 12) -> tuple:
    """
    Reduces feature space to latent dimensions via Truncated SVD.
    Captures hidden patterns (e.g. "dark psychological thrillers").
    """
    n = min(n_components, feature_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n, random_state=42)
    latent = svd.fit_transform(feature_matrix)
    explained = svd.explained_variance_ratio_.sum()
    print(f"[SVD] {n} components, explained variance: {explained:.1%}")
    return svd, latent


# ──────────────────────────────────────────────
# STEP 6 – Quiz classifier (quiz answers → anime cluster)
# ──────────────────────────────────────────────
def build_quiz_training_data(anime: list[dict], all_genres, all_moods):
    """
    Synthetically generates (quiz_answer_vector → anime_index) pairs
    so the classifier learns which quiz profile maps to which anime.
    """
    MOODS_MAP   = {"dark": 0, "funny": 1, "emotional": 2, "relaxing": 3, "intense": 4}
    GENRES_MAP  = {g: i for i, g in enumerate(all_genres)}
    LENGTH_MAP  = {"1": 0, "short": 1, "medium": 2, "long": 3}
    INTENS_MAP  = {"intense": 0, "balanced": 1, "slow": 2, "action": 3}
    CHAR_MAP    = {"underdog": 0, "genius": 1, "op": 2, "complex": 3}
    POP_MAP     = {"popular": 0, "hidden": 1, "seasonal": 2, "any": 3}

    X, y = [], []

    for idx, a in enumerate(anime):
        # Generate multiple synthetic quiz answers that correctly point to this anime
        for _ in range(8):   # 8 synthetic users per anime
            mood_val   = random.choice(a.get("mood", ["dark"]))
            genre_val  = random.choice(a.get("genres", ["Action"]))
            eps        = a.get("episodes", 12)
            if eps == 1:           length_val = "1"
            elif eps <= 25:        length_val = "short"
            elif eps <= 75:        length_val = "medium"
            else:                  length_val = "long"

            intens_val = random.choice(["intense", "balanced", "slow", "action"])
            char_val   = random.choice(["underdog", "genius", "op", "complex"])
            pop_val    = "hidden" if a.get("hidden_gem") else (
                         "seasonal" if a.get("seasonal") else "popular")

            vec = [
                MOODS_MAP.get(mood_val, 0) / 4.0,
                GENRES_MAP.get(genre_val, 0) / max(len(all_genres) - 1, 1),
                LENGTH_MAP.get(length_val, 0) / 3.0,
                INTENS_MAP.get(intens_val, 0) / 3.0,
                CHAR_MAP.get(char_val, 0) / 3.0,
                POP_MAP.get(pop_val, 0) / 3.0,
                a.get("rating", 7) / 10.0,
            ]
            # Add noise so the model doesn't overfit
            vec = [v + random.gauss(0, 0.05) for v in vec]
            X.append(vec)
            y.append(idx)

    return np.array(X, dtype=np.float32), np.array(y)


def train_quiz_classifier(anime, all_genres, all_moods):
    """Trains a Gradient Boosting classifier on synthetic quiz data."""
    X, y = build_quiz_training_data(anime, all_genres, all_moods)
    clf = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"[QUIZ-CLF] GradientBoosting trained | train acc: {train_acc:.1%} | samples: {len(X)}")
    return clf


# ──────────────────────────────────────────────
# STEP 7 – Collaborative filtering (watchlist-based)
# ──────────────────────────────────────────────
def build_collab_model(anime: list[dict]) -> dict:
    """
    Builds a simple item-item collaborative filtering model
    using the explicit similar_to relationships in the dataset
    as a proxy for user co-watch history.

    Returns a co-occurrence matrix: {anime_id: {anime_id: score}}
    """
    co_occur = defaultdict(lambda: defaultdict(float))

    for a in anime:
        aid = a["id"]
        for sid in a.get("similar_to", []):
            co_occur[aid][sid] += 1.0
            co_occur[sid][aid] += 1.0   # symmetric

    # Normalise by item frequency
    for aid in co_occur:
        total = sum(co_occur[aid].values())
        for sid in co_occur[aid]:
            co_occur[aid][sid] /= total

    print(f"[COLLAB] Built item-item co-occurrence for {len(co_occur)} items")
    return dict(co_occur)


# ──────────────────────────────────────────────
# STEP 8 – Hybrid scorer
# ──────────────────────────────────────────────
def build_hybrid_scorer(anime, feature_matrix, latent_matrix, tfidf_matrix):
    """
    Pre-computes combined similarity scores for every anime pair.
    Hybrid = 0.4 × cosine(features) + 0.3 × cosine(tfidf) + 0.3 × cosine(svd_latent)
    Stores top-10 for each anime for fast lookup.
    """
    print("[HYBRID] Computing cosine similarities (this may take a moment)…")

    feat_sim   = cosine_similarity(feature_matrix)
    tfidf_sim  = cosine_similarity(tfidf_matrix)
    latent_sim = cosine_similarity(latent_matrix)

    W_FEAT, W_TFIDF, W_SVD = 0.4, 0.3, 0.3
    hybrid_sim = W_FEAT * feat_sim + W_TFIDF * tfidf_sim + W_SVD * latent_sim

    # Store top-10 neighbours for each anime (excluding self)
    top_similar = {}
    for i, a in enumerate(anime):
        scores = list(enumerate(hybrid_sim[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_similar[a["id"]] = [
            {"id": anime[j]["id"], "score": float(s)}
            for j, s in scores
            if j != i
        ][:10]

    print(f"[HYBRID] Pre-computed top-10 for {len(top_similar)} anime")
    return top_similar, hybrid_sim


# ──────────────────────────────────────────────
# STEP 9 – Save all models
# ──────────────────────────────────────────────
def save_models(bundle: dict):
    for name, obj in bundle.items():
        path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(obj, path)
        size = path.stat().st_size / 1024
        print(f"[SAVE] {name}.pkl  ({size:.1f} KB)")


# ──────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────
def train():
    print("\n" + "="*54)
    print("   AnimeSensei – Local ML Training Pipeline")
    print("="*54 + "\n")

    # 1. Load data
    anime = load_dataset(DATA_FILE)

    # 2. Feature matrix
    feat_matrix, feat_names, all_genres, all_moods = build_feature_matrix(anime)

    # 3. TF-IDF
    tfidf_vec, tfidf_matrix = build_tfidf(anime)

    # 4. KNN
    knn = train_knn(feat_matrix)

    # 5. SVD
    svd, latent_matrix = train_svd(feat_matrix)

    # 6. Quiz classifier
    quiz_clf = train_quiz_classifier(anime, all_genres, all_moods)

    # 7. Collaborative filtering
    collab = build_collab_model(anime)

    # 8. Hybrid similarity
    top_similar, hybrid_matrix = build_hybrid_scorer(
        anime, feat_matrix, latent_matrix, tfidf_matrix
    )

    # 9. Scaler (for normalising new query vectors)
    scaler = MinMaxScaler()
    scaler.fit(feat_matrix)

    # 10. Save everything
    print("\n[SAVE] Writing models to ./models/")
    save_models({
        "anime_list":    anime,
        "feat_matrix":   feat_matrix,
        "feat_names":    feat_names,
        "all_genres":    all_genres,
        "all_moods":     all_moods,
        "tfidf_vec":     tfidf_vec,
        "tfidf_matrix":  tfidf_matrix,
        "knn":           knn,
        "svd":           svd,
        "latent_matrix": latent_matrix,
        "quiz_clf":      quiz_clf,
        "collab":        collab,
        "top_similar":   top_similar,
        "hybrid_matrix": hybrid_matrix,
        "scaler":        scaler,
    })

    print("\n[✓] Training complete!")
    print(f"[✓] Models saved to: {MODELS_DIR.resolve()}")
    print("[✓] Run  python local_server.py  to start the server\n")


if __name__ == "__main__":
    train()
