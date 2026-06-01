from pathlib import Path

from src.preprocessing.load_data import load_raw_data
from src.preprocessing.merge_data import compute_movie_stats, merge_movies
from src.preprocessing.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags,
    create_macro_genres,
    create_visual_sizes
)

from src.embeddings.tfidf_pipeline import get_tfidf_embeddings
from src.embeddings.dimensionality_reduction import get_umap_projection
from src.clustering.clustering import compute_clusters


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "data" / "processed" / "movie_map_v1.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1/7] Loading raw data...")
movies, ratings, tags = load_raw_data()


# =========================================================
# PREPROCESSING
# =========================================================

print("\n[2/7] Computing movie statistics...")
movie_stats = compute_movie_stats(ratings)

merged = merge_movies(movie_stats, movies)

merged = compute_weighted_rating(merged)


print("\n[3/7] Processing tags...")
movie_tags = concatenate_tags(tags)

final = merged.merge(
    movie_tags,
    on="movieId",
    how="left"
)

final["tags_text"] = final["tags_text"].fillna("")


# =========================================================
# TF-IDF EMBEDDINGS
# =========================================================

print("\n[4/7] Building TF-IDF embeddings...")
tfidf_matrix, vectorizer = get_tfidf_embeddings(final)


# =========================================================
# UMAP PROJECTION
# =========================================================

print("\n[5/7] Computing UMAP projection...")
embedding, umap_model = get_umap_projection(tfidf_matrix)

final["umap_x"] = embedding[:, 0]
final["umap_y"] = embedding[:, 1]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\n[6/7] Creating visual + semantic features...")
final = create_macro_genres(final)
final = create_visual_sizes(final)


# =========================================================
# CLUSTERING (HDBSCAN)
# =========================================================

print("\n[7/7] Computing clusters...")
final = compute_clusters(final)


# =========================================================
# EXPORT
# =========================================================

final.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n✅ Pipeline complete!")
print(f"Saved to: {OUTPUT_PATH}")