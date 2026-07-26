#src/domains/movies/pipeline/build_movie_map.py

from pathlib import Path

from src.domains.movies.preprocessing.load_data import load_raw_data
from src.domains.movies.preprocessing.merge_data import compute_movie_stats, merge_movies
from src.domains.movies.preprocessing.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags,
    create_macro_genres,
    create_visual_sizes,
    create_region_nodes,
    create_landmark_movies
)
from src.atlas.clustering.region_labels import create_region_labels
from src.atlas.embeddings.tfidf_pipeline import get_tfidf_embeddings
from src.atlas.embeddings.dimensionality_reduction import get_umap_projection
from src.atlas.clustering.clustering import compute_clusters
import json

# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "data" / "processed" / "movie_map_v1.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIG (NEW - ONLY ADDITION)
# =========================================================

VISUAL_SIZE_STRENGTH = 1.8   # controls bubble contrast (1.2 subtle → 2.5 dramatic)


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

# UPDATED: now uses contrast strength parameter
final = create_visual_sizes(final, strength=VISUAL_SIZE_STRENGTH)


# =========================================================
# CLUSTERING (HDBSCAN / KMEANS depending on your setup)
# =========================================================

print("\n[7/7] Computing clusters...")
final = compute_clusters(final)

print("Creating region labels...")
final = create_region_labels(final)

# =========================================================
# LOD DATASETS
# =========================================================

print("Creating region nodes...")
regions = create_region_nodes(final)

print("Creating landmark movies...")
landmarks = create_landmark_movies(
    final,
    min_ratings=1000
)

# =========================================================
# EXPORT CSV
# =========================================================

final.to_csv(
    OUTPUT_PATH,
    index=False
)

# =========================================================
# EXPORT JSON
# =========================================================

JSON_DIR = ROOT / "frontend" / "public" / "data" / "movies"

JSON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# full movie universe
final.to_json(
    JSON_DIR / "atlas.json",
    orient="records"
)

# region layer
regions.to_json(
    JSON_DIR / "regions.json",
    orient="records"
)

# landmark layer
landmarks.to_json(
    JSON_DIR / "landmarks.json",
    orient="records"
)

print("\n✅ Pipeline complete!")