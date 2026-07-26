#src/domains/restaurants/pipeline/build_restaurants_map.py

from pathlib import Path

from src.domains.restaurants.preprocessing.load_data import (
    load_raw_data
)

from src.domains.restaurants.preprocessing.feature_engineering import (
    filter_restaurants,
    create_tags_text,
    compute_popularity_score,
    create_visual_sizes,
    create_macro_genres,
    create_region_nodes,
    create_landmark_restaurants
)

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
)

from src.atlas.embeddings.dimensionality_reduction import (
    get_umap_projection
)

from src.atlas.clustering.clustering import (
    compute_clusters
)

from src.atlas.clustering.region_labels import (
    create_region_labels
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[4]

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "restaurant_map_v1.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIG
# =========================================================

MIN_REVIEWS = 20

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# LOAD
# =========================================================

print("\n[1/7] Loading restaurant dataset...")

businesses, reviews, tips = load_raw_data()


# =========================================================
# FILTER
# =========================================================

print("\n[2/7] Filtering restaurants...")

restaurants = filter_restaurants(
    businesses,
    min_reviews=MIN_REVIEWS
)

print(
    f"Restaurants remaining: {len(restaurants):,}"
)


# =========================================================
# TAG PROCESSING
# =========================================================

print("\n[3/7] Processing categories + tips...")

restaurants = create_tags_text(
    restaurants,
    tips
)


# =========================================================
# POPULARITY
# =========================================================

restaurants = compute_popularity_score(
    restaurants
)

restaurants = create_visual_sizes(
    restaurants,
    strength=VISUAL_SIZE_STRENGTH
)

restaurants = create_macro_genres(
    restaurants
)


# =========================================================
# EMBEDDINGS
# =========================================================

print("\n[4/7] Building TF-IDF embeddings...")

tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(
        restaurants
    )
)


# =========================================================
# UMAP
# =========================================================

print("\n[5/7] Computing UMAP projection...")

embedding, umap_model = (
    get_umap_projection(
        tfidf_matrix
    )
)

restaurants["umap_x"] = embedding[:, 0]
restaurants["umap_y"] = embedding[:, 1]


# =========================================================
# CLUSTERING
# =========================================================

print("\n[6/7] Computing clusters...")

restaurants = compute_clusters(
    restaurants
)

restaurants = create_region_labels(
    restaurants
)


# =========================================================
# LOD
# =========================================================

print("\n[7/7] Creating LOD layers...")

regions = create_region_nodes(
    restaurants
)

landmarks = create_landmark_restaurants(
    restaurants,
    min_reviews=500
)


# =========================================================
# EXPORT CSV
# =========================================================

restaurants.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# EXPORT JSON
# =========================================================

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "restaurants"
)

JSON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

restaurants.to_json(
    JSON_DIR / "atlas.json",
    orient="records"
)

regions.to_json(
    JSON_DIR / "regions.json",
    orient="records"
)

landmarks.to_json(
    JSON_DIR / "landmarks.json",
    orient="records"
)

print("\n✅ Restaurant Atlas complete!")