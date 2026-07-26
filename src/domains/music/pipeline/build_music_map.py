#src/domains/music/pipeline/build_music_map.py

from pathlib import Path

from src.domains.music.preprocessing.load_data import (
    load_raw_data
)

from src.domains.music.preprocessing.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score,
    create_visual_sizes,
    create_macro_genres,
    create_region_nodes,
    create_landmark_artists
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
    / "music_map_v1.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIG
# =========================================================

MIN_LISTENERS = 1000

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# LOAD DATA
# =========================================================

print("\n[1/7] Loading music dataset...")

music = load_raw_data()


# =========================================================
# FILTER
# =========================================================

print("\n[2/7] Filtering artists...")

music = filter_artists(
    music,
    min_listeners=MIN_LISTENERS
)

print(
    f"Artists remaining: {len(music):,}"
)


# =========================================================
# TAG PROCESSING
# =========================================================

print("\n[3/7] Processing tags...")

music = create_tags_text(music)


# =========================================================
# POPULARITY
# =========================================================

music = compute_popularity_score(
    music
)

music = create_visual_sizes(
    music,
    strength=VISUAL_SIZE_STRENGTH
)

music = create_macro_genres(
    music
)


# =========================================================
# EMBEDDINGS
# =========================================================

print("\n[4/7] Building TF-IDF embeddings...")

tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(music)
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

music["umap_x"] = embedding[:, 0]

music["umap_y"] = embedding[:, 1]


# =========================================================
# CLUSTERING
# =========================================================

print("\n[6/7] Computing clusters...")

music = compute_clusters(
    music
)

music = create_region_labels(
    music
)


# =========================================================
# LOD DATASETS
# =========================================================

print("\n[7/7] Creating LOD layers...")

regions = create_region_nodes(
    music
)

landmarks = create_landmark_artists(
    music,
    min_listeners=50000
)


# =========================================================
# EXPORT CSV
# =========================================================

music.to_csv(
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
    / "music"
)

JSON_DIR.mkdir(
    parents=True,
    exist_ok=True
)

music.to_json(
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

print("\n✅ Music Atlas complete!")