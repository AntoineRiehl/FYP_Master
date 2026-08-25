# src/pipelines/build_restaurants_map.py

import pandas as pd

from pathlib import Path


from src.domains.restaurants.load_data import (
    load_raw_data
)

from src.domains.restaurants.feature_engineering import (
    filter_restaurants,
    create_tags_text,
    compute_popularity_score,
    create_macro_genres
)

from src.atlas.visual.size_normalization import (
    normalize_visual_sizes
)

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
)

from src.atlas.embeddings.semantic_fusion import (
    fuse_semantic_embeddings
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

from src.atlas.schema.feature_config import (
    FeatureConfig
)

from src.atlas.builders.build_bundle import (
    build_bundle
)

from src.atlas.export.export_bundle import (
    export_bundle
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Analysis dataframe
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Sentence-BERT review embeddings
# ---------------------------------------------------------

RESTAURANT_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "restaurants"
    / "restaurant_review_embeddings.npz"
)


# ---------------------------------------------------------
# Restaurant review summary metadata
# ---------------------------------------------------------

RESTAURANT_REVIEW_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "restaurant_review_summary.csv"
)


# ---------------------------------------------------------
# Frontend bundle
# ---------------------------------------------------------

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "restaurants"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="restaurants_categories_tips_reviews",

    use_tags=True,

    use_categories=True,

    use_reviews=True,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# DATA CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Existing Yelp business filtering criterion.
#
# This is unrelated to the MIN_REVIEWS value used during
# fusion-weight evaluation.
#
# The current Restaurant atlas contains businesses having
# at least 20 Yelp reviews.
# ---------------------------------------------------------

MIN_REVIEWS = 20


# =========================================================
# SEMANTIC CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Existing TF-IDF representation
#
# Restaurant semantics currently come from:
#
#     Yelp categories
#         +
#     Yelp tips
#
# These are combined into tags_text and represented using
# TF-IDF.
#
# Before semantic fusion, the TF-IDF matrix is reduced with
# TruncatedSVD.
# ---------------------------------------------------------

TFIDF_COMPONENTS = 256


# ---------------------------------------------------------
# Review contribution
#
# Evaluated using:
#
#   src/domains/restaurants/evaluate_fusion_weights.py
#
#
# Diagnostic results:
#
#   Review share 0.00 -> 0.8519
#   Review share 0.20 -> 0.8536
#   Review share 0.35 -> 0.8543
#   Review share 0.50 -> 0.8541
#   Review share 0.65 -> 0.8523
#   Review share 0.80 -> 0.8467
#   Review share 1.00 -> 0.8110
#
#
# The numerical maximum occurred at 0.35.
#
# However:
#
#     0.35 -> 0.8543
#     0.50 -> 0.8541
#
# The difference is negligible.
#
# A common 0.50 / 0.50 configuration is therefore used
# across Movies, Music and Restaurants for methodological
# consistency and interpretability while remaining inside
# the high-performing Restaurant fusion region.
# ---------------------------------------------------------

REVIEW_SHARE = 0.50


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_STATE = 42


# =========================================================
# VISUAL CONFIGURATION
# =========================================================

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# VALIDATE REVIEW FILES
# =========================================================

if not RESTAURANT_REVIEW_EMBEDDINGS.exists():

    raise FileNotFoundError(

        "Restaurant review embeddings were not found:\n"
        f"{RESTAURANT_REVIEW_EMBEDDINGS}\n\n"

        "Run:\n"
        "python -m src.domains.restaurants."
        "embed_restaurant_reviews\n"

        "before building the enriched Restaurant atlas."

    )


if not RESTAURANT_REVIEW_SUMMARY.exists():

    raise FileNotFoundError(

        "Restaurant review summary was not found:\n"
        f"{RESTAURANT_REVIEW_SUMMARY}\n\n"

        "Run:\n"
        "python -m src.domains.restaurants."
        "prepare_restaurant_reviews\n"

        "before building the enriched Restaurant atlas."

    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "\n[1/9] Loading restaurant dataset..."
)


businesses, reviews, tips = (
    load_raw_data()
)


# =========================================================
# PREPROCESSING
# =========================================================

print(
    "\n[2/9] Filtering restaurants..."
)


restaurants = filter_restaurants(

    businesses,

    min_reviews=
        MIN_REVIEWS

)


print(
    f"Restaurants remaining: "
    f"{len(restaurants):,}"
)


# =========================================================
# TEXT PROCESSING
# =========================================================

print(
    "\n[3/9] Processing categories + tips..."
)


restaurants = create_tags_text(

    restaurants,

    tips

)


restaurants["tags_text"] = (

    restaurants["tags_text"]
    .fillna("")

)


# =========================================================
# REVIEW METADATA
# =========================================================

print(
    "\nPreparing restaurant review metadata..."
)


# ---------------------------------------------------------
# Yelp business.json already contains review_count.
#
# Unlike Movies and Music, we therefore do not need to
# merge the generated review summary just to obtain total
# review availability.
#
# Standardize the canonical Yelp identifier before
# Sentence-BERT embedding alignment.
# ---------------------------------------------------------

restaurants["business_id"] = (

    restaurants["business_id"]
    .astype(str)
    .str.strip()

)


restaurants["review_count"] = (

    pd.to_numeric(
        restaurants["review_count"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)

)


print(
    f"Restaurants with Yelp reviews: "
    f"{int((restaurants['review_count'] > 0).sum()):,}"
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print(
    "\nCreating atlas features..."
)


restaurants = compute_popularity_score(
    restaurants
)


restaurants = normalize_visual_sizes(

    restaurants,

    strength=
        VISUAL_SIZE_STRENGTH

)


restaurants = create_macro_genres(
    restaurants
)


# =========================================================
# TF-IDF
# =========================================================

print(
    "\n[4/9] Building TF-IDF embeddings..."
)


tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(

        restaurants,

        text_column=
            "tags_text",

        model_name=
            "restaurants_tags"

    )
)


# =========================================================
# SEMANTIC FUSION
# =========================================================

print(
    "\n[5/9] Fusing TF-IDF and review semantics..."
)


(
    semantic_matrix,
    fusion_info,
    review_counts

) = fuse_semantic_embeddings(

    # -----------------------------------------------------
    # Existing categories + tips representation
    # -----------------------------------------------------

    tfidf_matrix=
        tfidf_matrix,


    # -----------------------------------------------------
    # Canonical Yelp entity identifier
    # -----------------------------------------------------

    entity_ids=
        restaurants[
            "business_id"
        ].to_numpy(),


    # -----------------------------------------------------
    # Sentence-BERT full-review embeddings
    # -----------------------------------------------------

    review_embeddings_path=
        RESTAURANT_REVIEW_EMBEDDINGS,


    # -----------------------------------------------------
    # Fusion configuration
    # -----------------------------------------------------

    review_share=
        REVIEW_SHARE,

    tfidf_components=
        TFIDF_COMPONENTS,

    random_state=
        RANDOM_STATE,

    return_info=
        True

)


# =========================================================
# REVIEW AVAILABILITY METADATA
# =========================================================

# ---------------------------------------------------------
# review_count:
#
#     Total number of Yelp reviews available for the
#     restaurant.
#
#
# reviews_used_for_embedding:
#
#     Number of reviews actually used when constructing the
#     Sentence-BERT restaurant vector.
#
#     This is capped at 50.
#
#
# has_review_embedding:
#
#     Whether review-derived semantic information
#     contributed to the fused representation.
#
# For this Restaurant dataset all 33,941 atlas restaurants
# are expected to have a review embedding.
# ---------------------------------------------------------

restaurants[
    "reviews_used_for_embedding"
] = review_counts


restaurants[
    "has_review_embedding"
] = (

    review_counts > 0

)


# =========================================================
# FUSION INFORMATION
# =========================================================

print()

print(
    "Semantic fusion summary:"
)


print(
    f"  TF-IDF reduced dimensions: "
    f"{fusion_info['tfidf_components']}"
)


print(
    f"  Review dimensions:         "
    f"{fusion_info['review_components']}"
)


print(
    f"  Final fused dimensions:    "
    f"{fusion_info['fused_dimensions']}"
)


print(
    f"  Restaurants with reviews:  "
    f"{fusion_info['entities_with_reviews']:,}"
)


print(
    f"  Review share:              "
    f"{REVIEW_SHARE:.2f}"
)


print(
    f"  TF-IDF share:              "
    f"{1.0 - REVIEW_SHARE:.2f}"
)


# =========================================================
# UMAP
# =========================================================

print(

    "\n[6/9] Computing UMAP projection "
    "from fused semantic representation..."

)


embedding, umap_model = (
    get_umap_projection(

        semantic_matrix

    )
)


restaurants["umap_x"] = (
    embedding[:, 0]
)


restaurants["umap_y"] = (
    embedding[:, 1]
)


# =========================================================
# CLUSTERING
# =========================================================

print(
    "\n[7/9] Computing clusters..."
)


restaurants = compute_clusters(
    restaurants
)


restaurants = create_region_labels(
    restaurants
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print(
    "\n[8/9] Saving analysis dataframe..."
)


restaurants.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print(
    "\n[9/9] Building AtlasBundle..."
)


bundle = build_bundle(

    df=
        restaurants,

    domain=
        "restaurants",

    feature_config=
        FEATURE_CONFIG,

    metadata={

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        "pipeline":
            "restaurant_map",


        "minimum_reviews":
            MIN_REVIEWS,


        # -------------------------------------------------
        # Semantic representation
        # -------------------------------------------------

        "embedding":
            "TF-IDF + Sentence-BERT reviews",


        "tfidf_model":
            "restaurants_tags",


        "tfidf_sources":
            "Yelp categories + Yelp tips",


        "tfidf_original_dimensions":
            int(
                tfidf_matrix.shape[1]
            ),


        "tfidf_svd_components":
            int(
                fusion_info[
                    "tfidf_components"
                ]
            ),


        "tfidf_svd_explained_variance":
            float(
                fusion_info[
                    "svd_explained_variance"
                ]
            ),


        # -------------------------------------------------
        # Review representation
        # -------------------------------------------------

        "review_source":
            "Yelp full reviews",


        "review_embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",


        "review_embedding_dimensions":
            int(
                fusion_info[
                    "review_components"
                ]
            ),


        "max_reviews_per_restaurant":
            50,


        "review_aggregation":
            "mean pooling",


        "review_embedding_normalization":
            "L2",


        # -------------------------------------------------
        # Fusion
        # -------------------------------------------------

        "review_share":
            REVIEW_SHARE,


        "tfidf_share":
            1.0 - REVIEW_SHARE,


        "fused_dimensions":
            int(
                fusion_info[
                    "fused_dimensions"
                ]
            ),


        "items_with_review_embeddings":
            int(
                fusion_info[
                    "entities_with_reviews"
                ]
            ),


        # -------------------------------------------------
        # Projection
        # -------------------------------------------------

        "projection":
            "UMAP",


        "umap_metric":
            "cosine",


        "umap_n_neighbors":
            15,


        "umap_min_dist":
            0.1,


        # -------------------------------------------------
        # Visual configuration
        # -------------------------------------------------

        "visual_size_strength":
            VISUAL_SIZE_STRENGTH,


        # -------------------------------------------------
        # Atlas information
        # -------------------------------------------------

        "items":
            len(
                restaurants
            ),


        "clusters":
            int(
                restaurants[
                    "cluster"
                ]
                .nunique()
            )

    }

)


# =========================================================
# EXPORT FRONTEND DATA
# =========================================================

print(
    "\nExporting AtlasBundle..."
)


export_bundle(

    bundle,

    JSON_DIR

)


# =========================================================
# COMPLETE
# =========================================================

print()

print(
    "=" * 60
)

print(
    "✅ ENRICHED RESTAURANT ATLAS COMPLETE"
)

print(
    "=" * 60
)

print()


print(
    f"Restaurants: "
    f"{len(restaurants):,}"
)


print(
    f"Restaurants with total review metadata: "
    f"{int((restaurants['review_count'] > 0).sum()):,}"
)


print(
    f"Restaurants with review embeddings: "
    f"{int((review_counts > 0).sum()):,}"
)


print(
    f"TF-IDF / Review semantic share: "
    f"{1.0 - REVIEW_SHARE:.2f} / "
    f"{REVIEW_SHARE:.2f}"
)


print(
    f"Fused semantic dimensions: "
    f"{semantic_matrix.shape[1]}"
)


print(
    f"Clusters: "
    f"{restaurants['cluster'].nunique():,}"
)


print()

print(
    "Analysis dataframe:"
)

print(
    OUTPUT_PATH
)


print()

print(
    "Frontend data:"
)

print(
    JSON_DIR
)


print()