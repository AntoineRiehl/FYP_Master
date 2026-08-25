# src/pipelines/build_movies_music_atlas.py

import pandas as pd

from pathlib import Path


# =========================================================
# MOVIES
# =========================================================

from src.domains.movies.load_data import (
    load_raw_data as load_movie_data
)

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    compute_weighted_rating,
    concatenate_tags
)


# =========================================================
# MUSIC
# =========================================================

from src.domains.music.load_data import (
    load_raw_data as load_music_data
)

from src.domains.music.feature_engineering import (
    filter_artists,
    create_tags_text,
    compute_popularity_score
)


# =========================================================
# CROSS-DOMAIN
# =========================================================

from src.atlas.cross_domain.combine_data import (
    combine_domain_data
)

from src.atlas.cross_domain.semantic_text import (
    create_semantic_text
)

from src.atlas.cross_domain.domain_vocabulary import (
    get_domain_vocabulary
)

from src.atlas.cross_domain.review_fusion import (
    fuse_cross_domain_semantics
)


# =========================================================
# VISUALS
# =========================================================

from src.atlas.visual.size_normalization import (
    normalize_visual_sizes
)


# =========================================================
# EMBEDDINGS / PROJECTION
# =========================================================

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
)

from src.atlas.embeddings.dimensionality_reduction import (
    get_umap_projection
)


# =========================================================
# CLUSTERING
# =========================================================

from src.atlas.clustering.clustering import (
    compute_clusters
)

from src.atlas.clustering.region_labels import (
    create_region_labels
)


# =========================================================
# ATLAS SCHEMA / EXPORT
# =========================================================

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
    / "movies_music_atlas_v1.csv"
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Frontend bundle
# ---------------------------------------------------------

JSON_DIR = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "movies_music"
)


# =========================================================
# REVIEW METADATA PATHS
# =========================================================

MOVIE_REVIEW_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "movie_review_summary.csv"
)


MUSIC_REVIEW_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "reviews"
    / "music_review_summary.csv"
)


# =========================================================
# REVIEW EMBEDDING PATHS
# =========================================================

MOVIE_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "movies"
    / "movie_review_embeddings.npz"
)


MUSIC_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "music"
    / "music_review_embeddings.npz"
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURE_CONFIG = FeatureConfig(

    name="movies_music_semantic_reviews",

    use_tags=True,

    use_categories=True,

    use_reviews=True,

    use_metadata=True,

    use_statistics=True

)


# =========================================================
# CONFIGURATION
# =========================================================

MIN_LISTENERS = 1000


DOMAINS = [
    "movies",
    "music"
]


# =========================================================
# SEMANTIC CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# The cross-domain TF-IDF representation is built from the
# shared domain-neutral semantic_text field.
#
# Before review fusion it is reduced to 256 dimensions using
# TruncatedSVD.
# ---------------------------------------------------------

TFIDF_COMPONENTS = 256


# ---------------------------------------------------------
# Reviews contribute 50% of the semantic representation.
#
# This follows the configuration selected for the Movie and
# Music mono-domain atlases.
#
# Method A deliberately keeps this simple and consistent
# rather than introducing another cross-domain-specific
# hyperparameter.
# ---------------------------------------------------------

REVIEW_SHARE = 0.50


RANDOM_STATE = 42


# =========================================================
# VISUAL CONFIGURATION
# =========================================================

VISUAL_SIZE_STRENGTH = 1.8


# =========================================================
# VALIDATE ENRICHMENT FILES
# =========================================================

required_files = [

    MOVIE_REVIEW_SUMMARY,

    MUSIC_REVIEW_SUMMARY,

    MOVIE_REVIEW_EMBEDDINGS,

    MUSIC_REVIEW_EMBEDDINGS,

]


missing_files = [

    path

    for path in required_files

    if not path.exists()

]


if missing_files:

    formatted = "\n".join(

        f"  - {path}"

        for path in missing_files

    )


    raise FileNotFoundError(

        "Required review enrichment files were not found:\n\n"
        f"{formatted}"

    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "\n[1/10] Loading raw datasets..."
)


movies, ratings, tags = (
    load_movie_data()
)


music = (
    load_music_data()
)


# =========================================================
# PREPROCESS MOVIES
# =========================================================

print(
    "\n[2/10] Preprocessing movies..."
)


# ---------------------------------------------------------
# Existing MovieLens statistics
# ---------------------------------------------------------

movie_stats = compute_movie_stats(
    ratings
)


movies = merge_movies(
    movie_stats,
    movies
)


movies = compute_weighted_rating(
    movies
)


# ---------------------------------------------------------
# Movie tags
# ---------------------------------------------------------

movie_tags = concatenate_tags(
    tags
)


movies = movies.merge(

    movie_tags,

    on="movieId",

    how="left"

)


movies["tags_text"] = (

    movies["tags_text"]
    .fillna("")

)


# =========================================================
# MOVIE REVIEW METADATA
# =========================================================

movie_review_summary = pd.read_csv(

    MOVIE_REVIEW_SUMMARY,

    usecols=[
        "movieId",
        "review_count"
    ]

)


# ---------------------------------------------------------
# Standardize IDs defensively while preserving a numerical
# movieId.
# ---------------------------------------------------------

movies["movieId"] = pd.to_numeric(

    movies["movieId"],

    errors="raise"

)


movie_review_summary["movieId"] = pd.to_numeric(

    movie_review_summary["movieId"],

    errors="raise"

)


movies = movies.merge(

    movie_review_summary,

    on="movieId",

    how="left"

)


movies["review_count"] = (

    movies["review_count"]
    .fillna(0)
    .astype(int)

)


# =========================================================
# MOVIE POPULARITY / VISUAL SIZE
# =========================================================

# ---------------------------------------------------------
# Cross-domain visual size should NOT compare raw MovieLens
# rating counts directly with Last.fm listener counts.
#
# We therefore retain the Movie domain's own popularity
# measure and normalize its visual size BEFORE combination.
#
# rating_count is used here as the Movie popularity measure.
# ---------------------------------------------------------

movies["popularity_score"] = pd.to_numeric(

    movies["rating_count"],

    errors="coerce"

).fillna(0)


movies = normalize_visual_sizes(

    movies,

    strength=
        VISUAL_SIZE_STRENGTH

)


print(
    f"Movies remaining: "
    f"{len(movies):,}"
)


print(
    f"Movies with review metadata: "
    f"{int((movies['review_count'] > 0).sum()):,}"
)


print(
    "Movie visual sizes: "
    f"min={movies['visual_size'].min():.3f}, "
    f"median={movies['visual_size'].median():.3f}, "
    f"max={movies['visual_size'].max():.3f}"
)


# =========================================================
# PREPROCESS MUSIC
# =========================================================

print(
    "\n[3/10] Preprocessing music..."
)


music = filter_artists(

    music,

    min_listeners=
        MIN_LISTENERS

)


print(
    f"Artists remaining: "
    f"{len(music):,}"
)


music = create_tags_text(
    music
)


music["tags_text"] = (

    music["tags_text"]
    .fillna("")

)


music = compute_popularity_score(
    music
)


# =========================================================
# MUSIC REVIEW METADATA
# =========================================================

music_review_summary = pd.read_csv(

    MUSIC_REVIEW_SUMMARY,

    usecols=[
        "artist_mbid",
        "review_count"
    ]

)


music["mbid"] = (

    music["mbid"]
    .astype(str)
    .str.strip()

)


music_review_summary["artist_mbid"] = (

    music_review_summary["artist_mbid"]
    .astype(str)
    .str.strip()

)


music = music.merge(

    music_review_summary,

    left_on=
        "mbid",

    right_on=
        "artist_mbid",

    how=
        "left"

)


music = music.drop(

    columns=[
        "artist_mbid"
    ],

    errors="ignore"

)


music["review_count"] = (

    music["review_count"]
    .fillna(0)
    .astype(int)

)


# =========================================================
# MUSIC VISUAL SIZE
# =========================================================

# ---------------------------------------------------------
# Music popularity_score is computed using the Music
# domain's own popularity information.
#
# Visual size is normalized BEFORE combining domains, so
# Last.fm listener counts are never directly compared with
# MovieLens rating counts.
# ---------------------------------------------------------

music = normalize_visual_sizes(

    music,

    strength=
        VISUAL_SIZE_STRENGTH

)


print(
    f"Artists with review metadata: "
    f"{int((music['review_count'] > 0).sum()):,}"
)


print(
    "Music visual sizes: "
    f"min={music['visual_size'].min():.3f}, "
    f"median={music['visual_size'].median():.3f}, "
    f"max={music['visual_size'].max():.3f}"
)


# =========================================================
# COMBINE DOMAINS
# =========================================================

print(
    "\n[4/10] Combining domain datasets..."
)


combined = combine_domain_data(

    {

        "movies":
            movies,

        "music":
            music

    }

)


print(
    f"Combined items: "
    f"{len(combined):,}"
)


print(
    "\nItems by domain:"
)


print(

    combined[
        "domain"
    ]
    .value_counts()

)


# =========================================================
# VERIFY CROSS-DOMAIN VISUAL SIZE NORMALIZATION
# =========================================================

print()

print(
    "Visual-size distribution by domain:"
)


for domain in DOMAINS:

    domain_sizes = (

        combined.loc[

            combined["domain"]
            ==
            domain,

            "visual_size"

        ]

    )


    print(

        f"  {domain:<10} "
        f"min={domain_sizes.min():.3f} | "
        f"median={domain_sizes.median():.3f} | "
        f"max={domain_sizes.max():.3f}"

    )


# =========================================================
# DOMAIN-NEUTRAL SEMANTIC TEXT
# =========================================================

print(
    "\n[5/10] Building domain-neutral semantic text..."
)


domain_vocabulary = (
    get_domain_vocabulary(

        DOMAINS

    )
)


combined = create_semantic_text(

    combined,

    domain_vocabulary=
        domain_vocabulary

)


# ---------------------------------------------------------
# Diagnostic information
# ---------------------------------------------------------

semantic_text_available = int(

    (
        combined[
            "semantic_text"
        ]
        .str.strip()
        !=
        ""
    ).sum()

)


print(
    f"Items with semantic text: "
    f"{semantic_text_available:,} / "
    f"{len(combined):,}"
)


# =========================================================
# SHARED CROSS-DOMAIN TF-IDF
# =========================================================

print(
    "\n[6/10] Building shared cross-domain TF-IDF..."
)


tfidf_matrix, vectorizer = (
    get_tfidf_embeddings(

        combined,

        text_column=
            "semantic_text",

        model_name=
            "movies_music_semantic"

    )
)


print(
    f"Shared TF-IDF shape: "
    f"{tfidf_matrix.shape}"
)


# =========================================================
# CROSS-DOMAIN REVIEW FUSION
# =========================================================

print(
    "\n[7/10] Fusing TF-IDF and review semantics..."
)


(
    semantic_matrix,
    fusion_info,
    reviews_used

) = fuse_cross_domain_semantics(

    # -----------------------------------------------------
    # Shared cross-domain TF-IDF representation
    # -----------------------------------------------------

    tfidf_matrix=
        tfidf_matrix,


    # -----------------------------------------------------
    # Contains standardized:
    #
    #     domain
    #     source_id
    #
    # used by the fusion helper for alignment.
    # -----------------------------------------------------

    combined_df=
        combined,


    # -----------------------------------------------------
    # Existing domain-specific SBERT review embeddings.
    #
    # Both were generated using:
    #
    # sentence-transformers/all-MiniLM-L6-v2
    #
    # and therefore inhabit the same 384D embedding space.
    # -----------------------------------------------------

    review_embedding_paths={

        "movies":
            MOVIE_REVIEW_EMBEDDINGS,

        "music":
            MUSIC_REVIEW_EMBEDDINGS,

    },


    # -----------------------------------------------------
    # Shared semantic configuration
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
# REVIEW ENRICHMENT METADATA
# =========================================================

# ---------------------------------------------------------
# review_count
#
#     Total number of matched/prepared reviews available
#     for the entity.
#
#
# reviews_used_for_embedding
#
#     Number actually used to construct its SBERT review
#     vector.
#
#     Maximum = 50.
#
#
# has_review_embedding
#
#     Whether the review block contributes to this entity's
#     fused semantic representation.
# ---------------------------------------------------------

combined[
    "reviews_used_for_embedding"
] = reviews_used


combined[
    "has_review_embedding"
] = (

    reviews_used > 0

)


print()

print(
    "Review coverage in combined atlas:"
)


for domain in DOMAINS:

    domain_mask = (

        combined["domain"]
        ==
        domain

    )


    total = int(
        domain_mask.sum()
    )


    with_reviews = int(

        (
            combined.loc[
                domain_mask,
                "has_review_embedding"
            ]
            ==
            True
        ).sum()

    )


    percentage = (

        (
            with_reviews
            /
            total
            *
            100
        )

        if total > 0

        else 0.0

    )


    print(

        f"  {domain:<10} "
        f"{with_reviews:>7,} / "
        f"{total:>7,} "
        f"({percentage:6.2f}%)"

    )


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
    f"  Entities with reviews:     "
    f"{fusion_info['entities_with_reviews']:,}"
)


print(
    f"  Entities without reviews:  "
    f"{fusion_info['entities_without_reviews']:,}"
)


print(
    f"  TF-IDF / Review share:     "
    f"{fusion_info['tfidf_share']:.2f} / "
    f"{fusion_info['review_share']:.2f}"
)


# =========================================================
# CROSS-DOMAIN UMAP
# =========================================================

print(
    "\n[8/10] Computing cross-domain UMAP projection..."
)


# ---------------------------------------------------------
# IMPORTANT:
#
# UMAP is recomputed FROM SCRATCH using the single fused
# high-dimensional representation.
#
# Mono-domain UMAP coordinates are never combined or reused.
# ---------------------------------------------------------

embedding, umap_model = (
    get_umap_projection(

        semantic_matrix

    )
)


combined["umap_x"] = (
    embedding[:, 0]
)


combined["umap_y"] = (
    embedding[:, 1]
)


# =========================================================
# CLUSTERING
# =========================================================

print(
    "\n[9/10] Computing cross-domain clusters..."
)


combined = compute_clusters(
    combined
)


combined = create_region_labels(
    combined
)


# =========================================================
# EXPORT ANALYSIS DATASET
# =========================================================

print(
    "\nSaving analysis dataframe..."
)


combined.to_csv(

    OUTPUT_PATH,

    index=False

)


# =========================================================
# BUILD ATLAS BUNDLE
# =========================================================

print(
    "\n[10/10] Building AtlasBundle..."
)


bundle = build_bundle(

    df=
        combined,

    domain=
        "movies_music",

    feature_config=
        FEATURE_CONFIG,

    metadata={

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        "pipeline":
            "movies_music_atlas",

        "method":
            "general_semantic_fusion",

        "domains":
            DOMAINS,


        # -------------------------------------------------
        # Original semantic methodology
        # -------------------------------------------------

        "semantic_text":
            True,

        "domain_vocabulary_filtering":
            True,

        "tfidf_model":
            "movies_music_semantic",

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
        # Review semantics
        # -------------------------------------------------

        "embedding":
            "TF-IDF + Sentence-BERT reviews",

        "review_embedding_model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "review_embedding_dimensions":
            int(
                fusion_info[
                    "review_components"
                ]
            ),

        "review_share":
            REVIEW_SHARE,

        "tfidf_share":
            1.0 - REVIEW_SHARE,

        "max_reviews_per_entity":
            50,

        "review_aggregation":
            "mean pooling",

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

        "items_without_review_embeddings":
            int(
                fusion_info[
                    "entities_without_reviews"
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
        # Visual size
        # -------------------------------------------------

        "visual_size_strength":
            VISUAL_SIZE_STRENGTH,

        "visual_size_normalization":
            "within_domain_before_combination",

        "movie_popularity_measure":
            "MovieLens rating_count",

        "music_popularity_measure":
            "music popularity_score",


        # -------------------------------------------------
        # Population
        # -------------------------------------------------

        "items":
            len(
                combined
            ),

        "movies":
            int(
                (
                    combined[
                        "domain"
                    ]
                    ==
                    "movies"
                ).sum()
            ),

        "music":
            int(
                (
                    combined[
                        "domain"
                    ]
                    ==
                    "music"
                ).sum()
            ),


        # -------------------------------------------------
        # Clustering
        # -------------------------------------------------

        "clusters":
            int(
                combined[
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
    "=" * 65
)

print(
    "✅ MOVIES + MUSIC GENERAL SEMANTIC ATLAS COMPLETE"
)

print(
    "=" * 65
)

print()


print(
    f"Total items: "
    f"{len(combined):,}"
)


print(
    f"Movies: "
    f"{int((combined['domain'] == 'movies').sum()):,}"
)


print(
    f"Music: "
    f"{int((combined['domain'] == 'music').sum()):,}"
)


print(
    f"Items with review embeddings: "
    f"{int(combined['has_review_embedding'].sum()):,}"
)


print(
    f"Items without review embeddings: "
    f"{int((~combined['has_review_embedding']).sum()):,}"
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
    f"{combined['cluster'].nunique():,}"
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