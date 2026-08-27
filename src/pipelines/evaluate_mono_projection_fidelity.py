# src/pipelines/evaluate_mono_projection_fidelity.py

# ============================================================
# MONO-DOMAIN PROJECTION FIDELITY EVALUATION
# ============================================================
#
# Purpose
# -------
#
# Evaluate how faithfully the final 2D mono-domain atlases
# preserve the neighbourhood structure of the frozen
# high-dimensional semantic representation.
#
#
# ATLASES
# -------
#
# Movies
# Music
# Restaurants
#
#
# FROZEN MONO-DOMAIN REPRESENTATION
# ---------------------------------
#
# Semantic text
#       ↓
# TF-IDF
#       max_features = 5000
#       English stop words
#       ↓
# TruncatedSVD
#       256 dimensions
#       ↓
# L2 normalization
#
# Review embeddings
#       384D MiniLM
#       ↓
# L2 normalization
#
# Both modalities are fused as:
#
#     [sqrt(0.5) * TFIDF_SVD |
#      sqrt(0.5) * REVIEW]
#
# followed by L2 normalization.
#
#
# Missing review embeddings use a zero review block,
# reproducing the frozen mono-domain Method A behaviour.
#
#
# METRICS
# -------
#
# trustworthiness
#
#     Measures whether the 2D UMAP introduces neighbours that
#     were not genuinely close in the high-dimensional source
#     representation.
#
# kNN preservation
#
#     Fraction of exact high-dimensional k-neighbours retained
#     among the k nearest neighbours in 2D.
#
#
# IMPORTANT
# ---------
#
# This script:
#
#     - DOES reconstruct TF-IDF/SVD
#     - DOES load existing review embeddings
#
# but:
#
#     - DOES NOT recompute review embeddings
#     - DOES NOT recompute UMAP
#     - DOES NOT recompute clustering
#     - DOES NOT modify any atlas
#
#
# To keep memory use manageable, the TF-IDF/SVD model is fit
# on the complete frozen atlas population, but the final dense
# 640D fused vectors are created only for the deterministic
# evaluation sample.
#
# ============================================================


from pathlib import Path
import re

import numpy as np
import pandas as pd


from scipy.sparse import (
    csr_matrix
)


from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.decomposition import (
    TruncatedSVD
)

from sklearn.preprocessing import (
    normalize
)


from src.evaluation.atlas_metrics import (
    evaluate_projection_quality
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


EMBEDDING_DIR = (
    PROCESSED_DIR
    / "embeddings"
)


OUTPUT_DIR = (
    PROCESSED_DIR
    / "evaluation"
    / "mono_domain"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "mono_projection_fidelity.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TFIDF_MAX_FEATURES = 5_000

SVD_COMPONENTS = 256

REVIEW_SHARE = 0.50

RANDOM_STATE = 42

K_NEIGHBORS = 15


# ------------------------------------------------------------
# Projection metrics operate on a deterministic subset.
#
# 3,000 entities is consistent with our earlier evaluation
# and keeps pairwise neighbourhood calculations practical.
# ------------------------------------------------------------

EVALUATION_SAMPLE_SIZE = 3_000


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

ATLAS_CONFIGS = [

    {

        "id":
            "movies",

        "label":
            "Movies",

        "atlas_path":
            PROCESSED_DIR
            / "movie_map_v1.csv",

        "review_embedding_path":
            EMBEDDING_DIR
            / "movies"
            / "movie_review_embeddings.npz",

        "id_candidates": [

            "source_id",

            "movieId",

            "movie_id"

        ],

        "text_candidates": [

            "tags_text",

            "tags"

        ],

    },


    {

        "id":
            "music",

        "label":
            "Music",

        "atlas_path":
            PROCESSED_DIR
            / "music_map_v1.csv",

        "review_embedding_path":
            EMBEDDING_DIR
            / "music"
            / "music_review_embeddings.npz",

        "id_candidates": [

            "source_id",

            "mbid",

            "artist_mbid"

        ],

        "text_candidates": [

            "tags_text",

            "tags_lastfm",

            "tags_mb"

        ],

    },


    {

        "id":
            "restaurants",

        "label":
            "Restaurants",

        "atlas_path":
            PROCESSED_DIR
            / "restaurant_map_v1.csv",

        "review_embedding_path":
            EMBEDDING_DIR
            / "restaurants"
            / "restaurant_review_embeddings.npz",

        "id_candidates": [

            "source_id",

            "business_id"

        ],

        "text_candidates": [

            "tags_text",

            "categories"

        ],

    },

]


# ============================================================
# PRINTING
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 78
    )

    print(
        title
    )

    print(
        "=" * 78
    )

    print()


# ============================================================
# COLUMN DETECTION
# ============================================================


def first_available_column(
    columns,
    candidates
):

    columns = set(
        columns
    )


    for candidate in candidates:

        if candidate in columns:

            return candidate


    return None


# ============================================================
# ID NORMALIZATION
# ============================================================


def normalize_entity_id(
    value
):
    """
    Normalize heterogeneous IDs into stable strings.

    Examples:

        123
        123.0
        "123"

    all become:

        "123"

    UUIDs remain unchanged.
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    if re.fullmatch(

        r"[+-]?\d+\.0+",

        text

    ):

        return text.split(
            "."
        )[0]


    return text


# ============================================================
# SAMPLE INDICES
# ============================================================


def create_evaluation_sample(

    n_items,

    sample_size=
        EVALUATION_SAMPLE_SIZE,

    random_state=
        RANDOM_STATE

):
    """
    Deterministic uniform sample.
    """

    if n_items <= sample_size:

        return np.arange(

            n_items,

            dtype=np.int64

        )


    rng = np.random.default_rng(
        random_state
    )


    return np.sort(

        rng.choice(

            n_items,

            size=
                sample_size,

            replace=
                False

        )

    )


# ============================================================
# LOAD ATLAS
# ============================================================


def load_atlas(
    config
):

    section(

        f"LOADING {config['label'].upper()} ATLAS"

    )


    path = config[
        "atlas_path"
    ]


    if not path.exists():

        raise FileNotFoundError(

            f"Atlas CSV not found:\n"
            f"{path}"

        )


    header = pd.read_csv(

        path,

        nrows=0

    )


    columns = list(
        header.columns
    )


    # ========================================================
    # REQUIRED FIELDS
    # ========================================================

    id_column = first_available_column(

        columns,

        config[
            "id_candidates"
        ]

    )


    text_column = first_available_column(

        columns,

        config[
            "text_candidates"
        ]

    )


    x_column = first_available_column(

        columns,

        [

            "umap_x",

            "x",

            "position_x"

        ]

    )


    y_column = first_available_column(

        columns,

        [

            "umap_y",

            "y",

            "position_y"

        ]

    )


    if id_column is None:

        raise ValueError(

            f"Could not identify entity ID column for "
            f"{config['label']}.\n\n"

            f"Available columns:\n"
            f"{columns}"

        )


    if text_column is None:

        raise ValueError(

            f"Could not identify semantic text column for "
            f"{config['label']}.\n\n"

            f"Available columns:\n"
            f"{columns}"

        )


    if (

        x_column is None

        or

        y_column is None

    ):

        raise ValueError(

            f"Could not identify UMAP coordinate columns for "
            f"{config['label']}."

        )


    print(
        f"ID column:       "
        f"{id_column}"
    )


    print(
        f"Semantic column: "
        f"{text_column}"
    )


    print(
        f"Coordinates:     "
        f"{x_column}, {y_column}"
    )


    # ========================================================
    # LOAD ONLY REQUIRED DATA
    # ========================================================

    df = pd.read_csv(

        path,

        usecols=[

            id_column,

            text_column,

            x_column,

            y_column

        ],

        low_memory=
            False

    )


    df[
        "entity_id"
    ] = (

        df[
            id_column
        ]
        .apply(
            normalize_entity_id
        )

    )


    df[
        "semantic_text"
    ] = (

        df[
            text_column
        ]
        .fillna("")
        .astype(str)

    )


    df[
        "umap_x"
    ] = pd.to_numeric(

        df[
            x_column
        ],

        errors=
            "coerce"

    )


    df[
        "umap_y"
    ] = pd.to_numeric(

        df[
            y_column
        ],

        errors=
            "coerce"

    )


    valid_coordinates = (

        df[
            [
                "umap_x",
                "umap_y"
            ]
        ]
        .notna()
        .all(
            axis=1
        )

    )


    if not valid_coordinates.all():

        removed = int(

            (
                ~valid_coordinates
            ).sum()

        )


        print(

            f"Removing {removed:,} rows with invalid "
            f"2D coordinates."

        )


        df = df.loc[

            valid_coordinates

        ].reset_index(
            drop=True
        )


    print(
        f"Atlas entities: "
        f"{len(df):,}"
    )


    return df


# ============================================================
# TF-IDF
# ============================================================


def build_tfidf(
    texts
):

    section(
        "FITTING FROZEN TF-IDF REPRESENTATION"
    )


    vectorizer = TfidfVectorizer(

        max_features=
            TFIDF_MAX_FEATURES,

        stop_words=
            "english"

    )


    tfidf_matrix = vectorizer.fit_transform(
        texts
    )


    tfidf_matrix = csr_matrix(

        tfidf_matrix,

        dtype=np.float32

    )


    print(
        f"TF-IDF shape: "
        f"{tfidf_matrix.shape}"
    )


    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )


    return (

        vectorizer,

        tfidf_matrix

    )


# ============================================================
# SVD
# ============================================================


def fit_svd_and_transform_sample(

    tfidf_matrix,

    sample_indices

):
    """
    Fit SVD on the complete frozen population but transform
    only the evaluation sample into the dense representation.
    """

    section(
        "FITTING 256D TRUNCATED SVD"
    )


    max_components = min(

        SVD_COMPONENTS,

        tfidf_matrix.shape[1]
        -
        1,

        tfidf_matrix.shape[0]
        -
        1

    )


    if max_components < 2:

        raise ValueError(

            "TF-IDF matrix is too small for SVD."

        )


    svd = TruncatedSVD(

        n_components=
            max_components,

        random_state=
            RANDOM_STATE

    )


    # --------------------------------------------------------
    # Fit using the full atlas semantic corpus.
    # --------------------------------------------------------

    svd.fit(
        tfidf_matrix
    )


    explained_variance = float(

        svd.explained_variance_ratio_.sum()

    )


    print(
        f"SVD components: "
        f"{max_components}"
    )


    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )


    # --------------------------------------------------------
    # Only the evaluation sample becomes dense.
    # --------------------------------------------------------

    sample_semantics = svd.transform(

        tfidf_matrix[
            sample_indices
        ]

    )


    sample_semantics = np.asarray(

        sample_semantics,

        dtype=np.float32

    )


    sample_semantics = normalize(

        sample_semantics,

        norm=
            "l2",

        axis=
            1

    )


    sample_semantics = np.asarray(

        sample_semantics,

        dtype=np.float32

    )


    print(
        f"Sample semantic representation: "
        f"{sample_semantics.shape}"
    )


    return (

        sample_semantics,

        explained_variance

    )


# ============================================================
# LOAD REVIEW EMBEDDINGS
# ============================================================


def load_review_embeddings(
    path
):

    section(
        "LOADING EXISTING REVIEW EMBEDDINGS"
    )


    if not path.exists():

        raise FileNotFoundError(

            f"Review embedding file not found:\n"
            f"{path}"

        )


    data = np.load(

        path,

        allow_pickle=True

    )


    if "entity_ids" not in data:

        raise ValueError(

            f"'entity_ids' missing from:\n"
            f"{path}\n\n"

            f"Available arrays: "
            f"{list(data.files)}"

        )


    if "embeddings" not in data:

        raise ValueError(

            f"'embeddings' missing from:\n"
            f"{path}"

        )


    entity_ids = np.asarray(

        data[
            "entity_ids"
        ]

    )


    embeddings = np.asarray(

        data[
            "embeddings"
        ],

        dtype=np.float32

    )


    if len(
        entity_ids
    ) != len(
        embeddings
    ):

        raise ValueError(

            "Review embedding IDs and vectors have "
            "different lengths."

        )


    if embeddings.ndim != 2:

        raise ValueError(

            "Review embeddings must be a 2D matrix."

        )


    # --------------------------------------------------------
    # Protect against tiny numerical normalization drift.
    # --------------------------------------------------------

    embeddings = normalize(

        embeddings,

        norm=
            "l2",

        axis=
            1

    )


    embeddings = np.asarray(

        embeddings,

        dtype=np.float32

    )


    normalized_ids = [

        normalize_entity_id(
            value
        )

        for value in entity_ids

    ]


    review_lookup = {

        entity_id:
            index

        for index, entity_id
        in enumerate(
            normalized_ids
        )

    }


    print(
        f"Review entities: "
        f"{len(entity_ids):,}"
    )


    print(
        f"Review dimensions: "
        f"{embeddings.shape[1]}"
    )


    return (

        review_lookup,

        embeddings

    )


# ============================================================
# ALIGN REVIEW SAMPLE
# ============================================================


def align_sample_review_embeddings(

    sample_entity_ids,

    review_lookup,

    review_embeddings

):
    """
    Align the existing 384D review vectors to the sampled
    atlas entities.

    Missing review embeddings receive an all-zero vector.
    """

    review_dimensions = (
        review_embeddings.shape[1]
    )


    aligned = np.zeros(

        (

            len(
                sample_entity_ids
            ),

            review_dimensions

        ),

        dtype=np.float32

    )


    has_review = np.zeros(

        len(
            sample_entity_ids
        ),

        dtype=bool

    )


    for sample_index, entity_id in enumerate(

        sample_entity_ids

    ):


        review_index = review_lookup.get(
            entity_id
        )


        if review_index is None:

            continue


        aligned[
            sample_index
        ] = review_embeddings[
            review_index
        ]


        has_review[
            sample_index
        ] = True


    print(
        f"Sample entities with review embeddings: "
        f"{has_review.sum():,} / {len(has_review):,}"
    )


    print(

        f"Sample review coverage: "
        f"{has_review.mean() * 100:.2f}%"

    )


    return (

        aligned,

        has_review

    )


# ============================================================
# FROZEN BLOCK FUSION
# ============================================================


def fuse_semantic_blocks(

    semantic_vectors,

    review_vectors,

    review_share=
        REVIEW_SHARE

):
    """
    Reproduce the frozen mono-domain block fusion.

    When both blocks are unit-normalized:

        [sqrt(1-r) * semantic |
         sqrt(r)   * review]

    makes cosine similarity approximately:

        (1-r) * semantic_similarity
        +
        r * review_similarity

    for entities with both modalities.

    Missing review vectors remain zero, reproducing the
    known Method A missing-review geometry.
    """

    if (

        len(
            semantic_vectors
        )

        !=

        len(
            review_vectors
        )

    ):

        raise ValueError(

            "Semantic and review vectors are not aligned."

        )


    semantic_weight = np.sqrt(

        1.0
        -
        review_share

    )


    review_weight = np.sqrt(
        review_share
    )


    fused = np.concatenate(

        [

            semantic_weight
            *
            semantic_vectors,

            review_weight
            *
            review_vectors

        ],

        axis=1

    )


    fused = normalize(

        fused,

        norm=
            "l2",

        axis=
            1

    )


    fused = np.asarray(

        fused,

        dtype=np.float32

    )


    return fused


# ============================================================
# EVALUATE ONE DOMAIN
# ============================================================


def evaluate_domain(
    config
):

    section(

        f"EVALUATING PROJECTION FIDELITY: "
        f"{config['label']}"

    )


    # ========================================================
    # 1. LOAD FROZEN ATLAS
    # ========================================================

    df = load_atlas(
        config
    )


    # ========================================================
    # 2. DETERMINISTIC SAMPLE
    # ========================================================

    sample_indices = (
        create_evaluation_sample(

            n_items=
                len(
                    df
                ),

            sample_size=
                EVALUATION_SAMPLE_SIZE,

            random_state=
                RANDOM_STATE

        )
    )


    print()


    print(
        f"Evaluation sample: "
        f"{len(sample_indices):,}"
    )


    sample_entity_ids = (

        df.iloc[
            sample_indices
        ][
            "entity_id"
        ]
        .tolist()

    )


    low_dimensional_sample = (

        df.iloc[
            sample_indices
        ][
            [
                "umap_x",
                "umap_y"
            ]
        ]
        .to_numpy(
            dtype=np.float32
        )

    )


    # ========================================================
    # 3. TF-IDF
    # ========================================================

    _, tfidf_matrix = (
        build_tfidf(

            df[
                "semantic_text"
            ]

        )
    )


    # ========================================================
    # 4. SVD
    # ========================================================

    (

        semantic_sample,

        explained_variance

    ) = fit_svd_and_transform_sample(

        tfidf_matrix=
            tfidf_matrix,

        sample_indices=
            sample_indices

    )


    # --------------------------------------------------------
    # TF-IDF is no longer required.
    # --------------------------------------------------------

    del tfidf_matrix


    # ========================================================
    # 5. REVIEW EMBEDDINGS
    # ========================================================

    (

        review_lookup,

        review_embeddings

    ) = load_review_embeddings(

        config[
            "review_embedding_path"
        ]

    )


    (

        review_sample,

        has_review

    ) = align_sample_review_embeddings(

        sample_entity_ids=
            sample_entity_ids,

        review_lookup=
            review_lookup,

        review_embeddings=
            review_embeddings

    )


    # --------------------------------------------------------
    # Full review matrix no longer needed after sample align.
    # --------------------------------------------------------

    del review_lookup

    del review_embeddings


    # ========================================================
    # 6. RECONSTRUCT FROZEN 640D FUSION
    # ========================================================

    section(
        "RECONSTRUCTING FROZEN FUSED REPRESENTATION"
    )


    fused_sample = fuse_semantic_blocks(

        semantic_vectors=
            semantic_sample,

        review_vectors=
            review_sample,

        review_share=
            REVIEW_SHARE

    )


    print(
        f"Fused sample shape: "
        f"{fused_sample.shape}"
    )


    # ========================================================
    # 7. PROJECTION METRICS
    # ========================================================

    section(
        "COMPUTING PROJECTION METRICS"
    )


    # --------------------------------------------------------
    # The matrix is already exactly our chosen evaluation
    # sample, so passing its complete size prevents the helper
    # from resampling it again.
    # --------------------------------------------------------

    projection_metrics = (
        evaluate_projection_quality(

            source_matrix=
                fused_sample,

            low_dimensional_matrix=
                low_dimensional_sample,

            k=
                K_NEIGHBORS,

            source_metric=
                "cosine",

            sample_size=
                len(
                    fused_sample
                ),

            random_state=
                RANDOM_STATE

        )
    )


    trust = projection_metrics[
        "trustworthiness"
    ]


    preservation = projection_metrics[
        "knn_preservation"
    ]


    print(
        f"Trustworthiness:  "
        f"{trust:.4f}"
    )


    print(
        f"kNN preservation: "
        f"{preservation:.4f}"
    )


    # ========================================================
    # 8. RESULT
    # ========================================================

    return {

        "atlas_id":
            config[
                "id"
            ],

        "atlas_label":
            config[
                "label"
            ],

        "n_items":
            int(
                len(
                    df
                )
            ),

        "evaluation_sample_size":
            int(
                len(
                    sample_indices
                )
            ),

        "knn_k":
            K_NEIGHBORS,

        "tfidf_max_features":
            TFIDF_MAX_FEATURES,

        "svd_components":
            int(
                semantic_sample.shape[1]
            ),

        "svd_explained_variance":
            explained_variance,

        "review_embedding_dimensions":
            int(
                review_sample.shape[1]
            ),

        "fused_dimensions":
            int(
                fused_sample.shape[1]
            ),

        "sample_review_coverage":
            float(
                has_review.mean()
            ),

        "review_share":
            REVIEW_SHARE,

        "source_metric":
            "cosine",

        "trustworthiness":
            trust,

        "knn_preservation":
            preservation,

    }


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MONO-DOMAIN PROJECTION FIDELITY"
    )


    print(
        f"Evaluation sample size: "
        f"{EVALUATION_SAMPLE_SIZE:,}"
    )


    print(
        f"k-neighbourhood: "
        f"{K_NEIGHBORS}"
    )


    print(
        f"TF-IDF maximum features: "
        f"{TFIDF_MAX_FEATURES:,}"
    )


    print(
        f"SVD dimensions: "
        f"{SVD_COMPONENTS}"
    )


    print(
        f"Review share: "
        f"{REVIEW_SHARE:.2f}"
    )


    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )


    results = []


    # ========================================================
    # EACH MONO DOMAIN
    # ========================================================

    for config in ATLAS_CONFIGS:


        result = evaluate_domain(
            config
        )


        results.append(
            result
        )


    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(

        OUTPUT_PATH,

        index=False

    )


    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    section(
        "PROJECTION FIDELITY COMPARISON"
    )


    display_columns = [

        "atlas_label",

        "n_items",

        "sample_review_coverage",

        "svd_explained_variance",

        "trustworthiness",

        "knn_preservation",

    ]


    display = results_df[
        display_columns
    ].copy()


    numeric_columns = (

        display
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )


    display[
        numeric_columns
    ] = (

        display[
            numeric_columns
        ]
        .round(
            4
        )

    )


    print(

        display.to_string(
            index=False
        )

    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    section(
        "INTERPRETATION GUIDE"
    )


    print(
        "Trustworthiness"
    )


    print(
        "  Higher is better."
    )


    print(
        "  Measures whether 2D introduces false local "
        "neighbours that were distant in the source space."
    )


    print()


    print(
        "kNN preservation"
    )


    print(
        "  Higher is better."
    )


    print(
        "  Measures the fraction of the exact source-space "
        "k nearest neighbours retained among the exact 2D "
        "k nearest neighbours."
    )


    print()


    print(
        "These metrics should be interpreted together:"
    )


    print(
        "  trustworthiness can remain high even when exact "
        "neighbour preservation is modest, because UMAP may "
        "retain meaningful locality without preserving the "
        "precise ranking of every nearest neighbour."
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    section(
        "OUTPUT"
    )


    print(
        OUTPUT_PATH
    )


    section(
        "MONO-DOMAIN PROJECTION EVALUATION COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()