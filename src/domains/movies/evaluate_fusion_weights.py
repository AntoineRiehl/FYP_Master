# src/domains/movies/evaluate_fusion_weights.py

# ============================================================
# MOVIE SEMANTIC FUSION WEIGHT EVALUATION
#
# Purpose:
#
#   Choose a sensible REVIEW_SHARE for combining:
#
#       MovieLens tag TF-IDF semantics
#           +
#       IMDb Sentence-BERT review semantics
#
#
# Evaluation principle:
#
#   A good fused neighbourhood should remain coherent
#   according to BOTH semantic sources.
#
# For each candidate review share:
#
#   1. Build fused cosine similarities
#   2. Find nearest neighbours in the fused space
#   3. Measure TF-IDF coherence of those neighbours
#   4. Measure review coherence of those neighbours
#   5. Combine the two using a harmonic mean
#
#
# This is NOT supervised ground truth.
#
# It is a pragmatic unsupervised criterion for selecting
# a balanced fusion weight.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

from src.domains.movies.load_data import (
    load_raw_data
)

from src.domains.movies.merge_data import (
    compute_movie_stats,
    merge_movies
)

from src.domains.movies.feature_engineering import (
    concatenate_tags
)

from src.atlas.embeddings.tfidf_pipeline import (
    get_tfidf_embeddings
)

from src.atlas.embeddings.semantic_fusion import (
    prepare_semantic_fusion_components
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MOVIE_REVIEW_EMBEDDINGS = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "movies"
    / "movie_review_embeddings.npz"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "movie_fusion_weight_evaluation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# TF-IDF dimensionality after TruncatedSVD.
# ------------------------------------------------------------

TFIDF_COMPONENTS = 256


# ------------------------------------------------------------
# Candidate review shares.
#
# 0.00 and 1.00 are included as diagnostic endpoints.
#
# We are NOT creating reviews-only or TF-IDF-only atlases
# here. They simply help show how the scoring behaves at
# the extremes.
# ------------------------------------------------------------

REVIEW_SHARES = [

    0.00,

    0.20,

    0.35,

    0.50,

    0.65,

    0.80,

    1.00

]


# ------------------------------------------------------------
# Only evaluate movies with enough IMDb reviews to make their
# review embedding reasonably representative.
# ------------------------------------------------------------

MIN_REVIEWS = 10


# ------------------------------------------------------------
# Evaluating every movie pair would be unnecessary.
#
# A deterministic sample of 3,000 movies gives:
#
#     3,000 x 3,000
#
# similarity matrices, which are manageable and sufficient
# for this parameter comparison.
# ------------------------------------------------------------

EVALUATION_SAMPLE_SIZE = 3000


# ------------------------------------------------------------
# Number of fused neighbours used when evaluating local
# semantic coherence.
# ------------------------------------------------------------

K_NEIGHBORS = 15


RANDOM_SEED = 42


# ============================================================
# HELPER
# ============================================================


def harmonic_mean(
    a,
    b
):
    """
    Harmonic mean of two positive scores.

    Compared with a normal arithmetic mean, this penalizes
    solutions that perform very strongly on one modality
    but poorly on the other.
    """

    if (
        a <= 0
        or b <= 0
    ):

        return 0.0

    return (
        2.0
        * a
        * b
        / (
            a + b
        )
    )


# ============================================================
# LOAD CURRENT MOVIE SEMANTIC INPUT
# ============================================================


def build_movie_tfidf():
    """
    Reproduce the movie TF-IDF representation used by
    build_movie_map.py.
    """

    print()
    print("=" * 60)
    print("BUILDING MOVIE TF-IDF REPRESENTATION")
    print("=" * 60)
    print()

    movies, ratings, tags = (
        load_raw_data()
    )

    movie_stats = compute_movie_stats(
        ratings
    )

    merged = merge_movies(
        movie_stats,
        movies
    )

    movie_tags = concatenate_tags(
        tags
    )

    final = merged.merge(

        movie_tags,

        on="movieId",

        how="left"

    )

    final["tags_text"] = (

        final["tags_text"]
        .fillna("")

    )

    tfidf_matrix, vectorizer = (
        get_tfidf_embeddings(

            final,

            text_column=
                "tags_text",

            model_name=
                "movies_tags"

        )
    )

    return (
        final,
        tfidf_matrix
    )


# ============================================================
# SAMPLE ELIGIBLE MOVIES
# ============================================================


def select_evaluation_sample(
    tfidf_embeddings,
    review_embeddings,
    review_counts
):
    """
    Select movies that:

        - have at least MIN_REVIEWS reviews
        - have a non-zero TF-IDF representation
        - have a non-zero review representation
    """

    tfidf_norms = np.linalg.norm(

        tfidf_embeddings,

        axis=1

    )

    review_norms = np.linalg.norm(

        review_embeddings,

        axis=1

    )

    eligible_mask = (

        (review_counts >= MIN_REVIEWS)

        &

        (tfidf_norms > 0)

        &

        (review_norms > 0)

    )

    eligible_indices = np.flatnonzero(
        eligible_mask
    )

    print(
        f"Eligible movies: "
        f"{len(eligible_indices):,}"
    )

    if len(eligible_indices) == 0:

        raise ValueError(
            "No movies satisfy the evaluation "
            "criteria."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_size = min(

        EVALUATION_SAMPLE_SIZE,

        len(eligible_indices)

    )

    if sample_size < len(
        eligible_indices
    ):

        sample_indices = rng.choice(

            eligible_indices,

            size=sample_size,

            replace=False

        )

    else:

        sample_indices = (
            eligible_indices
        )

    sample_indices = np.sort(
        sample_indices
    )

    print(
        f"Evaluation sample: "
        f"{len(sample_indices):,}"
    )

    return sample_indices


# ============================================================
# COSINE SIMILARITIES
# ============================================================


def compute_similarity_matrices(
    tfidf_sample,
    review_sample
):
    """
    Both input matrices are already L2-normalized.

    Therefore:

        A @ A.T

    gives cosine similarity directly.
    """

    print()
    print(
        "Computing TF-IDF similarity matrix..."
    )

    tfidf_similarity = (

        tfidf_sample

        @

        tfidf_sample.T

    ).astype(
        np.float32
    )

    print(
        "Computing review similarity matrix..."
    )

    review_similarity = (

        review_sample

        @

        review_sample.T

    ).astype(
        np.float32
    )

    # Numerical clipping protects against tiny floating-point
    # values slightly outside [-1, 1].

    tfidf_similarity = np.clip(

        tfidf_similarity,

        -1.0,

        1.0

    )

    review_similarity = np.clip(

        review_similarity,

        -1.0,

        1.0

    )

    return (
        tfidf_similarity,
        review_similarity
    )


# ============================================================
# EVALUATE ONE REVIEW SHARE
# ============================================================


def evaluate_review_share(
    review_share,
    tfidf_similarity,
    review_similarity
):
    """
    Evaluate one candidate review share.

    Since each semantic block is L2-normalized and the
    actual fusion uses square-root block weighting:

        fused similarity

        = (1-r) * TF-IDF similarity
          +
          r * review similarity

    for entities containing both representations.
    """

    tfidf_share = (
        1.0 - review_share
    )

    # ========================================================
    # FUSED SIMILARITY
    # ========================================================

    fused_similarity = (

        tfidf_share
        * tfidf_similarity

        +

        review_share
        * review_similarity

    )

    # --------------------------------------------------------
    # Do not allow a movie to select itself as a neighbour.
    # --------------------------------------------------------

    np.fill_diagonal(

        fused_similarity,

        -np.inf

    )

    n_movies = fused_similarity.shape[0]

    k = min(

        K_NEIGHBORS,

        n_movies - 1

    )

    # ========================================================
    # TOP-K NEIGHBOURS
    # ========================================================

    neighbour_indices = np.argpartition(

        fused_similarity,

        kth=n_movies - k,

        axis=1

    )[
        :,
        -k:
    ]

    rows = np.arange(
        n_movies
    )[:, None]

    # ========================================================
    # MODALITY COHERENCE
    # ========================================================

    tfidf_values = tfidf_similarity[

        rows,

        neighbour_indices

    ]

    review_values = review_similarity[

        rows,

        neighbour_indices

    ]

    mean_tfidf_similarity = float(
        tfidf_values.mean()
    )

    mean_review_similarity = float(
        review_values.mean()
    )

    # --------------------------------------------------------
    # Convert cosine [-1, 1] to a positive [0, 1] scale
    # before calculating the harmonic mean.
    #
    # This avoids mathematical issues if one of the cosine
    # averages becomes slightly negative.
    # --------------------------------------------------------

    tfidf_score = (

        mean_tfidf_similarity
        + 1.0

    ) / 2.0

    review_score = (

        mean_review_similarity
        + 1.0

    ) / 2.0

    balanced_score = harmonic_mean(

        tfidf_score,

        review_score

    )

    return {

        "review_share":
            review_share,

        "tfidf_share":
            tfidf_share,

        "tfidf_neighbor_similarity":
            mean_tfidf_similarity,

        "review_neighbor_similarity":
            mean_review_similarity,

        "tfidf_score_normalized":
            tfidf_score,

        "review_score_normalized":
            review_score,

        "balanced_score":
            balanced_score

    }


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 60)
    print("MOVIE FUSION WEIGHT EVALUATION")
    print("=" * 60)

    print()
    print(
        f"Minimum reviews:       "
        f"{MIN_REVIEWS}"
    )

    print(
        f"Evaluation sample:     "
        f"{EVALUATION_SAMPLE_SIZE:,}"
    )

    print(
        f"K neighbours:          "
        f"{K_NEIGHBORS}"
    )

    print(
        f"TF-IDF SVD components: "
        f"{TFIDF_COMPONENTS}"
    )

    # ========================================================
    # EXISTING TF-IDF
    # ========================================================

    (
        final,
        tfidf_matrix

    ) = build_movie_tfidf()

    # ========================================================
    # PREPARE TF-IDF + SBERT COMPONENTS
    # ========================================================

    print()
    print("=" * 60)
    print("PREPARING SEMANTIC COMPONENTS")
    print("=" * 60)

    (
        tfidf_embeddings,
        review_embeddings,
        review_counts,
        information

    ) = prepare_semantic_fusion_components(

        tfidf_matrix=
            tfidf_matrix,

        entity_ids=
            final[
                "movieId"
            ].to_numpy(),

        review_embeddings_path=
            MOVIE_REVIEW_EMBEDDINGS,

        tfidf_components=
            TFIDF_COMPONENTS,

        random_state=
            RANDOM_SEED

    )

    print()
    print(
        f"TF-IDF dimensions: "
        f"{tfidf_embeddings.shape[1]}"
    )

    print(
        f"Review dimensions: "
        f"{review_embeddings.shape[1]}"
    )

    print(
        f"Movies with reviews: "
        f"{(review_counts > 0).sum():,}"
    )

    # ========================================================
    # EVALUATION SAMPLE
    # ========================================================

    print()
    print("=" * 60)
    print("SELECTING EVALUATION SAMPLE")
    print("=" * 60)

    sample_indices = (
        select_evaluation_sample(

            tfidf_embeddings=
                tfidf_embeddings,

            review_embeddings=
                review_embeddings,

            review_counts=
                review_counts

        )
    )

    tfidf_sample = (
        tfidf_embeddings[
            sample_indices
        ]
    )

    review_sample = (
        review_embeddings[
            sample_indices
        ]
    )

    # ========================================================
    # SIMILARITY MATRICES
    # ========================================================

    (
        tfidf_similarity,
        review_similarity

    ) = compute_similarity_matrices(

        tfidf_sample=
            tfidf_sample,

        review_sample=
            review_sample

    )

    # ========================================================
    # TEST REVIEW SHARES
    # ========================================================

    print()
    print("=" * 60)
    print("TESTING FUSION WEIGHTS")
    print("=" * 60)
    print()

    results = []

    for review_share in REVIEW_SHARES:

        result = evaluate_review_share(

            review_share=
                review_share,

            tfidf_similarity=
                tfidf_similarity,

            review_similarity=
                review_similarity

        )

        results.append(
            result
        )

        print(

            f"Review share "
            f"{review_share:>4.2f} | "

            f"TF-IDF coherence "
            f"{result['tfidf_neighbor_similarity']:.4f} | "

            f"Review coherence "
            f"{result['review_neighbor_similarity']:.4f} | "

            f"Balanced "
            f"{result['balanced_score']:.4f}"

        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(

        OUTPUT_FILE,

        index=False

    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    best_index = (

        results_df[
            "balanced_score"
        ]
        .idxmax()

    )

    best = results_df.loc[
        best_index
    ]

    recommended_review_share = float(
        best[
            "review_share"
        ]
    )

    print()
    print("=" * 60)
    print("RECOMMENDED FUSION")
    print("=" * 60)
    print()

    print(
        f"Recommended REVIEW_SHARE: "
        f"{recommended_review_share:.2f}"
    )

    print(
        f"Corresponding TFIDF_SHARE: "
        f"{1.0 - recommended_review_share:.2f}"
    )

    print()

    print(
        f"TF-IDF neighbour coherence: "
        f"{best['tfidf_neighbor_similarity']:.4f}"
    )

    print(
        f"Review neighbour coherence: "
        f"{best['review_neighbor_similarity']:.4f}"
    )

    print(
        f"Balanced score: "
        f"{best['balanced_score']:.4f}"
    )

    print()

    print(
        "Evaluation results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Use the recommended REVIEW_SHARE "
        "as the initial value in build_movie_map.py."
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()