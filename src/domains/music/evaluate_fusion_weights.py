# src/domains/music/evaluate_fusion_weights.py

# ============================================================
# EVALUATE MUSIC SEMANTIC FUSION WEIGHTS
#
# Purpose:
#
#   Evaluate how much influence CritiqueBrainz review
#   embeddings should have relative to the existing
#   tag-based TF-IDF representation.
#
#
# Existing semantic representation:
#
#       artist tags
#           ↓
#       TF-IDF
#           ↓
#       Truncated SVD
#           ↓
#       L2 normalization
#
#
# Review representation:
#
#       CritiqueBrainz reviews
#           ↓
#       Sentence-BERT
#           ↓
#       mean pooling per artist
#           ↓
#       L2 normalization
#
#
# Fusion:
#
#       similarity =
#
#           (1 - review_share) * TF-IDF similarity
#
#           +
#
#           review_share * review similarity
#
#
# For each candidate review share:
#
#   1. Construct the fused similarity matrix
#   2. Find nearest neighbours in fused space
#   3. Measure how coherent those neighbours remain
#      according to:
#
#          - original TF-IDF semantics
#          - CritiqueBrainz review semantics
#
#   4. Combine the two using a harmonic mean
#
#
# IMPORTANT:
#
#   This is a diagnostic unsupervised criterion.
#
#   It is NOT an objective ground-truth accuracy metric.
#
#
# Artists with only one review ARE retained.
#
# This is an intentional methodological decision because
# CritiqueBrainz coverage is sparse and excluding them would
# discard most reviewed artists.
#
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd


from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.decomposition import (
    TruncatedSVD
)

from sklearn.preprocessing import (
    normalize
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------
# Current exported Music atlas.
#
# Using the actual exported atlas means:
#
#   - we evaluate only artists currently present in the atlas
#   - we use the actual tag text exported by the current
#     Music pipeline
#
# This avoids accidentally evaluating a slightly different
# preprocessing configuration.
# ------------------------------------------------------------

ATLAS_FILE = (
    ROOT
    / "frontend"
    / "public"
    / "data"
    / "music"
    / "atlas.json"
)


# ------------------------------------------------------------
# CritiqueBrainz SBERT embeddings
# ------------------------------------------------------------

REVIEW_EMBEDDINGS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "music"
    / "music_review_embeddings.npz"
)


# ------------------------------------------------------------
# Evaluation output
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "music_fusion_weight_evaluation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# Existing TF-IDF configuration
# ------------------------------------------------------------

TFIDF_MAX_FEATURES = 5000

TFIDF_COMPONENTS = 256


# ------------------------------------------------------------
# Review coverage
# ------------------------------------------------------------
#
# Deliberately set to 1.
#
# Artists represented by a single review have a noisier
# review-derived representation, but excluding them would
# remove most CritiqueBrainz-covered artists.
#
# Their review signal will still be moderated by the selected
# fusion weight.
# ------------------------------------------------------------

MIN_REVIEWS = 1


# ------------------------------------------------------------
# Number of artists used for the pairwise evaluation.
#
# 3000 keeps the similarity matrices manageable while giving
# us a substantial and deterministic sample.
# ------------------------------------------------------------

MAX_EVALUATION_SAMPLE = 3000


# ------------------------------------------------------------
# Number of nearest neighbours evaluated for each artist
# ------------------------------------------------------------

K_NEIGHBORS = 15


# ------------------------------------------------------------
# Deterministic sampling
# ------------------------------------------------------------

RANDOM_STATE = 42


# ------------------------------------------------------------
# Candidate REVIEW shares
#
# 0.00 = TF-IDF only
# 1.00 = reviews only
# ------------------------------------------------------------

REVIEW_SHARES = [

    0.00,

    0.20,

    0.35,

    0.50,

    0.65,

    0.80,

    1.00,

]


# ============================================================
# SECTION PRINTING
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 60
    )

    print(
        title
    )

    print(
        "=" * 60
    )

    print()


# ============================================================
# LOAD CURRENT MUSIC ATLAS
# ============================================================


def load_music_atlas():
    """
    Load the currently exported Music atlas and return a
    dataframe containing:

        artist_mbid
        artist_name
        tags_text
    """

    section(
        "LOADING CURRENT MUSIC ATLAS"
    )


    if not ATLAS_FILE.exists():

        raise FileNotFoundError(

            "Current Music atlas was not found:\n\n"
            f"{ATLAS_FILE}\n\n"
            "Build the existing Music atlas first."

        )


    with open(
        ATLAS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )


    # --------------------------------------------------------
    # Support either:
    #
    #     [node, node, ...]
    #
    # or:
    #
    #     {"atlas": [node, node, ...]}
    # --------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        nodes = data


    elif (
        isinstance(
            data,
            dict
        )
        and
        isinstance(
            data.get("atlas"),
            list
        )
    ):

        nodes = data[
            "atlas"
        ]


    else:

        raise ValueError(

            "Unexpected Music atlas JSON structure."

        )


    records = []


    for node in nodes:


        # ----------------------------------------------------
        # MBID
        # ----------------------------------------------------

        artist_mbid = (
            node.get(
                "source_id"
            )
            or
            node.get(
                "id"
            )
        )


        if artist_mbid is None:

            continue


        artist_mbid = str(
            artist_mbid
        ).strip()


        if not artist_mbid:

            continue


        # ----------------------------------------------------
        # Artist name
        # ----------------------------------------------------

        artist_name = (

            node.get(
                "title"
            )

            or

            node.get(
                "metadata",
                {}
            ).get(
                "artist"
            )

        )


        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        text_features = node.get(
            "text",
            {}
        )


        tags = text_features.get(
            "tags",
            []
        )


        if isinstance(
            tags,
            list
        ):

            tags_text = " ".join(

                str(tag)

                for tag in tags

                if tag is not None

            )


        elif tags is None:

            tags_text = ""


        else:

            tags_text = str(
                tags
            )


        tags_text = (
            tags_text
            .strip()
        )


        records.append({

            "artist_mbid":
                artist_mbid,

            "artist_name":
                artist_name,

            "tags_text":
                tags_text,

        })


    artists = pd.DataFrame(
        records
    )


    # --------------------------------------------------------
    # Protect against accidental duplicate IDs
    # --------------------------------------------------------

    duplicates = int(

        artists[
            "artist_mbid"
        ].duplicated().sum()

    )


    if duplicates > 0:

        raise ValueError(

            f"Current Music atlas contains "
            f"{duplicates:,} duplicate artist MBIDs."

        )


    artists = artists.reset_index(
        drop=True
    )


    print(
        f"Music atlas artists: "
        f"{len(artists):,}"
    )


    print(
        f"Artists with tag text: "
        f"{int((artists['tags_text'] != '').sum()):,}"
    )


    print(
        f"Artists without tag text: "
        f"{int((artists['tags_text'] == '').sum()):,}"
    )


    return artists


# ============================================================
# BUILD EXISTING TF-IDF REPRESENTATION
# ============================================================


def build_tfidf_representation(
    artists
):
    """
    Rebuild the current tag-based semantic representation.

    TF-IDF is fit over the COMPLETE current Music atlas before
    reviewed artists are selected for evaluation.

    This preserves the vocabulary / document-frequency context
    of the atlas rather than fitting TF-IDF only on artists that
    happen to have CritiqueBrainz reviews.
    """

    section(
        "BUILDING MUSIC TF-IDF REPRESENTATION"
    )


    vectorizer = TfidfVectorizer(

        max_features=
            TFIDF_MAX_FEATURES,

        stop_words=
            "english"

    )


    tfidf_matrix = (
        vectorizer.fit_transform(
            artists[
                "tags_text"
            ]
        )
    )


    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )


    # --------------------------------------------------------
    # SVD dimensions cannot exceed matrix constraints.
    # --------------------------------------------------------

    max_components = min(

        TFIDF_COMPONENTS,

        tfidf_matrix.shape[0] - 1,

        tfidf_matrix.shape[1] - 1,

    )


    if max_components < 2:

        raise ValueError(

            "Not enough TF-IDF dimensions to perform "
            "semantic reduction."

        )


    print(
        f"Reducing "
        f"{tfidf_matrix.shape[1]:,}D "
        f"-> {max_components:,}D"
    )


    svd = TruncatedSVD(

        n_components=
            max_components,

        random_state=
            RANDOM_STATE

    )


    tfidf_reduced = (
        svd.fit_transform(
            tfidf_matrix
        )
    )


    explained_variance = float(

        svd.explained_variance_ratio_
        .sum()

    )


    print(
        f"SVD explained variance: "
        f"{explained_variance:.4f}"
    )


    # --------------------------------------------------------
    # Normalize so dot product = cosine similarity
    # --------------------------------------------------------

    tfidf_reduced = normalize(

        tfidf_reduced,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    return (
        tfidf_reduced,
        explained_variance
    )


# ============================================================
# LOAD AND ALIGN REVIEW EMBEDDINGS
# ============================================================


def align_review_embeddings(
    artists
):
    """
    Align CritiqueBrainz artist embeddings with the row order
    of the current Music atlas.
    """

    section(
        "ALIGNING CRITIQUEBRAINZ EMBEDDINGS"
    )


    if not REVIEW_EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(

            "Music review embeddings were not found:\n\n"
            f"{REVIEW_EMBEDDINGS_FILE}"

        )


    data = np.load(

        REVIEW_EMBEDDINGS_FILE,

        allow_pickle=True

    )


    required_keys = {

        "entity_ids",

        "embeddings",

        "review_counts",

    }


    missing_keys = (

        required_keys

        -

        set(
            data.files
        )

    )


    if missing_keys:

        raise ValueError(

            "Music review embedding file is missing "
            f"required arrays: {sorted(missing_keys)}"

        )


    review_ids = [

        str(value).strip()

        for value
        in data[
            "entity_ids"
        ]

    ]


    review_embeddings = np.asarray(

        data[
            "embeddings"
        ],

        dtype=np.float32

    )


    review_counts = np.asarray(

        data[
            "review_counts"
        ],

        dtype=np.int32

    )


    if (
        len(review_ids)
        !=
        review_embeddings.shape[0]
        or
        len(review_ids)
        !=
        len(review_counts)
    ):

        raise ValueError(

            "Review embedding arrays are not aligned."

        )


    # --------------------------------------------------------
    # Normalize again defensively.
    #
    # They should already be normalized by the reusable
    # embedding engine.
    # --------------------------------------------------------

    review_embeddings = normalize(

        review_embeddings,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    review_lookup = {

        artist_mbid:
            index

        for index, artist_mbid
        in enumerate(
            review_ids
        )

    }


    embedding_dimension = (
        review_embeddings.shape[1]
    )


    aligned_embeddings = np.zeros(

        (
            len(artists),
            embedding_dimension
        ),

        dtype=np.float32

    )


    aligned_counts = np.zeros(

        len(
            artists
        ),

        dtype=np.int32

    )


    aligned_entities = 0


    for index, artist_mbid in enumerate(

        artists[
            "artist_mbid"
        ]

    ):

        review_index = review_lookup.get(

            str(
                artist_mbid
            ).strip()

        )


        if review_index is None:

            continue


        aligned_embeddings[
            index
        ] = review_embeddings[
            review_index
        ]


        aligned_counts[
            index
        ] = review_counts[
            review_index
        ]


        aligned_entities += 1


    print(
        f"Review embeddings available: "
        f"{len(review_ids):,}"
    )


    print(
        f"Review embeddings aligned:   "
        f"{aligned_entities:,} / "
        f"{len(artists):,} artists"
    )


    print(
        f"Review embedding dimensions: "
        f"{embedding_dimension:,}"
    )


    return (
        aligned_embeddings,
        aligned_counts
    )


# ============================================================
# SELECT EVALUATION ARTISTS
# ============================================================


def select_evaluation_sample(

    artists,

    tfidf_embeddings,

    review_embeddings,

    review_counts

):
    """
    Select artists that have usable information from BOTH
    semantic modalities.

    Artists with one review are intentionally allowed.
    """


    section(
        "SELECTING EVALUATION ARTISTS"
    )


    # --------------------------------------------------------
    # TF-IDF vector must contain information
    # --------------------------------------------------------

    tfidf_norms = np.linalg.norm(

        tfidf_embeddings,

        axis=1

    )


    # --------------------------------------------------------
    # Review vector must contain information
    # --------------------------------------------------------

    review_norms = np.linalg.norm(

        review_embeddings,

        axis=1

    )


    eligible_mask = (

        (
            review_counts
            >=
            MIN_REVIEWS
        )

        &

        (
            tfidf_norms
            > 0
        )

        &

        (
            review_norms
            > 0
        )

    )


    eligible_indices = np.flatnonzero(

        eligible_mask

    )


    print(
        f"Minimum reviews required: "
        f"{MIN_REVIEWS}"
    )


    print(
        f"Eligible artists: "
        f"{len(eligible_indices):,}"
    )


    if len(
        eligible_indices
    ) < (
        K_NEIGHBORS + 1
    ):

        raise ValueError(

            "Too few eligible artists for the "
            "nearest-neighbour evaluation."

        )


    # --------------------------------------------------------
    # Deterministic sample
    # --------------------------------------------------------

    if (
        len(
            eligible_indices
        )
        >
        MAX_EVALUATION_SAMPLE
    ):

        rng = np.random.default_rng(

            RANDOM_STATE

        )


        sample_indices = rng.choice(

            eligible_indices,

            size=
                MAX_EVALUATION_SAMPLE,

            replace=False

        )


        # Sorting gives deterministic atlas-row order after
        # deterministic selection.
        sample_indices = np.sort(

            sample_indices

        )


    else:

        sample_indices = (
            eligible_indices
        )


    print(
        f"Evaluation sample: "
        f"{len(sample_indices):,}"
    )


    sampled_review_counts = (

        review_counts[
            sample_indices
        ]

    )


    print()

    print(
        "Reviews used per sampled artist:"
    )


    print(

        pd.Series(
            sampled_review_counts
        )
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .to_string()

    )


    return sample_indices


# ============================================================
# COSINE SIMILARITY MATRIX
# ============================================================


def cosine_similarity_matrix(
    embeddings
):
    """
    Embeddings are already L2-normalized, therefore:

        cosine_similarity(A, B)
            =
        A dot B
    """

    return (

        embeddings

        @

        embeddings.T

    ).astype(
        np.float32
    )


# ============================================================
# NEAREST NEIGHBOURS
# ============================================================


def nearest_neighbors(

    similarity_matrix,

    k

):
    """
    Return the K nearest neighbours for every entity.

    Self-similarity is removed.
    """


    similarities = (
        similarity_matrix.copy()
    )


    np.fill_diagonal(

        similarities,

        -np.inf

    )


    # --------------------------------------------------------
    # Fast top-k selection
    # --------------------------------------------------------

    neighbor_indices = np.argpartition(

        -similarities,

        kth=
            k - 1,

        axis=1

    )[
        :,
        :k
    ]


    # --------------------------------------------------------
    # Sort the selected neighbours by actual similarity
    # for deterministic ordering.
    # --------------------------------------------------------

    row_indices = np.arange(

        similarities.shape[0]

    )[
        :,
        None
    ]


    selected_similarities = (

        similarities[

            row_indices,

            neighbor_indices

        ]

    )


    ordering = np.argsort(

        -selected_similarities,

        axis=1

    )


    neighbor_indices = np.take_along_axis(

        neighbor_indices,

        ordering,

        axis=1

    )


    return neighbor_indices


# ============================================================
# NEIGHBOUR COHERENCE
# ============================================================


def mean_neighbor_coherence(

    reference_similarity,

    neighbor_indices

):
    """
    Measure how similar each artist is to its fused-space
    neighbours according to a reference semantic space.
    """


    row_indices = np.arange(

        reference_similarity.shape[0]

    )[
        :,
        None
    ]


    values = reference_similarity[

        row_indices,

        neighbor_indices

    ]


    return float(

        values.mean()

    )


# ============================================================
# BALANCED SCORE
# ============================================================


def balanced_score(

    tfidf_coherence,

    review_coherence

):
    """
    Convert cosine similarities from [-1, 1] to [0, 1],
    then combine them using the harmonic mean.

    The harmonic mean penalizes solutions that perform well
    in one semantic space while performing poorly in the
    other.
    """


    tfidf_scaled = (

        tfidf_coherence
        + 1.0

    ) / 2.0


    review_scaled = (

        review_coherence
        + 1.0

    ) / 2.0


    denominator = (

        tfidf_scaled
        +
        review_scaled

    )


    if denominator <= 0:

        return 0.0


    return float(

        2.0

        *

        tfidf_scaled

        *

        review_scaled

        /

        denominator

    )


# ============================================================
# EVALUATE REVIEW SHARES
# ============================================================


def evaluate_weights(

    tfidf_similarity,

    review_similarity

):

    section(
        "EVALUATING FUSION WEIGHTS"
    )


    results = []


    print(

        f"{'Review share':>12} | "
        f"{'TF-IDF coherence':>16} | "
        f"{'Review coherence':>16} | "
        f"{'Balanced':>10}"

    )


    print(
        "-" * 65
    )


    for review_share in (
        REVIEW_SHARES
    ):


        # ----------------------------------------------------
        # Because both semantic blocks are individually
        # normalized and fusion uses:
        #
        #     sqrt(1-r) * TF-IDF
        #     sqrt(r)   * reviews
        #
        # fused cosine similarity is equivalent to:
        #
        #     (1-r) * sim_tfidf
        #       +
        #     r * sim_reviews
        #
        # for entities represented in both modalities.
        # ----------------------------------------------------

        tfidf_share = (
            1.0
            -
            review_share
        )


        fused_similarity = (

            tfidf_share
            *
            tfidf_similarity

            +

            review_share
            *
            review_similarity

        )


        neighbors = nearest_neighbors(

            fused_similarity,

            K_NEIGHBORS

        )


        tfidf_coherence = (
            mean_neighbor_coherence(

                tfidf_similarity,

                neighbors

            )
        )


        review_coherence = (
            mean_neighbor_coherence(

                review_similarity,

                neighbors

            )
        )


        score = balanced_score(

            tfidf_coherence,

            review_coherence

        )


        results.append({

            "review_share":
                review_share,

            "tfidf_share":
                tfidf_share,

            "tfidf_coherence":
                tfidf_coherence,

            "review_coherence":
                review_coherence,

            "balanced_score":
                score,

        })


        print(

            f"{review_share:12.2f} | "
            f"{tfidf_coherence:16.4f} | "
            f"{review_coherence:16.4f} | "
            f"{score:10.4f}"

        )


    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "MUSIC FUSION WEIGHT EVALUATION"
    )


    print(
        "Configuration:"
    )

    print(
        f"  TF-IDF max features: "
        f"{TFIDF_MAX_FEATURES:,}"
    )

    print(
        f"  TF-IDF SVD dimensions: "
        f"{TFIDF_COMPONENTS:,}"
    )

    print(
        f"  Minimum reviews: "
        f"{MIN_REVIEWS}"
    )

    print(
        f"  Maximum sample size: "
        f"{MAX_EVALUATION_SAMPLE:,}"
    )

    print(
        f"  Neighbours: "
        f"{K_NEIGHBORS}"
    )

    print(
        f"  Random state: "
        f"{RANDOM_STATE}"
    )


    # ========================================================
    # 1. CURRENT MUSIC ATLAS
    # ========================================================

    artists = (
        load_music_atlas()
    )


    # ========================================================
    # 2. EXISTING TF-IDF SEMANTICS
    # ========================================================

    (
        tfidf_embeddings,
        explained_variance

    ) = build_tfidf_representation(

        artists

    )


    # ========================================================
    # 3. REVIEW EMBEDDINGS
    # ========================================================

    (
        review_embeddings,
        review_counts

    ) = align_review_embeddings(

        artists

    )


    # ========================================================
    # 4. SELECT ARTISTS
    # ========================================================

    sample_indices = (
        select_evaluation_sample(

            artists,

            tfidf_embeddings,

            review_embeddings,

            review_counts

        )
    )


    # ========================================================
    # 5. EXTRACT SAMPLE REPRESENTATIONS
    # ========================================================

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


    # Normalize defensively again after sampling.

    tfidf_sample = normalize(

        tfidf_sample,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    review_sample = normalize(

        review_sample,

        norm="l2",

        axis=1

    ).astype(
        np.float32
    )


    # ========================================================
    # 6. COMPUTE SIMILARITY MATRICES ONCE
    # ========================================================

    section(
        "COMPUTING SIMILARITY MATRICES"
    )


    print(
        f"Matrix size: "
        f"{len(sample_indices):,} × "
        f"{len(sample_indices):,}"
    )


    tfidf_similarity = (
        cosine_similarity_matrix(
            tfidf_sample
        )
    )


    review_similarity = (
        cosine_similarity_matrix(
            review_sample
        )
    )


    print(
        "Similarity matrices complete."
    )


    # ========================================================
    # 7. EVALUATE WEIGHTS
    # ========================================================

    results = evaluate_weights(

        tfidf_similarity,

        review_similarity

    )


    # ========================================================
    # 8. ADD EVALUATION METADATA
    # ========================================================

    results[
        "evaluation_sample_size"
    ] = len(
        sample_indices
    )


    results[
        "min_reviews"
    ] = MIN_REVIEWS


    results[
        "k_neighbors"
    ] = K_NEIGHBORS


    results[
        "tfidf_components"
    ] = tfidf_sample.shape[1]


    results[
        "review_dimensions"
    ] = review_sample.shape[1]


    results[
        "svd_explained_variance"
    ] = explained_variance


    # ========================================================
    # 9. SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    results.to_csv(

        OUTPUT_FILE,

        index=False

    )


    # ========================================================
    # 10. BEST RESULT
    # ========================================================

    best_index = (

        results[
            "balanced_score"
        ]
        .idxmax()

    )


    best = results.loc[
        best_index
    ]


    section(
        "EVALUATION COMPLETE"
    )


    print(
        "Best diagnostic result:"
    )

    print()

    print(
        f"Review share:       "
        f"{best['review_share']:.2f}"
    )

    print(
        f"TF-IDF share:       "
        f"{best['tfidf_share']:.2f}"
    )

    print(
        f"TF-IDF coherence:   "
        f"{best['tfidf_coherence']:.4f}"
    )

    print(
        f"Review coherence:   "
        f"{best['review_coherence']:.4f}"
    )

    print(
        f"Balanced score:     "
        f"{best['balanced_score']:.4f}"
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(

        "The maximum balanced score should be interpreted "
        "as a diagnostic recommendation, not as objective "
        "ground truth."

    )


    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()