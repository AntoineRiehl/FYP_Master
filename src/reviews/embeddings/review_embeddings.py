# src/reviews/embeddings/review_embeddings.py

# ============================================================
# GENERIC REVIEW EMBEDDING ENGINE
#
# Purpose:
#   Convert review-level text data into one semantic embedding
#   per entity.
#
# Designed to be reusable across:
#
#   - Movies
#   - Music
#   - Restaurants
#
#
# Method:
#
#   1. Read review IDs without loading all review text
#   2. Select up to N reviews per entity using deterministic
#      hash-based sampling
#   3. Encode selected reviews with SentenceTransformer
#   4. L2-normalize individual review embeddings
#   5. Mean-pool review embeddings per entity
#   6. L2-normalize final entity embeddings
#   7. Save embeddings + metadata
#
# ============================================================


from pathlib import Path
import hashlib
import heapq
import json

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# HELPERS
# ============================================================


def deterministic_score(
    review_id,
    random_seed
):
    """
    Create a deterministic pseudo-random score for a review.

    We avoid Python's built-in hash() because its exact values
    are not intended to be stable across different runs.

    Using a cryptographic hash gives us reproducible sampling:

        same review_id
        + same seed
        = same score

    This allows us to select a random-like but completely
    reproducible subset of reviews.
    """

    value = (
        f"{random_seed}|{review_id}"
        .encode("utf-8")
    )

    digest = hashlib.blake2b(
        value,
        digest_size=8
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False
    )


# ============================================================
# REVIEW SELECTION
# ============================================================


def select_reviews(
    input_file,
    entity_id_column,
    review_id_column,
    max_reviews_per_entity=50,
    random_seed=42,
    chunk_size=100_000
):
    """
    Select at most max_reviews_per_entity reviews for each
    entity without loading review text into memory.

    Selection is deterministic.

    For each entity, we retain the reviews with the smallest
    deterministic hash scores.

    Conceptually this behaves like reproducible random
    sampling.
    """

    print()
    print("=" * 60)
    print("SELECTING REVIEWS")
    print("=" * 60)
    print()

    selected = {}

    total_rows = 0

    reader = pd.read_csv(
        input_file,
        usecols=[
            entity_id_column,
            review_id_column
        ],
        chunksize=chunk_size
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):

        total_rows += len(chunk)

        for row in chunk.itertuples(
            index=False,
            name=None
        ):

            entity_id = row[0]
            review_id = str(row[1])

            score = deterministic_score(
                review_id,
                random_seed
            )

            # ------------------------------------------------
            # Use a max-heap.
            #
            # Python heapq is a min-heap, so we store negative
            # scores. The largest selected score is therefore
            # always at the top and can be replaced easily.
            # ------------------------------------------------

            heap = selected.setdefault(
                entity_id,
                []
            )

            item = (
                -score,
                review_id
            )

            if len(heap) < max_reviews_per_entity:

                heapq.heappush(
                    heap,
                    item
                )

            else:

                largest_selected_score = (
                    -heap[0][0]
                )

                if score < largest_selected_score:

                    heapq.heapreplace(
                        heap,
                        item
                    )

        print(

            f"\r"
            f"Selection chunks: "
            f"{chunk_number:,} | "

            f"Reviews scanned: "
            f"{total_rows:,} | "

            f"Entities found: "
            f"{len(selected):,}",

            end=""

        )

    print()
    print()

    # ========================================================
    # CONVERT HEAPS TO REVIEW ID SET
    # ========================================================

    selected_review_ids = set()

    selected_counts = {}

    for entity_id, heap in selected.items():

        review_ids = [
            item[1]
            for item in heap
        ]

        selected_review_ids.update(
            review_ids
        )

        selected_counts[
            entity_id
        ] = len(
            review_ids
        )

    print(
        f"Reviews scanned:       "
        f"{total_rows:,}"
    )

    print(
        f"Entities represented:  "
        f"{len(selected):,}"
    )

    print(
        f"Reviews selected:      "
        f"{len(selected_review_ids):,}"
    )

    print(
        f"Maximum per entity:    "
        f"{max_reviews_per_entity:,}"
    )

    return (
        selected_review_ids,
        selected_counts
    )


# ============================================================
# L2 NORMALIZATION
# ============================================================


def l2_normalize_rows(matrix):
    """
    L2-normalize every row in a 2D matrix.
    """

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return (
        matrix
        / norms
    )


# ============================================================
# EMBEDDING + AGGREGATION
# ============================================================


def encode_selected_reviews(
    input_file,
    selected_review_ids,
    selected_counts,
    entity_id_column,
    review_id_column,
    text_column,
    model_name,
    batch_size=64,
    chunk_size=10_000,
    device=None
):
    """
    Encode selected reviews and aggregate them into one
    embedding per entity.

    Individual review embeddings are L2-normalized first.

    Entity embedding:

        mean(review embeddings)

    The resulting entity embedding is then L2-normalized.
    """

    print()
    print("=" * 60)
    print("LOADING SENTENCE TRANSFORMER")
    print("=" * 60)
    print()

    model = SentenceTransformer(
        model_name,
        device=device
    )

    embedding_dimension = (
        model.get_sentence_embedding_dimension()
    )

    print(
        f"Model:      {model_name}"
    )

    print(
        f"Device:     {model.device}"
    )

    print(
        f"Dimensions: {embedding_dimension}"
    )

    # ========================================================
    # ENTITY INDEX
    # ========================================================

    entity_ids = sorted(
        selected_counts.keys()
    )

    entity_to_index = {

        entity_id: index

        for index, entity_id
        in enumerate(entity_ids)

    }

    # --------------------------------------------------------
    # Running sum of review embeddings.
    #
    # This means we NEVER need to keep millions of review
    # embeddings in memory.
    # --------------------------------------------------------

    embedding_sums = np.zeros(

        (
            len(entity_ids),
            embedding_dimension
        ),

        dtype=np.float32

    )

    encoded_counts = np.zeros(

        len(entity_ids),

        dtype=np.int32

    )

    # ========================================================
    # SECOND PASS THROUGH REVIEW DATA
    # ========================================================

    print()
    print("=" * 60)
    print("ENCODING REVIEWS")
    print("=" * 60)
    print()

    total_rows_scanned = 0

    total_reviews_encoded = 0

    reader = pd.read_csv(

        input_file,

        usecols=[
            entity_id_column,
            review_id_column,
            text_column
        ],

        chunksize=chunk_size

    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):

        total_rows_scanned += len(
            chunk
        )

        # ----------------------------------------------------
        # Keep only reviews selected in pass 1.
        # ----------------------------------------------------

        chunk[
            review_id_column
        ] = (
            chunk[
                review_id_column
            ]
            .astype(str)
        )

        selected_chunk = chunk[

            chunk[
                review_id_column
            ]
            .isin(
                selected_review_ids
            )

        ].copy()

        if selected_chunk.empty:

            continue

        # ----------------------------------------------------
        # Ensure clean text.
        # ----------------------------------------------------

        selected_chunk[
            text_column
        ] = (

            selected_chunk[
                text_column
            ]
            .fillna("")
            .astype(str)
            .str.strip()

        )

        selected_chunk = selected_chunk[

            selected_chunk[
                text_column
            ]
            != ""

        ]

        if selected_chunk.empty:

            continue

        texts = (
            selected_chunk[
                text_column
            ]
            .tolist()
        )

        # ====================================================
        # SENTENCE-BERT
        # ====================================================

        embeddings = model.encode(

            texts,

            batch_size=batch_size,

            show_progress_bar=False,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        embeddings = embeddings.astype(
            np.float32,
            copy=False
        )

        # ====================================================
        # MAP REVIEWS TO ENTITY ROWS
        # ====================================================

        indices = np.array(

            [

                entity_to_index[
                    entity_id
                ]

                for entity_id
                in selected_chunk[
                    entity_id_column
                ]

            ],

            dtype=np.int64

        )

        # ----------------------------------------------------
        # Efficiently add review embeddings to corresponding
        # entity sums.
        # ----------------------------------------------------

        np.add.at(

            embedding_sums,

            indices,

            embeddings

        )

        np.add.at(

            encoded_counts,

            indices,

            1

        )

        total_reviews_encoded += len(
            embeddings
        )

        print(

            f"\r"
            f"Encoding chunks: "
            f"{chunk_number:,} | "

            f"Rows scanned: "
            f"{total_rows_scanned:,} | "

            f"Reviews encoded: "
            f"{total_reviews_encoded:,}",

            end=""

        )

    print()
    print()

    # ========================================================
    # MEAN POOLING
    # ========================================================

    print(
        "Mean-pooling review embeddings..."
    )

    valid_mask = (
        encoded_counts > 0
    )

    if not valid_mask.all():

        missing_count = int(
            (~valid_mask).sum()
        )

        print(
            f"WARNING: "
            f"{missing_count:,} entities received "
            f"no encoded reviews."
        )

    # Avoid division by zero.
    divisor = np.maximum(
        encoded_counts,
        1
    ).astype(
        np.float32
    )

    entity_embeddings = (

        embedding_sums
        /
        divisor[:, None]

    )

    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    entity_embeddings = (
        l2_normalize_rows(
            entity_embeddings
        )
        .astype(
            np.float32
        )
    )

    return (

        np.asarray(
            entity_ids
        ),

        entity_embeddings,

        encoded_counts,

        embedding_dimension,

        total_reviews_encoded,

        str(
            model.device
        )

    )


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_embedding_outputs(
    output_npz,
    output_index_csv,
    output_metadata_json,
    entity_ids,
    embeddings,
    review_counts,
    entity_id_column,
    model_name,
    embedding_dimension,
    max_reviews_per_entity,
    random_seed,
    total_reviews_encoded,
    device
):
    """
    Save vectors and reproducibility metadata.
    """

    output_npz = Path(
        output_npz
    )

    output_index_csv = Path(
        output_index_csv
    )

    output_metadata_json = Path(
        output_metadata_json
    )

    output_npz.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_index_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_metadata_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # NPZ
    # ========================================================

    np.savez_compressed(

        output_npz,

        entity_ids=entity_ids,

        embeddings=embeddings,

        review_counts=review_counts

    )

    # ========================================================
    # HUMAN-READABLE INDEX
    # ========================================================

    index_df = pd.DataFrame({

        entity_id_column:
            entity_ids,

        "reviews_used":
            review_counts

    })

    index_df.to_csv(
        output_index_csv,
        index=False
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "model":
            model_name,

        "embedding_dimension":
            int(
                embedding_dimension
            ),

        "max_reviews_per_entity":
            int(
                max_reviews_per_entity
            ),

        "selection_method":
            "deterministic_hash_sampling",

        "random_seed":
            int(
                random_seed
            ),

        "aggregation_method":
            "mean_pooling",

        "individual_review_normalization":
            "L2",

        "entity_embedding_normalization":
            "L2",

        "long_review_policy":
            "model_default_truncation",

        "review_weighting":
            "equal",

        "total_entities":
            int(
                len(entity_ids)
            ),

        "total_reviews_encoded":
            int(
                total_reviews_encoded
            ),

        "device":
            device

    }

    with open(
        output_metadata_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


# ============================================================
# PUBLIC BUILD FUNCTION
# ============================================================


def build_review_embeddings(
    input_file,
    output_npz,
    output_index_csv,
    output_metadata_json,
    entity_id_column,
    review_id_column="review_id",
    text_column="review_text",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_reviews_per_entity=50,
    random_seed=42,
    batch_size=64,
    selection_chunk_size=100_000,
    encoding_chunk_size=10_000,
    device=None
):
    """
    Main reusable review embedding function.

    This is the function domain-specific scripts should call.
    """

    input_file = Path(
        input_file
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Review input file not found:\n"
            f"{input_file}"
        )

    print()
    print("=" * 60)
    print("BUILDING REVIEW EMBEDDINGS")
    print("=" * 60)

    print()
    print(
        f"Input:                "
        f"{input_file}"
    )

    print(
        f"Entity ID:            "
        f"{entity_id_column}"
    )

    print(
        f"Model:                "
        f"{model_name}"
    )

    print(
        f"Max reviews/entity:   "
        f"{max_reviews_per_entity}"
    )

    print(
        f"Random seed:          "
        f"{random_seed}"
    )

    # ========================================================
    # PASS 1 — REVIEW SELECTION
    # ========================================================

    (
        selected_review_ids,
        selected_counts

    ) = select_reviews(

        input_file=input_file,

        entity_id_column=
            entity_id_column,

        review_id_column=
            review_id_column,

        max_reviews_per_entity=
            max_reviews_per_entity,

        random_seed=
            random_seed,

        chunk_size=
            selection_chunk_size

    )

    # ========================================================
    # PASS 2 — EMBEDDING
    # ========================================================

    (
        entity_ids,
        embeddings,
        review_counts,
        embedding_dimension,
        total_reviews_encoded,
        actual_device

    ) = encode_selected_reviews(

        input_file=
            input_file,

        selected_review_ids=
            selected_review_ids,

        selected_counts=
            selected_counts,

        entity_id_column=
            entity_id_column,

        review_id_column=
            review_id_column,

        text_column=
            text_column,

        model_name=
            model_name,

        batch_size=
            batch_size,

        chunk_size=
            encoding_chunk_size,

        device=
            device

    )

    # ========================================================
    # SAVE
    # ========================================================

    save_embedding_outputs(

        output_npz=
            output_npz,

        output_index_csv=
            output_index_csv,

        output_metadata_json=
            output_metadata_json,

        entity_ids=
            entity_ids,

        embeddings=
            embeddings,

        review_counts=
            review_counts,

        entity_id_column=
            entity_id_column,

        model_name=
            model_name,

        embedding_dimension=
            embedding_dimension,

        max_reviews_per_entity=
            max_reviews_per_entity,

        random_seed=
            random_seed,

        total_reviews_encoded=
            total_reviews_encoded,

        device=
            actual_device

    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("REVIEW EMBEDDINGS COMPLETE")
    print("=" * 60)

    print()

    print(
        f"Entities embedded: "
        f"{len(entity_ids):,}"
    )

    print(
        f"Reviews encoded:   "
        f"{total_reviews_encoded:,}"
    )

    print(
        f"Embedding shape:   "
        f"{embeddings.shape}"
    )

    print()

    print(
        "Embedding output:"
    )

    print(
        output_npz
    )

    print()

    print(
        "Index output:"
    )

    print(
        output_index_csv
    )

    print()

    print(
        "Metadata output:"
    )

    print(
        output_metadata_json
    )

    return (
        entity_ids,
        embeddings,
        review_counts
    )