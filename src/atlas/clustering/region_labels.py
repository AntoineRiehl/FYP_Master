# src/atlas/clustering/region_labels.py

import re

import pandas as pd


# =========================================================
# TEXT UTILITIES
# =========================================================

def _clean_label_text(
    text
) -> str:
    """
    Clean text before extracting representative
    words for a cluster label.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# GENERIC STOP WORDS
# =========================================================

GENERIC_LABEL_STOPWORDS = {

    # Common English words
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "for",
    "into",
    "about",
    "their",
    "they",
    "them",
    "there",
    "which",
    "when",
    "where",
    "while",
    "through",

}


# =========================================================
# EXTRACT TOP TEXT TERMS
# =========================================================

def _get_top_text_terms(
    cluster_df: pd.DataFrame,
    text_column: str,
    n_terms: int = 2
):
    """
    Extract the most frequent meaningful words from
    a cluster's textual representation.
    """

    if text_column not in cluster_df.columns:
        return []

    word_counts = {}

    for text in cluster_df[text_column]:

        text = _clean_label_text(
            text
        )

        if not text:
            continue

        words = text.split()

        for word in words:

            if (
                not word
                or word in GENERIC_LABEL_STOPWORDS
                or len(word) < 3
            ):
                continue

            word_counts[word] = (
                word_counts.get(
                    word,
                    0
                )
                + 1
            )

    if not word_counts:
        return []

    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    return [
        word
        for word, _ in sorted_words[:n_terms]
    ]


# =========================================================
# DETERMINE LABEL SOURCE
# =========================================================

def _get_label_from_cluster(
    cluster_df: pd.DataFrame
):
    """
    Determine the best available information to use
    when creating a region label.

    Priority:

        1. macro_genre
        2. semantic_text
        3. genres
        4. categories
        5. tags_text
    """

    # =====================================================
    # MACRO GENRE
    # =====================================================

    if "macro_genre" in cluster_df.columns:

        values = (
            cluster_df["macro_genre"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        if not values.empty:

            top_genres = (
                values
                .value_counts()
                .head(2)
                .index
                .tolist()
            )

            if top_genres:

                return " / ".join(
                    top_genres
                )

    # =====================================================
    # SEMANTIC TEXT
    # =====================================================

    if "semantic_text" in cluster_df.columns:

        terms = _get_top_text_terms(
            cluster_df,
            "semantic_text",
            n_terms=2
        )

        if terms:

            return " / ".join(
                terms
            )

    # =====================================================
    # GENRES
    # =====================================================

    if "genres" in cluster_df.columns:

        terms = _get_top_text_terms(
            cluster_df,
            "genres",
            n_terms=2
        )

        if terms:

            return " / ".join(
                terms
            )

    # =====================================================
    # CATEGORIES
    # =====================================================

    if "categories" in cluster_df.columns:

        terms = _get_top_text_terms(
            cluster_df,
            "categories",
            n_terms=2
        )

        if terms:

            return " / ".join(
                terms
            )

    # =====================================================
    # TAGS
    # =====================================================

    if "tags_text" in cluster_df.columns:

        terms = _get_top_text_terms(
            cluster_df,
            "tags_text",
            n_terms=2
        )

        if terms:

            return " / ".join(
                terms
            )

    # =====================================================
    # FALLBACK
    # =====================================================

    return "Unknown"


# =========================================================
# CREATE REGION LABELS
# =========================================================

def create_region_labels(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create human-readable labels for atlas clusters.

    This function is domain agnostic and works for both
    mono-domain and cross-domain atlases.

    Mono-domain examples:

        Movies
        Music
        Restaurants

    Cross-domain examples:

        Movies + Music
        Movies + Music + Restaurants

    The function prefers an existing 'macro_genre'
    column when available. Cross-domain datasets
    normally do not contain this column, so the function
    automatically falls back to semantic text.

    Parameters
    ----------
    df:
        Atlas dataframe containing a 'cluster' column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe containing:

            cluster_label
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if "cluster" not in df.columns:

        raise ValueError(
            "Dataframe must contain a "
            "'cluster' column."
        )

    result = df.copy()

    cluster_labels = {}

    clusters = sorted(
        result["cluster"]
        .dropna()
        .unique()
    )

    # =====================================================
    # PROCESS EACH CLUSTER
    # =====================================================

    for cluster in clusters:

        cluster_df = result[
            result["cluster"] == cluster
        ]

        cluster_labels[cluster] = (
            _get_label_from_cluster(
                cluster_df
            )
        )

    # =====================================================
    # MAP LABELS
    # =====================================================

    result["cluster_label"] = (
        result["cluster"]
        .map(cluster_labels)
    )

    return result