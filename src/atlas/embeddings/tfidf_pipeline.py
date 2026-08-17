#src/atlas/embeddings/tfidf_pipeline.py

from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT
    / "models"
    / "embeddings"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEFAULT CONFIGURATION
# =========================================================

DEFAULT_MAX_FEATURES = 5000


# =========================================================
# VECTORISER PATH
# =========================================================

def _get_vectorizer_path(
    model_name: str
) -> Path:
    """
    Return the storage path for a specific TF-IDF
    vectorizer.

    Each atlas gets its own vectorizer so that
    vocabularies trained for different semantic spaces
    are never accidentally mixed.
    """

    safe_name = (
        model_name
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return (
        MODEL_DIR
        / f"tfidf_{safe_name}.pkl"
    )


# =========================================================
# VALIDATE INPUT
# =========================================================

def _validate_input(
    df,
    text_column: str
):
    """
    Validate the dataframe and requested text column.
    """

    if df is None:

        raise ValueError(
            "df cannot be None."
        )


    if text_column not in df.columns:

        raise ValueError(
            f"Column '{text_column}' "
            f"was not found in the dataframe. "
            f"Available columns: "
            f"{list(df.columns)}"
        )


# =========================================================
# PREPARE TEXT
# =========================================================

def _prepare_text(
    df,
    text_column: str
):
    """
    Prepare the selected text column for TF-IDF.

    Missing values are converted to empty strings.
    """

    return (
        df[text_column]
        .fillna("")
        .astype(str)
    )


# =========================================================
# BUILD OR LOAD TF-IDF
# =========================================================

def get_tfidf_embeddings(
    df,
    text_column: str = "tags_text",
    model_name: str = "default",
    max_features: int = DEFAULT_MAX_FEATURES
):
    """
    Build or load a TF-IDF vectorizer and transform
    the supplied dataframe.

    Parameters
    ----------
    df:
        Dataframe containing the text representation.

    text_column:
        Column containing the text to embed.

        Examples:
            "tags_text"
            "semantic_text"

    model_name:
        Unique name identifying the semantic space.

        Examples:
            "movies_tags"
            "music_tags"
            "restaurants_tags"
            "movies_music_semantic"
            "all_domains_semantic"

    max_features:
        Maximum number of TF-IDF features.

    Returns
    -------
    tfidf_matrix, vectorizer
    """


    # =====================================================
    # VALIDATION
    # =====================================================

    _validate_input(
        df,
        text_column
    )


    if not isinstance(
        model_name,
        str
    ) or not model_name.strip():

        raise ValueError(
            "model_name must be a "
            "non-empty string."
        )


    if not isinstance(
        max_features,
        int
    ) or max_features <= 0:

        raise ValueError(
            "max_features must be "
            "a positive integer."
        )


    # =====================================================
    # PREPARE TEXT
    # =====================================================

    text = _prepare_text(
        df,
        text_column
    )


    # =====================================================
    # VECTORISER PATH
    # =====================================================

    tfidf_path = _get_vectorizer_path(
        model_name
    )


    # =====================================================
    # LOAD EXISTING VECTORISER
    # =====================================================

    if tfidf_path.exists():

        print(
            f"Loading existing TF-IDF "
            f"vectorizer: {model_name}"
        )

        vectorizer = joblib.load(
            tfidf_path
        )


        # -------------------------------------------------
        # BASIC COMPATIBILITY CHECK
        # -------------------------------------------------

        if not isinstance(
            vectorizer,
            TfidfVectorizer
        ):

            raise TypeError(
                f"Stored object at "
                f"{tfidf_path} is not a "
                f"TfidfVectorizer."
            )


    # =====================================================
    # TRAIN NEW VECTORISER
    # =====================================================

    else:

        print(
            f"Training new TF-IDF "
            f"vectorizer: {model_name}"
        )

        vectorizer = TfidfVectorizer(

            stop_words="english",

            max_features=max_features

        )


        vectorizer.fit(
            text
        )


        joblib.dump(
            vectorizer,
            tfidf_path
        )


        print(
            f"TF-IDF vectorizer saved: "
            f"{tfidf_path.name}"
        )


    # =====================================================
    # TRANSFORM TEXT
    # =====================================================

    tfidf_matrix = (
        vectorizer.transform(
            text
        )
    )


    # =====================================================
    # INFORMATION
    # =====================================================

    print(
        "TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )


    return (
        tfidf_matrix,
        vectorizer
    )