#src/atlas/cross_domain/semantic_text.py

import re

import pandas as pd


# =========================================================
# TEXT CLEANING
# =========================================================

def _clean_text(
    text
) -> str:
    """
    Clean a text value before semantic processing.

    The cleaning is intentionally lightweight.
    We do not want to remove meaningful descriptive
    information at this stage.
    """

    if pd.isna(text):

        return ""


    text = str(text)


    # -----------------------------------------------------
    # NORMALISE CASE
    # -----------------------------------------------------

    text = text.lower()


    # -----------------------------------------------------
    # NORMALISE SEPARATORS
    # -----------------------------------------------------

    text = text.replace(
        ";",
        " "
    )

    text = text.replace(
        "|",
        " "
    )

    text = text.replace(
        ",",
        " "
    )


    # -----------------------------------------------------
    # REMOVE UNNECESSARY PUNCTUATION
    # -----------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )


    # -----------------------------------------------------
    # NORMALISE WHITESPACE
    # -----------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# =========================================================
# DOMAIN VOCABULARY FILTERING
# =========================================================

def _remove_domain_vocabulary(
    text: str,
    domain_vocabulary: set[str]
) -> str:
    """
    Remove domain-specific words from semantic text.

    The purpose is to reduce the influence of words such
    as 'movie', 'film', 'actor', 'artist', 'album',
    'restaurant', etc.

    Only complete words are removed.
    """

    if not text:

        return ""


    if not domain_vocabulary:

        return text


    words = text.split()


    filtered_words = [

        word

        for word in words

        if word not in domain_vocabulary

    ]


    return " ".join(
        filtered_words
    )


# =========================================================
# TEXT SOURCES
# =========================================================

def _get_text_column(
    df: pd.DataFrame,
    column: str
) -> pd.Series:
    """
    Safely retrieve a text column.

    Missing columns are represented by empty strings.
    """

    if column not in df.columns:

        return pd.Series(
            "",
            index=df.index,
            dtype="object"
        )


    return (
        df[column]
        .fillna("")
        .astype(str)
    )


# =========================================================
# BUILD SEMANTIC TEXT
# =========================================================

def create_semantic_text(
    df: pd.DataFrame,
    domain_vocabulary: set[str] | None = None
) -> pd.DataFrame:
    """
    Create a domain-neutral semantic text representation.

    The function combines the textual information already
    prepared by the individual domain pipelines.

    Current supported textual sources:

        tags_text
        genres
        categories
        tip_text

    Missing columns are ignored automatically.

    Parameters
    ----------
    df:
        Combined cross-domain dataframe.

    domain_vocabulary:
        Words that should be removed because they describe
        the item's domain rather than its semantic character.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with a new:

            semantic_text

        column.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )


    result = df.copy()


    if domain_vocabulary is None:

        domain_vocabulary = set()


    # =====================================================
    # TEXT SOURCES
    # =====================================================

    text_sources = [

        "tags_text",

        "genres",

        "categories",

        "tip_text",

    ]


    text_parts = []


    for column in text_sources:

        series = _get_text_column(
            result,
            column
        )

        text_parts.append(
            series
        )


    # =====================================================
    # COMBINE TEXT
    # =====================================================

    semantic_text = pd.Series(
        "",
        index=result.index,
        dtype="object"
    )


    for series in text_parts:

        semantic_text = (
            semantic_text
            + " "
            + series
        )


    # =====================================================
    # CLEAN TEXT
    # =====================================================

    semantic_text = (
        semantic_text
        .apply(_clean_text)
    )


    # =====================================================
    # REMOVE DOMAIN VOCABULARY
    # =====================================================

    semantic_text = (
        semantic_text
        .apply(
            lambda text:
                _remove_domain_vocabulary(
                    text,
                    domain_vocabulary
                )
        )
    )


    # =====================================================
    # REMOVE DUPLICATE WORDS
    # =====================================================

    # We preserve the first occurrence of each word.
    #
    # This prevents a word appearing repeatedly across
    # several source columns from receiving excessive
    # importance simply because it was duplicated.

    def _deduplicate_words(
        text: str
    ) -> str:

        words = text.split()

        seen = set()

        unique_words = []


        for word in words:

            if word not in seen:

                unique_words.append(
                    word
                )

                seen.add(
                    word
                )


        return " ".join(
            unique_words
        )


    semantic_text = (
        semantic_text
        .apply(
            _deduplicate_words
        )
    )


    # =====================================================
    # STORE RESULT
    # =====================================================

    result["semantic_text"] = (
        semantic_text
    )


    return result