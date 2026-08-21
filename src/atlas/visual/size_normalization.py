#src/atlas/visual/size_normalization

import numpy as np
import pandas as pd


# =========================================================
# DEFAULT CONFIGURATION
# =========================================================

DEFAULT_MIN_SIZE = 4.0
DEFAULT_MAX_SIZE = 80.0
DEFAULT_STRENGTH = 1.8


# =========================================================
# NORMALIZE VISUAL SIZES
# =========================================================

def normalize_visual_sizes(

    df: pd.DataFrame,

    popularity_column: str = "popularity_score",

    min_size: float = DEFAULT_MIN_SIZE,

    max_size: float = DEFAULT_MAX_SIZE,

    strength: float = DEFAULT_STRENGTH

) -> pd.DataFrame:
    """
    Create a normalized visual size for each item.

    The original popularity score is preserved.

    Visual size is intended purely as a visual encoding and
    therefore does not represent the original popularity
    value directly.

    The normalization works across the dataframe supplied
    to this function.

    Therefore:

        Mono-domain atlas:
            normalize_visual_sizes(movies)

        Cross-domain atlas:
            normalize_visual_sizes(combined_movies_music)

    This ensures that the maximum visual size is determined
    by the complete population being visualised.

    Parameters
    ----------
    df:
        Input dataframe.

    popularity_column:
        Column containing the original popularity score.

    min_size:
        Minimum visual size.

    max_size:
        Maximum visual size.

    strength:
        Controls how strongly popularity differences are
        exaggerated visually.

        1.0 = linear
        >1.0 = stronger separation between popular items
        <1.0 = flatter distribution

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with a new 'visual_size'
        column.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )


    if popularity_column not in df.columns:

        raise ValueError(
            f"Popularity column '{popularity_column}' "
            "was not found in the dataframe."
        )


    if min_size <= 0:

        raise ValueError(
            "min_size must be greater than zero."
        )


    if max_size <= min_size:

        raise ValueError(
            "max_size must be greater than min_size."
        )


    if strength <= 0:

        raise ValueError(
            "strength must be greater than zero."
        )


    result = df.copy()


    # =====================================================
    # NUMERIC POPULARITY
    # =====================================================

    popularity = pd.to_numeric(

        result[popularity_column],

        errors="coerce"

    )


    # -----------------------------------------------------
    # HANDLE MISSING VALUES
    # -----------------------------------------------------

    if popularity.notna().sum() == 0:

        result["visual_size"] = min_size

        return result


    # =====================================================
    # PERCENTILE RANK
    # =====================================================

    # Percentile ranking makes the visual scale depend on
    # relative popularity rather than raw popularity units.
    #
    # Example:
    #
    #   0.00 → least popular
    #   0.50 → middle
    #   1.00 → most popular

    percentile = (

        popularity
        .rank(
            method="average",
            pct=True
        )

    )


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    percentile = percentile.fillna(0)


    # =====================================================
    # APPLY VISUAL STRENGTH
    # =====================================================

    # Values are first converted to [0, 1].
    #
    # A power > 1 exaggerates the difference towards the
    # popular end while keeping the scale bounded.

    scaled = np.power(

        percentile,

        strength

    )


    # =====================================================
    # MAP TO VISUAL SIZE
    # =====================================================

    result["visual_size"] = (

        min_size

        +

        scaled
        *
        (
            max_size
            -
            min_size
        )

    )


    # =====================================================
    # FINAL CLEANUP
    # =====================================================

    result["visual_size"] = (

        result["visual_size"]
        .astype(float)
        .clip(
            lower=min_size,
            upper=max_size
        )

    )


    return result