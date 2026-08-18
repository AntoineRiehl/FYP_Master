# src/atlas/converters/dataframe_to_items.py

from typing import List

import pandas as pd

from src.atlas.schema.atlas_item import (
    AtlasItem,
    AtlasPosition,
    AtlasVisual,
    MediaMetadata,
    Statistics,
    TextFeatures
)


# =========================================================
# GENERIC VALUE GETTER
# =========================================================

def _get(
    row,
    column,
    default=None
):

    if column in row.index:

        value = row[column]

        if pd.isna(value):

            return default

        return value

    return default


# =========================================================
# SOURCE ID
# =========================================================

def _get_source_id(
    row
):

    return str(

        _get(
            row,
            "source_id",
            _get(
                row,
                "movieId",
                _get(
                    row,
                    "business_id",
                    _get(
                        row,
                        "mbid",
                        ""
                    )
                )
            )
        )

    )


# =========================================================
# ITEM DOMAIN
# =========================================================

def _get_item_domain(
    row,
    atlas_domain
):
    """
    Determine the original domain of an individual item.

    For cross-domain atlases, the dataframe contains a
    'domain' column created by combine_domain_data().

    For mono-domain atlases, the dataframe may not contain
    an explicit domain column, so the atlas domain is used
    as the fallback.

    Important distinction:

        atlas_domain
            = identity of the complete atlas

        item_domain
            = original domain of this individual item
    """

    item_domain = _get(
        row,
        "domain"
    )

    if item_domain is not None:

        return str(
            item_domain
        )

    return str(
        atlas_domain
    )


# =========================================================
# TITLE
# =========================================================

def _get_title(
    row
):

    return (

        _get(
            row,
            "title"
        )

        or

        _get(
            row,
            "artist_lastfm"
        )

        or

        _get(
            row,
            "name"
        )

        or

        "Unknown"

    )


# =========================================================
# SAFE FLOAT
# =========================================================

def _safe_float(
    value,
    default=0
):

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


# =========================================================
# SAFE INTEGER
# =========================================================

def _safe_int(
    value,
    default=-1
):

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


# =========================================================
# CATEGORIES
# =========================================================

def _get_categories(
    row
):

    categories = []


    if _get(
        row,
        "genres"
    ):

        categories.append(
            str(
                _get(
                    row,
                    "genres"
                )
            )
        )


    elif _get(
        row,
        "categories"
    ):

        categories.append(
            str(
                _get(
                    row,
                    "categories"
                )
            )
        )


    elif _get(
        row,
        "macro_genre"
    ):

        categories.append(
            str(
                _get(
                    row,
                    "macro_genre"
                )
            )
        )


    return categories


# =========================================================
# STATISTICS
# =========================================================

def _get_statistics(
    row
):

    rating = (

        _get(
            row,
            "avg_rating"
        )

        or

        _get(
            row,
            "stars"
        )

    )


    rating_count = (

        _get(
            row,
            "rating_count"
        )

        or

        _get(
            row,
            "review_count"
        )

    )


    popularity = (

        _get(
            row,
            "popularity"
        )

        or

        _get(
            row,
            "popularity_score"
        )

    )


    return Statistics(

        rating=rating,

        rating_count=rating_count,

        popularity=popularity

    )


# =========================================================
# DATAFRAME → ATLAS ITEMS
# =========================================================

def dataframe_to_items(
    df: pd.DataFrame,
    domain: str
) -> List[AtlasItem]:
    """
    Convert a dataframe into AtlasItem objects.

    Parameters
    ----------
    df:
        Atlas dataframe containing the processed items.

    domain:
        Domain/identity of the complete atlas.

        Examples:

            "movies"
            "music"
            "restaurants"
            "movies_music"
            "movies_music_restaurants"

    Important
    ---------
    The supplied 'domain' identifies the ATLAS itself.

    Individual item domains are taken from the dataframe's
    'domain' column when available.

    Therefore:

        AtlasBundle.domain
            -> "movies_music"

        AtlasItem.domain
            -> "movies" or "music"

    This allows cross-domain atlases to preserve original
    domain membership without using domain as a semantic
    feature.
    """

    items = []


    # =====================================================
    # VALIDATION
    # =====================================================

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "df must be a pandas DataFrame."
        )


    if not isinstance(
        domain,
        str
    ) or not domain.strip():

        raise ValueError(
            "domain must be a non-empty string."
        )


    # =====================================================
    # CONVERT ROWS
    # =====================================================

    for _, row in df.iterrows():


        # -------------------------------------------------
        # IDENTIFIERS
        # -------------------------------------------------

        source_id = _get_source_id(
            row
        )


        item_domain = _get_item_domain(
            row,
            domain
        )


        # -------------------------------------------------
        # ITEM
        # -------------------------------------------------

        item = AtlasItem(

            # -------------------------------------------------
            # GLOBAL ITEM ID
            # -------------------------------------------------
            #
            # Cross-domain dataframes created by
            # combine_domain_data() already contain a
            # globally unique 'id' such as:
            #
            #     movies:123
            #     music:abc
            #     restaurants:xyz
            #
            # We preserve it when available.
            #
            id=str(
                _get(
                    row,
                    "id",
                    source_id
                )
            ),


            # -------------------------------------------------
            # SOURCE ID
            # -------------------------------------------------

            source_id=source_id,


            # -------------------------------------------------
            # TITLE
            # -------------------------------------------------

            title=_get_title(
                row
            ),


            # -------------------------------------------------
            # ORIGINAL ITEM DOMAIN
            # -------------------------------------------------

            domain=item_domain,


            # -------------------------------------------------
            # METADATA
            # -------------------------------------------------

            metadata=MediaMetadata(

                year=_get(
                    row,
                    "year"
                ),

                country=_get(
                    row,
                    "country"
                ),

                artist=_get(
                    row,
                    "artist_lastfm"
                ),

                album=_get(
                    row,
                    "album"
                ),

                address=_get(
                    row,
                    "address"
                ),

                city=_get(
                    row,
                    "city"
                ),

                latitude=_get(
                    row,
                    "latitude"
                ),

                longitude=_get(
                    row,
                    "longitude"
                )

            ),


            # -------------------------------------------------
            # TEXT FEATURES
            # -------------------------------------------------

            text=TextFeatures(

                tags=(

                    [

                        str(
                            _get(
                                row,
                                "tags_text"
                            )
                        )

                    ]

                    if _get(
                        row,
                        "tags_text"
                    )

                    else []

                ),

                categories=_get_categories(
                    row
                )

            ),


            # -------------------------------------------------
            # STATISTICS
            # -------------------------------------------------

            statistics=_get_statistics(
                row
            ),


            # -------------------------------------------------
            # POSITION
            # -------------------------------------------------

            position=AtlasPosition(

                x=_safe_float(
                    _get(
                        row,
                        "umap_x",
                        0
                    )
                ),

                y=_safe_float(
                    _get(
                        row,
                        "umap_y",
                        0
                    )
                )

            ),


            # -------------------------------------------------
            # VISUAL INFORMATION
            # -------------------------------------------------

            visual=AtlasVisual(

                size=_safe_float(
                    _get(
                        row,
                        "visual_size",
                        1
                    ),
                    1
                ),

                cluster=_safe_int(
                    _get(
                        row,
                        "cluster",
                        -1
                    )
                ),

                cluster_label=_get(
                    row,
                    "cluster_label"
                )

            )

        )


        items.append(
            item
        )


    return items