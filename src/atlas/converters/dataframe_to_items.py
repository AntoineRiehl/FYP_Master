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



def _get_source_id(row):

    return str(

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



def _get_title(row):

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



def _safe_float(value, default=0):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return default



def _safe_int(value, default=-1):

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):
        return default



def _get_categories(row):

    categories=[]


    if _get(row, "genres"):

        categories.append(
            str(
                _get(row,"genres")
            )
        )


    elif _get(row, "categories"):

        categories.append(
            str(
                _get(row,"categories")
            )
        )


    elif _get(row, "macro_genre"):

        categories.append(
            str(
                _get(row,"macro_genre")
            )
        )


    return categories



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



def dataframe_to_items(
    df: pd.DataFrame,
    domain: str
) -> List[AtlasItem]:


    items=[]


    for _, row in df.iterrows():


        source_id = _get_source_id(
            row
        )


        item = AtlasItem(


            id=source_id,


            source_id=source_id,


            title=_get_title(
                row
            ),


            domain=domain,



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



            statistics=_get_statistics(
                row
            ),



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