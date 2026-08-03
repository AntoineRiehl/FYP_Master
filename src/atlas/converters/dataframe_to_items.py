#src/atlas/converters/dataframe_to_items.py

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



def _get(row, column, default=None):

    if column in row.index:
        return row[column]

    return default



def dataframe_to_items(
    df: pd.DataFrame
) -> List[AtlasItem]:


    items=[]


    for _, row in df.iterrows():


        source_id = str(
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


        title = (
            _get(row,"title")
            or
            _get(row,"artist_lastfm")
            or
            _get(row,"name")
            or
            "Unknown"
        )


        item = AtlasItem(

            id=source_id,

            source_id=source_id,

            title=title,

            domain=_get(
                row,
                "domain",
                "unknown"
            ),


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

                tags=[

                    _get(
                        row,
                        "tags_text",
                        ""
                    )

                ]
                if _get(row,"tags_text")
                else [],


                categories=[

                    _get(
                        row,
                        "genres",
                        ""
                    )

                ]
                if _get(row,"genres")
                else []

            ),


            statistics=Statistics(

                rating=_get(
                    row,
                    "avg_rating"
                ),

                rating_count=_get(
                    row,
                    "rating_count"
                ),

                popularity=_get(
                    row,
                    "popularity"
                )

            ),



            position=AtlasPosition(

                x=float(
                    _get(
                        row,
                        "umap_x",
                        0
                    )
                ),

                y=float(
                    _get(
                        row,
                        "umap_y",
                        0
                    )
                )

            ),



            visual=AtlasVisual(

                size=float(
                    _get(
                        row,
                        "visual_size",
                        1
                    )
                ),

                cluster=int(
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


        items.append(item)


    return items