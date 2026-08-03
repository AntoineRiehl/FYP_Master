#src/atlas/converters/dataframe_to_landmarks.py

from typing import List

import pandas as pd

from src.atlas.schema.landmark import Landmark



def dataframe_to_landmarks(
    df: pd.DataFrame,
    top_per_cluster:int=5
) -> List[Landmark]:


    landmarks=[]


    for cluster in sorted(
        df.cluster.unique()
    ):


        cluster_df=(

            df[
                df.cluster == cluster
            ]

            .sort_values(
                "visual_size",
                ascending=False
            )

            .head(
                top_per_cluster
            )

        )


        for _, row in cluster_df.iterrows():


            source_id = str(

                row.get(
                    "movieId",

                    row.get(
                        "business_id",

                        row.get(
                            "mbid"
                        )
                    )

                )

            )


            label=(

                row.get(
                    "title"
                )

                or

                row.get(
                    "artist_lastfm"
                )

                or

                row.get(
                    "name"
                )

            )


            landmarks.append(

                Landmark(

                    id=source_id,

                    label=label,

                    x=float(
                        row.umap_x
                    ),

                    y=float(
                        row.umap_y
                    ),

                    importance=float(
                        row.visual_size
                    ),

                    cluster=int(
                        cluster
                    )

                )

            )


    return landmarks