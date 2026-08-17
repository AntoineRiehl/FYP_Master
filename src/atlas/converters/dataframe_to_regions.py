#src/atlas/converters/dataframe_to_regions.py

from typing import List

import pandas as pd

from src.atlas.schema.region import Region



def dataframe_to_regions(
    df: pd.DataFrame
) -> List[Region]:


    grouped = (

        df

        .groupby(
            "cluster"
        )

        .agg(

            x=(
                "umap_x",
                "mean"
            ),

            y=(
                "umap_y",
                "mean"
            ),

            label=(
                "cluster_label",
                "first"
            )
            if "cluster_label" in df.columns
            else (
                "cluster",
                "first"
            ),

            item_count=(
                "cluster",
                "count"
            ),

            size=(
                "visual_size",
                "mean"
            )

        )

        .reset_index()

    )


    regions=[]


    for _, row in grouped.iterrows():

        regions.append(

            Region(

                id=int(row.cluster),

                label=str(row.label),

                x=float(row.x),

                y=float(row.y),

                size=float(row.size),

                item_count=int(row.item_count)

            )

        )


    return regions