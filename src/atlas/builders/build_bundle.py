# src/atlas/builders/build_bundle.py

import pandas as pd

from src.atlas.schema.atlas_bundle import AtlasBundle

from src.atlas.converters.dataframe_to_items import (
    dataframe_to_items
)

from src.atlas.converters.dataframe_to_regions import (
    dataframe_to_regions
)

from src.atlas.converters.dataframe_to_landmarks import (
    dataframe_to_landmarks
)


def build_bundle(
    df: pd.DataFrame,
    domain: str,
    feature_config,
    metadata=None
):

    atlas_items = dataframe_to_items(
        df,
        domain
    )


    regions = dataframe_to_regions(
        df
    )


    landmarks = dataframe_to_landmarks(
        df
    )


    return AtlasBundle(

        domain=domain,

        feature_config=feature_config,

        atlas=atlas_items,

        regions=regions,

        landmarks=landmarks,

        metadata=metadata or {}

    )