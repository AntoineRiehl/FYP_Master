# src/atlas/cross_domain/combine_data.py

import pandas as pd


# =========================================================
# SOURCE ID
# =========================================================

def _get_source_id(
    df: pd.DataFrame
) -> pd.Series:
    """
    Extract the domain-specific source identifier
    and expose it through the common 'source_id' column.

    Supported domains:

        Movies       -> movieId
        Music        -> mbid
        Restaurants  -> business_id

    The original identifier is preserved as a string.
    """

    if "movieId" in df.columns:

        return (
            df["movieId"]
            .astype(str)
        )

    if "mbid" in df.columns:

        return (
            df["mbid"]
            .astype(str)
        )

    if "business_id" in df.columns:

        return (
            df["business_id"]
            .astype(str)
        )

    if "source_id" in df.columns:

        return (
            df["source_id"]
            .astype(str)
        )

    raise ValueError(
        "Could not determine source ID. "
        "Expected one of: "
        "'movieId', 'mbid', 'business_id', "
        "or 'source_id'."
    )


# =========================================================
# PREPARE DOMAIN DATAFRAME
# =========================================================

def _prepare_domain_dataframe(
    df: pd.DataFrame,
    domain: str
) -> pd.DataFrame:
    """
    Prepare one domain dataframe for combination.

    Adds the following standardised columns:

        source_id
        domain
        id

    Important:
        'domain' represents the ORIGINAL domain of the item.

        For example:

            movie     -> movies
            artist    -> music
            restaurant -> restaurants

        It does NOT represent the cross-domain atlas.

    The original domain-specific columns are preserved.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            f"Expected a pandas DataFrame "
            f"for domain '{domain}', "
            f"got {type(df).__name__}."
        )


    if df.empty:

        raise ValueError(
            f"The dataframe for domain "
            f"'{domain}' is empty."
        )


    df = df.copy()


    # -----------------------------------------------------
    # VALIDATE DOMAIN
    # -----------------------------------------------------

    domain = domain.strip().lower()


    if not domain:

        raise ValueError(
            "Domain name cannot be empty."
        )


    # -----------------------------------------------------
    # SOURCE ID
    # -----------------------------------------------------

    df["source_id"] = (
        _get_source_id(df)
    )


    # -----------------------------------------------------
    # ORIGINAL DOMAIN
    # -----------------------------------------------------

    # IMPORTANT:
    #
    # This is the item's actual domain.
    #
    # It must remain:
    #
    #     movies
    #     music
    #     restaurants
    #
    # even when the dataframe is later used to
    # construct a cross-domain atlas.
    #
    # The cross-domain nature of the atlas is represented
    # by the pipeline / bundle configuration, NOT here.

    df["domain"] = domain


    # -----------------------------------------------------
    # GLOBAL ATLAS ID
    # -----------------------------------------------------

    # Prefixing the source ID with the ORIGINAL domain
    # prevents collisions between domains.
    #
    # Examples:
    #
    #     movies:123
    #     music:123
    #     restaurants:123

    df["id"] = (
        df["domain"]
        + ":"
        + df["source_id"]
    )


    return df


# =========================================================
# COMBINE DOMAIN DATA
# =========================================================

def combine_domain_data(
    domain_data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine any number of domain-specific dataframes
    into one cross-domain dataframe.

    Parameters
    ----------
    domain_data:
        Dictionary mapping ORIGINAL domain names to
        preprocessed dataframes.

        Example:

            {
                "movies": movies_df,
                "music": music_df,
                "restaurants": restaurants_df
            }

    Returns
    -------
    pd.DataFrame
        A single dataframe containing all supplied
        domains with standardised:

            id
            source_id
            domain

        columns.

    Important
    ---------
    The 'domain' column ALWAYS represents the original
    domain of each item.

    Therefore, for a Movies + Music atlas:

        movie  -> domain = "movies"
        artist -> domain = "music"

    and NOT:

        movie  -> domain = "movies_music"
        artist -> domain = "movies_music"

    This allows the frontend and later analysis to inspect
    domain membership independently from the semantic atlas.

    Notes
    -----
    This function deliberately does NOT perform:

        - semantic text construction
        - domain vocabulary filtering
        - embeddings
        - dimensionality reduction
        - clustering

    Those operations belong to later stages of the
    cross-domain pipeline.
    """

    if not isinstance(
        domain_data,
        dict
    ):

        raise TypeError(
            "domain_data must be a dictionary "
            "mapping domain names to pandas DataFrames."
        )


    if not domain_data:

        raise ValueError(
            "domain_data cannot be empty."
        )


    prepared_dataframes = []


    # =====================================================
    # PREPARE EACH DOMAIN
    # =====================================================

    for domain, df in domain_data.items():

        if not isinstance(
            domain,
            str
        ) or not domain.strip():

            raise ValueError(
                "Every domain must have a "
                "non-empty string name."
            )


        prepared = _prepare_domain_dataframe(
            df,
            domain
        )


        prepared_dataframes.append(
            prepared
        )


    # =====================================================
    # COMBINE
    # =====================================================

    combined = pd.concat(
        prepared_dataframes,
        axis=0,
        ignore_index=True,
        sort=False
    )


    # =====================================================
    # VALIDATE GLOBAL IDS
    # =====================================================

    if combined["id"].duplicated().any():

        duplicates = (
            combined.loc[
                combined["id"].duplicated(
                    keep=False
                ),
                "id"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate global atlas IDs detected: "
            f"{duplicates[:10]}"
        )


    # =====================================================
    # VALIDATE DOMAINS
    # =====================================================

    if combined["domain"].isna().any():

        raise ValueError(
            "One or more rows have a missing domain."
        )


    # =====================================================
    # VALIDATE EXPECTED DOMAIN VALUES
    # =====================================================

    invalid_domains = (
        combined["domain"]
        .astype(str)
        .str.strip()
        .eq("")
    )


    if invalid_domains.any():

        raise ValueError(
            "One or more rows contain an empty "
            "domain value."
        )


    # =====================================================
    # RESET INDEX
    # =====================================================

    combined = combined.reset_index(
        drop=True
    )


    return combined