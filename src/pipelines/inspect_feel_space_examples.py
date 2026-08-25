# src/pipelines/inspect_feel_space_examples.py

# ============================================================
# QUALITATIVE INSPECTION OF THE SHARED FEEL SPACE
# ============================================================
#
# Purpose
# -------
#
# Perform a lightweight qualitative sanity check of the
# already-generated 13D Shared Experiential / Feel Space.
#
#
# This script:
#
#     1. Loads the existing feel_space_scores.csv
#
#     2. Reconnects source IDs to human-readable names:
#
#            Movies       -> movie titles
#            Music        -> artist names
#            Restaurants  -> business names
#
#     3. Finds highest / lowest scoring examples for every
#        Feel dimension:
#
#            globally
#            within Movies
#            within Music
#            within Restaurants
#
#     4. Finds a small set of recognisable entities and
#        prints their full 13-dimensional profiles.
#
#
# IMPORTANT
# ---------
#
# This script does NOT:
#
#     - recompute SBERT embeddings
#     - recompute review fusion
#     - recompute anchor projection
#     - run UMAP
#     - modify the Feel Space
#
# It is purely a qualitative diagnostic.
#
# ============================================================


from pathlib import Path

import json
import re
import unicodedata

import numpy as np
import pandas as pd


from src.atlas.cross_domain.feel_space.feel_anchors import (
    FEEL_DIMENSIONS,
    BIPOLAR_AXES,
    UNIPOLAR_AXES
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Existing Feel Space
# ------------------------------------------------------------

FEEL_SPACE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "feel_space"
    / "movies_music_restaurants"
    / "feel_space_scores.csv"
)


# ------------------------------------------------------------
# Movie name sources
#
# Prefer the processed file because it corresponds closely
# to the current atlas population.
# ------------------------------------------------------------

MOVIE_FILE_CANDIDATES = [

    ROOT
    / "data"
    / "processed"
    / "movies"
    / "movies_prepared.csv",

    ROOT
    / "data"
    / "raw"
    / "movies"
    / "movies.csv",

]


# ------------------------------------------------------------
# Music name sources
# ------------------------------------------------------------

MUSIC_FILE_CANDIDATES = [

    ROOT
    / "data"
    / "data_scraping"
    / "music_data.csv",

    ROOT
    / "data"
    / "raw"
    / "music"
    / "music_data.csv",

]


# ------------------------------------------------------------
# Yelp businesses
# ------------------------------------------------------------

RESTAURANT_BUSINESS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "restaurants"
    / "business.json"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "feel_space"
    / "qualitative"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EXTREMES_OUTPUT = (
    OUTPUT_DIR
    / "dimension_extremes_named.csv"
)


KNOWN_PROFILES_OUTPUT = (
    OUTPUT_DIR
    / "known_item_profiles.csv"
)


NAME_RESOLUTION_OUTPUT = (
    OUTPUT_DIR
    / "name_resolution_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# How many entities to retain at each extreme.
# ------------------------------------------------------------

N_EXTREMES = 5


# ------------------------------------------------------------
# Recognisable entities used for sanity checking.
#
# These are NOT evaluation labels.
#
# They simply help us inspect whether the generated semantic
# profiles look plausible.
# ------------------------------------------------------------

KNOWN_EXAMPLES = {

    "movies": [

        "Toy Story",

        "The Dark Knight",

        "Titanic",

        "The Shining",

        "La La Land",

        "Mad Max Fury Road",

        "Schindler's List",

        "The Notebook",

    ],


    "music": [

        "Coldplay",

        "Radiohead",

        "Eminem",

        "Rihanna",

        "Taylor Swift",

        "Metallica",

        "Enya",

        "Daft Punk",

    ],


    "restaurants": [

        "Starbucks",

        "McDonald's",

        "Chipotle",

        "The Cheesecake Factory",

        "Panda Express",

    ],

}


# ============================================================
# PRINT SECTION
# ============================================================


def section(
    title
):

    print()

    print(
        "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    print()


# ============================================================
# SOURCE ID NORMALISATION
# ============================================================


def normalise_source_id(
    value
):
    """
    Convert heterogeneous source IDs into stable strings.

    In particular:

        123
        123.0
        "123"

    all become:

        "123"
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    if re.fullmatch(

        r"[+-]?\d+\.0+",

        text

    ):

        return text.split(
            "."
        )[0]


    return text


# ============================================================
# NAME NORMALISATION
# ============================================================


def normalise_name(
    value
):
    """
    Create a forgiving form used only for matching known
    examples.

    This is NOT used in the semantic model.
    """

    if pd.isna(
        value
    ):

        return ""


    text = str(
        value
    ).strip()


    # --------------------------------------------------------
    # Remove MovieLens-style trailing years:
    #
    #     Toy Story (1995)
    #
    # becomes:
    #
    #     Toy Story
    # --------------------------------------------------------

    text = re.sub(

        r"\s*\(\d{4}\)\s*$",

        "",

        text

    )


    text = unicodedata.normalize(

        "NFKD",

        text

    )


    text = "".join(

        character

        for character in text

        if not unicodedata.combining(
            character
        )

    )


    text = text.lower()


    text = re.sub(

        r"[^a-z0-9]+",

        " ",

        text

    )


    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# FIND EXISTING FILE
# ============================================================


def first_existing_file(
    paths,
    label
):

    for path in paths:

        if path.exists():

            return path


    formatted = "\n".join(

        f"  - {path}"

        for path in paths

    )


    raise FileNotFoundError(

        f"Could not find a usable {label} file.\n\n"
        f"Checked:\n{formatted}"

    )


# ============================================================
# FIND COLUMN
# ============================================================


def find_column(

    columns,

    candidates,

    label

):
    """
    Find the first available candidate column.
    """

    lookup = {

        str(column).lower():
            column

        for column in columns

    }


    for candidate in candidates:

        key = candidate.lower()


        if key in lookup:

            return lookup[
                key
            ]


    raise ValueError(

        f"Could not identify {label} column.\n\n"

        f"Available columns:\n"
        f"{list(columns)}"

    )


# ============================================================
# LOAD FEEL SPACE
# ============================================================


def load_feel_space():

    section(
        "LOADING EXISTING FEEL SPACE"
    )


    if not FEEL_SPACE_FILE.exists():

        raise FileNotFoundError(

            "Feel-space score file was not found:\n"
            f"{FEEL_SPACE_FILE}\n\n"

            "Run evaluate_feel_space.py first."

        )


    feel_df = pd.read_csv(

        FEEL_SPACE_FILE,

        dtype={

            "domain":
                str,

            "source_id":
                str,

            "id":
                str,

        }

    )


    required_columns = {

        "domain",

        "source_id",

        "is_semantically_defined",

        *FEEL_DIMENSIONS,

    }


    missing = (

        required_columns

        -

        set(
            feel_df.columns
        )

    )


    if missing:

        raise ValueError(

            "Feel-space CSV is missing required columns:\n"
            f"{sorted(missing)}"

        )


    feel_df[
        "source_id_key"
    ] = (

        feel_df[
            "source_id"
        ]
        .apply(
            normalise_source_id
        )

    )


    # --------------------------------------------------------
    # Handle bool values robustly if CSV parsing returns text.
    # --------------------------------------------------------

    if (

        feel_df[
            "is_semantically_defined"
        ].dtype
        !=
        bool

    ):

        feel_df[
            "is_semantically_defined"
        ] = (

            feel_df[
                "is_semantically_defined"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )

        )


    print(
        f"Entities loaded:          "
        f"{len(feel_df):,}"
    )


    defined_count = int(

        feel_df[
            "is_semantically_defined"
        ].sum()

    )


    print(
        f"Semantically defined:    "
        f"{defined_count:,}"
    )


    print(
        f"Semantically undefined:  "
        f"{len(feel_df) - defined_count:,}"
    )


    print()


    print(
        "Entities by domain:"
    )


    print(

        feel_df[
            "domain"
        ]
        .value_counts()
        .to_string()

    )


    return feel_df


# ============================================================
# MOVIE NAME MAPPING
# ============================================================


def load_movie_names(
    target_ids
):

    section(
        "LOADING MOVIE TITLES"
    )


    movie_file = first_existing_file(

        MOVIE_FILE_CANDIDATES,

        "Movie"

    )


    print(
        f"Source: {movie_file}"
    )


    header = pd.read_csv(

        movie_file,

        nrows=0

    )


    id_column = find_column(

        header.columns,

        [
            "movieId",
            "movie_id"
        ],

        "Movie ID"

    )


    name_column = find_column(

        header.columns,

        [
            "title",
            "name"
        ],

        "Movie title"

    )


    movies = pd.read_csv(

        movie_file,

        usecols=[

            id_column,

            name_column

        ]

    )


    movies[
        "source_id_key"
    ] = (

        movies[
            id_column
        ]
        .apply(
            normalise_source_id
        )

    )


    movies = movies.loc[

        movies[
            "source_id_key"
        ].isin(
            target_ids
        )

    ].copy()


    movies = movies.drop_duplicates(

        subset=[
            "source_id_key"
        ],

        keep="first"

    )


    mapping = pd.DataFrame({

        "domain":
            "movies",

        "source_id_key":
            movies[
                "source_id_key"
            ],

        "display_name":
            movies[
                name_column
            ].astype(str),

    })


    print(
        f"Movie names recovered: "
        f"{len(mapping):,}"
    )


    return mapping


# ============================================================
# MUSIC NAME MAPPING
# ============================================================


def load_music_names(
    target_ids
):

    section(
        "LOADING ARTIST NAMES"
    )


    music_file = first_existing_file(

        MUSIC_FILE_CANDIDATES,

        "Music"

    )


    print(
        f"Source: {music_file}"
    )


    header = pd.read_csv(

        music_file,

        nrows=0

    )


    id_column = find_column(

        header.columns,

        [
            "mbid",
            "artist_mbid",
            "musicbrainz_id",
            "musicbrainz_mbid"
        ],

        "MusicBrainz ID"

    )


    name_column = find_column(

        header.columns,

        [
            "artist_mb",
            "artist_lastfm",
            "artist_name",
            "name",
            "artist",
            "artistName"
        ],

        "artist name"

    )


    music = pd.read_csv(

        music_file,

        usecols=[

            id_column,

            name_column

        ],

        dtype={
            id_column:
                str
        }

    )


    music[
        "source_id_key"
    ] = (

        music[
            id_column
        ]
        .apply(
            normalise_source_id
        )

    )


    music = music.loc[

        music[
            "source_id_key"
        ].isin(
            target_ids
        )

    ].copy()


    music = music.drop_duplicates(

        subset=[
            "source_id_key"
        ],

        keep="first"

    )


    mapping = pd.DataFrame({

        "domain":
            "music",

        "source_id_key":
            music[
                "source_id_key"
            ],

        "display_name":
            music[
                name_column
            ].astype(str),

    })


    print(
        f"Artist names recovered: "
        f"{len(mapping):,}"
    )


    return mapping


# ============================================================
# RESTAURANT NAME MAPPING
# ============================================================


def load_restaurant_names(
    target_ids
):
    """
    Read Yelp business.json line-by-line.

    This deliberately avoids loading the 7-million-row review
    dataset simply to recover restaurant names.
    """

    section(
        "LOADING RESTAURANT NAMES"
    )


    if not RESTAURANT_BUSINESS_FILE.exists():

        raise FileNotFoundError(

            "Yelp business file was not found:\n"
            f"{RESTAURANT_BUSINESS_FILE}"

        )


    print(
        f"Source: {RESTAURANT_BUSINESS_FILE}"
    )


    records = []


    with open(

        RESTAURANT_BUSINESS_FILE,

        "r",

        encoding="utf-8"

    ) as file:


        for line in file:


            line = line.strip()


            if not line:

                continue


            record = json.loads(
                line
            )


            business_id = normalise_source_id(

                record.get(
                    "business_id"
                )

            )


            if business_id not in target_ids:

                continue


            records.append({

                "domain":
                    "restaurants",

                "source_id_key":
                    business_id,

                "display_name":
                    str(
                        record.get(
                            "name",
                            ""
                        )
                    ),

            })


    mapping = pd.DataFrame(
        records
    )


    if not mapping.empty:

        mapping = mapping.drop_duplicates(

            subset=[
                "source_id_key"
            ],

            keep="first"

        )


    print(
        f"Restaurant names recovered: "
        f"{len(mapping):,}"
    )


    return mapping


# ============================================================
# ATTACH HUMAN-READABLE NAMES
# ============================================================


def attach_entity_names(
    feel_df
):

    section(
        "ATTACHING HUMAN-READABLE ENTITY NAMES"
    )


    movie_ids = set(

        feel_df.loc[

            feel_df[
                "domain"
            ]
            ==
            "movies",

            "source_id_key"

        ]

    )


    music_ids = set(

        feel_df.loc[

            feel_df[
                "domain"
            ]
            ==
            "music",

            "source_id_key"

        ]

    )


    restaurant_ids = set(

        feel_df.loc[

            feel_df[
                "domain"
            ]
            ==
            "restaurants",

            "source_id_key"

        ]

    )


    movie_mapping = load_movie_names(
        movie_ids
    )


    music_mapping = load_music_names(
        music_ids
    )


    restaurant_mapping = load_restaurant_names(
        restaurant_ids
    )


    mapping = pd.concat(

        [

            movie_mapping,

            music_mapping,

            restaurant_mapping,

        ],

        ignore_index=True

    )


    named_df = feel_df.merge(

        mapping,

        on=[

            "domain",

            "source_id_key"

        ],

        how="left",

        validate="many_to_one"

    )


    named_df[
        "display_name"
    ] = (

        named_df[
            "display_name"
        ]
        .fillna(
            ""
        )
        .astype(str)

    )


    # ========================================================
    # RESOLUTION SUMMARY
    # ========================================================

    records = []


    for domain in sorted(

        named_df[
            "domain"
        ].unique()

    ):


        domain_df = named_df.loc[

            named_df[
                "domain"
            ]
            ==
            domain

        ]


        total = len(
            domain_df
        )


        resolved = int(

            (
                domain_df[
                    "display_name"
                ]
                .str.strip()
                !=
                ""
            ).sum()

        )


        records.append({

            "domain":
                domain,

            "entities":
                total,

            "names_resolved":
                resolved,

            "names_missing":
                total
                -
                resolved,

            "resolution_pct":
                (
                    resolved
                    /
                    total
                    *
                    100

                    if total > 0

                    else 0.0
                )

        })


    resolution = pd.DataFrame(
        records
    )


    resolution.to_csv(

        NAME_RESOLUTION_OUTPUT,

        index=False

    )


    print()

    print(
        resolution.to_string(
            index=False
        )
    )


    return named_df


# ============================================================
# SCORE INTERPRETATION
# ============================================================


def score_direction_label(

    dimension,

    direction

):
    """
    Convert mathematical high/low into semantic pole labels.
    """

    if dimension in BIPOLAR_AXES:

        axis = BIPOLAR_AXES[
            dimension
        ]


        if direction == "highest":

            return axis[
                "high_label"
            ]


        return axis[
            "low_label"
        ]


    axis = UNIPOLAR_AXES[
        dimension
    ]


    if direction == "highest":

        return (
            f"high_{axis['label']}"
        )


    return (
        f"low_{axis['label']}"
    )


# ============================================================
# BUILD DIMENSION EXTREMES
# ============================================================


def build_named_extremes(
    named_df
):
    """
    Find highest / lowest examples:
    
        globally
        Movies
        Music
        Restaurants
    """

    section(
        "BUILDING NAMED DIMENSION EXTREMES"
    )


    defined_df = named_df.loc[

        named_df[
            "is_semantically_defined"
        ]

    ].copy()


    records = []


    scopes = {

        "all_domains":
            defined_df,

        "movies":
            defined_df.loc[
                defined_df["domain"] == "movies"
            ],

        "music":
            defined_df.loc[
                defined_df["domain"] == "music"
            ],

        "restaurants":
            defined_df.loc[
                defined_df["domain"] == "restaurants"
            ],

    }


    metadata_columns = [

        column

        for column in [

            "domain",

            "source_id",

            "id",

            "display_name",

            "semantic_source",

            "has_base_semantics",

            "has_review_semantics",

            "reviews_used_for_embedding",

        ]

        if column in defined_df.columns

    ]


    for dimension in FEEL_DIMENSIONS:


        for scope_name, scope_df in (
            scopes.items()
        ):


            if scope_df.empty:

                continue


            lowest = (

                scope_df
                .nsmallest(

                    N_EXTREMES,

                    dimension

                )

            )


            highest = (

                scope_df
                .nlargest(

                    N_EXTREMES,

                    dimension

                )

            )


            # =================================================
            # LOWEST
            # =================================================

            for rank, (_, row) in enumerate(

                lowest.iterrows(),

                start=1

            ):


                record = {

                    "scope":
                        scope_name,

                    "dimension":
                        dimension,

                    "direction":
                        "lowest",

                    "semantic_interpretation":
                        score_direction_label(

                            dimension,

                            "lowest"

                        ),

                    "rank":
                        rank,

                    "score":
                        float(
                            row[
                                dimension
                            ]
                        ),

                }


                for column in metadata_columns:

                    record[
                        column
                    ] = row[
                        column
                    ]


                records.append(
                    record
                )


            # =================================================
            # HIGHEST
            # =================================================

            for rank, (_, row) in enumerate(

                highest.iterrows(),

                start=1

            ):


                record = {

                    "scope":
                        scope_name,

                    "dimension":
                        dimension,

                    "direction":
                        "highest",

                    "semantic_interpretation":
                        score_direction_label(

                            dimension,

                            "highest"

                        ),

                    "rank":
                        rank,

                    "score":
                        float(
                            row[
                                dimension
                            ]
                        ),

                }


                for column in metadata_columns:

                    record[
                        column
                    ] = row[
                        column
                    ]


                records.append(
                    record
                )


    extremes = pd.DataFrame(
        records
    )


    extremes.to_csv(

        EXTREMES_OUTPUT,

        index=False,

        encoding="utf-8"

    )


    print(
        f"Saved {len(extremes):,} named extreme examples."
    )


    print(
        f"Output: {EXTREMES_OUTPUT}"
    )


    return extremes


# ============================================================
# PRINT GLOBAL EXTREMES
# ============================================================


def print_global_extremes(
    extremes
):
    """
    Print only the top three global examples per direction so
    terminal output remains readable.
    """

    section(
        "GLOBAL EXTREME EXAMPLES"
    )


    global_extremes = extremes.loc[

        extremes[
            "scope"
        ]
        ==
        "all_domains"

    ]


    for dimension in FEEL_DIMENSIONS:


        print()

        print(
            "-" * 72
        )


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            print(

                f"{dimension.upper()} | "
                f"{axis['low_label']} "
                f"<-> "
                f"{axis['high_label']}"

            )


        else:

            print(
                dimension.upper()
            )


        print(
            "-" * 72
        )


        lowest = global_extremes.loc[

            (
                global_extremes[
                    "dimension"
                ]
                ==
                dimension
            )

            &

            (
                global_extremes[
                    "direction"
                ]
                ==
                "lowest"
            )

        ].head(
            3
        )


        highest = global_extremes.loc[

            (
                global_extremes[
                    "dimension"
                ]
                ==
                dimension
            )

            &

            (
                global_extremes[
                    "direction"
                ]
                ==
                "highest"
            )

        ].head(
            3
        )


        print(
            "Lowest:"
        )


        for _, row in (
            lowest.iterrows()
        ):

            print(

                f"  {row['score']:+.4f} | "
                f"{row['domain']:<11} | "
                f"{row['display_name']}"

            )


        print()


        print(
            "Highest:"
        )


        for _, row in (
            highest.iterrows()
        ):

            print(

                f"  {row['score']:+.4f} | "
                f"{row['domain']:<11} | "
                f"{row['display_name']}"

            )


# ============================================================
# FIND KNOWN EXAMPLE
# ============================================================


def find_known_example(

    named_df,

    domain,

    query

):
    """
    Locate one recognisable entity.

    Preference:
        1. Exact normalized-name match and semantically defined
        2. Partial normalized-name match and semantically defined
        3. Exact match even if undefined
        4. Partial match even if undefined
    """

    domain_df = named_df.loc[

        named_df[
            "domain"
        ]
        ==
        domain

    ].copy()


    domain_df = domain_df.loc[

        domain_df[
            "display_name"
        ]
        .str.strip()
        !=
        ""

    ].copy()


    if domain_df.empty:

        return None


    domain_df[
        "_name_key"
    ] = (

        domain_df[
            "display_name"
        ]
        .apply(
            normalise_name
        )

    )


    query_key = normalise_name(
        query
    )


    # ========================================================
    # EXACT
    # ========================================================

    exact = domain_df.loc[

        domain_df[
            "_name_key"
        ]
        ==
        query_key

    ].copy()


    if not exact.empty:

        defined_exact = exact.loc[

            exact[
                "is_semantically_defined"
            ]

        ]


        if not defined_exact.empty:

            return defined_exact.iloc[
                0
            ]


        return exact.iloc[
            0
        ]


    # ========================================================
    # PARTIAL
    # ========================================================

    partial = domain_df.loc[

        domain_df[
            "_name_key"
        ]
        .str.contains(

            query_key,

            regex=False,

            na=False

        )

    ].copy()


    if not partial.empty:

        defined_partial = partial.loc[

            partial[
                "is_semantically_defined"
            ]

        ]


        if not defined_partial.empty:

            return defined_partial.iloc[
                0
            ]


        return partial.iloc[
            0
        ]


    return None


# ============================================================
# BUILD KNOWN-ITEM PROFILES
# ============================================================


def build_known_profiles(
    named_df
):

    section(
        "KNOWN-ITEM SANITY CHECKS"
    )


    records = []


    for domain, queries in (
        KNOWN_EXAMPLES.items()
    ):


        for query in queries:


            row = find_known_example(

                named_df,

                domain,

                query

            )


            if row is None:

                print(

                    f"[NOT FOUND] "
                    f"{domain:<11} | "
                    f"{query}"

                )

                continue


            record = {

                "requested_name":
                    query,

                "domain":
                    domain,

                "source_id":
                    row[
                        "source_id"
                    ],

                "display_name":
                    row[
                        "display_name"
                    ],

                "is_semantically_defined":
                    bool(
                        row[
                            "is_semantically_defined"
                        ]
                    ),

            }


            for optional_column in [

                "semantic_source",

                "has_base_semantics",

                "has_review_semantics",

                "reviews_used_for_embedding",

            ]:


                if optional_column in row.index:

                    record[
                        optional_column
                    ] = row[
                        optional_column
                    ]


            for dimension in FEEL_DIMENSIONS:

                record[
                    dimension
                ] = float(
                    row[
                        dimension
                    ]
                )


            records.append(
                record
            )


    profiles = pd.DataFrame(
        records
    )


    profiles.to_csv(

        KNOWN_PROFILES_OUTPUT,

        index=False,

        encoding="utf-8"

    )


    print()

    print(
        f"Known profiles recovered: "
        f"{len(profiles):,}"
    )


    print(
        f"Output: "
        f"{KNOWN_PROFILES_OUTPUT}"
    )


    return profiles


# ============================================================
# PRINT KNOWN PROFILES
# ============================================================


def print_known_profiles(
    profiles
):

    section(
        "RECOGNISABLE ENTITY FEEL PROFILES"
    )


    if profiles.empty:

        print(
            "No known examples were recovered."
        )

        return


    for _, row in (
        profiles.iterrows()
    ):


        print()

        print(
            "-" * 72
        )


        print(

            f"{row['display_name']} "
            f"[{row['domain']}]"

        )


        print(
            "-" * 72
        )


        print(

            f"Semantic source: "
            f"{row.get('semantic_source', 'unknown')}"

        )


        print(

            f"Defined: "
            f"{row['is_semantically_defined']}"

        )


        if not bool(
            row[
                "is_semantically_defined"
            ]
        ):

            print(

                "No semantic representation available; "
                "scores should not be interpreted."

            )

            continue


        for dimension in FEEL_DIMENSIONS:


            score = float(
                row[
                    dimension
                ]
            )


            if dimension in BIPOLAR_AXES:

                axis = BIPOLAR_AXES[
                    dimension
                ]


                if score >= 0:

                    pole = axis[
                        "high_label"
                    ]


                else:

                    pole = axis[
                        "low_label"
                    ]


                print(

                    f"  {dimension:<12} "
                    f"{score:+.4f} "
                    f"-> {pole}"

                )


            else:

                print(

                    f"  {dimension:<12} "
                    f"{score:+.4f}"

                )


# ============================================================
# FINAL SUMMARY
# ============================================================


def print_final_summary(

    named_df,

    extremes,

    profiles

):

    section(
        "QUALITATIVE FEEL-SPACE INSPECTION COMPLETE"
    )


    defined = int(

        named_df[
            "is_semantically_defined"
        ].sum()

    )


    named_defined = int(

        (

            named_df[
                "is_semantically_defined"
            ]

            &

            (
                named_df[
                    "display_name"
                ]
                .str.strip()
                !=
                ""
            )

        ).sum()

    )


    print(
        f"Defined entities:              "
        f"{defined:,}"
    )


    print(
        f"Defined entities with names:   "
        f"{named_defined:,}"
    )


    print(
        f"Extreme examples saved:        "
        f"{len(extremes):,}"
    )


    print(
        f"Known profiles recovered:      "
        f"{len(profiles):,}"
    )


    print()


    print(
        "Outputs:"
    )


    print(
        f"  {EXTREMES_OUTPUT}"
    )


    print(
        f"  {KNOWN_PROFILES_OUTPUT}"
    )


    print(
        f"  {NAME_RESOLUTION_OUTPUT}"
    )


    print()

    print(
        "If the examples are broadly semantically plausible, "
        "the anchor definitions can be frozen and the Feel "
        "atlases can be built."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    section(
        "QUALITATIVE SHARED FEEL-SPACE INSPECTION"
    )


    # ========================================================
    # 1. LOAD EXISTING 13D FEEL SPACE
    # ========================================================

    feel_df = load_feel_space()


    # ========================================================
    # 2. ATTACH MOVIE / ARTIST / RESTAURANT NAMES
    # ========================================================

    named_df = attach_entity_names(
        feel_df
    )


    # ========================================================
    # 3. EXTREME EXAMPLES
    # ========================================================

    extremes = build_named_extremes(
        named_df
    )


    print_global_extremes(
        extremes
    )


    # ========================================================
    # 4. RECOGNISABLE ENTITY PROFILES
    # ========================================================

    profiles = build_known_profiles(
        named_df
    )


    print_known_profiles(
        profiles
    )


    # ========================================================
    # 5. FINAL SUMMARY
    # ========================================================

    print_final_summary(

        named_df=
            named_df,

        extremes=
            extremes,

        profiles=
            profiles

    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()