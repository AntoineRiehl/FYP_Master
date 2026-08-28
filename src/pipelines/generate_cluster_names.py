# src/pipelines/generate_cluster_names.py

import argparse
import json
import re
import shutil

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

FRONTEND_DATA_ROOT = Path(
    "frontend/public/data"
)

OUTPUT_ROOT = Path(
    "data/processed/cluster_names"
)

FEEL_SCORES_PATH = Path(
    "data/processed/feel_space/"
    "movies_music_restaurants/"
    "feel_space_scores.csv"
)


ATLAS_IDS = [

    "movies",
    "music",
    "restaurants",

    "movies_music",
    "movies_music_feel",

    "movies_music_restaurants",
    "movies_music_restaurants_feel",

]


FEEL_DIMENSIONS = [

    "valence",
    "activation",
    "potency",
    "tension",
    "warmth",
    "scale",
    "tone",
    "familiarity",
    "refinement",
    "complexity",
    "nostalgia",
    "wonder",
    "tenderness",

]


# =========================================================
# REPRESENTATIVE COUNTS
# =========================================================

CENTRAL_REPRESENTATIVE_COUNT = 15

POPULAR_REPRESENTATIVES_PER_DOMAIN = 5

TOP_CATEGORY_COUNT = 12

TOP_TERM_COUNT = 18


# =========================================================
# SIMPLE TERM FILTERING
# =========================================================

STOPWORDS = {

    "the", "and", "for", "with", "this", "that",
    "from", "into", "about", "movie", "film",
    "music", "artist", "restaurant", "restaurants",
    "very", "good", "great", "best", "favorite",
    "favourite", "see", "seen", "watch", "watched",
    "like", "liked", "love", "loved", "one",
    "two", "three", "really", "also", "much",
    "more", "most", "has", "have", "had",
    "was", "were", "are", "is", "be",
    "been", "being", "it", "its", "of",
    "to", "in", "on", "at", "as", "by",
    "an", "a", "or", "not", "but",

}


# =========================================================
# JSON HELPERS
# =========================================================

def load_json(
    path: Path
):

    with open(
        path,
        "r",
        encoding="utf8"
    ) as file:

        return json.load(
            file
        )


def write_json(
    path: Path,
    payload
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf8"
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )


def write_text(
    path: Path,
    text: str
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        text,
        encoding="utf8"
    )


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_id(
    value
):

    if value is None:

        return None


    value = str(
        value
    ).strip()


    return (
        value
        if value
        else None
    )


def get_cluster_id(
    node
):

    cluster = (
        node
        .get(
            "visual",
            {}
        )
        .get(
            "cluster"
        )
    )


    if cluster is None:

        return None


    try:

        cluster = int(
            cluster
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if cluster < 0:

        return None


    return cluster


def get_existing_cluster_label(
    node
):

    value = (
        node
        .get(
            "visual",
            {}
        )
        .get(
            "cluster_label"
        )
    )


    if value is None:

        return None


    value = str(
        value
    ).strip()


    return (
        value
        if value
        else None
    )


def get_popularity(
    node
):

    value = (
        node
        .get(
            "statistics",
            {}
        )
        .get(
            "popularity"
        )
    )


    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if not np.isfinite(
        value
    ):

        return None


    return value


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(
    text: str
):

    if not text:

        return []


    tokens = re.findall(

        r"[A-Za-z][A-Za-z\-']{2,}",

        text.lower()

    )


    return [

        token
        for token
        in tokens

        if (
            token not in STOPWORDS
            and
            len(token) >= 3
        )

    ]


# =========================================================
# CATEGORY EXTRACTION
# =========================================================

def extract_categories(
    node
):

    categories = (
        node
        .get(
            "text",
            {}
        )
        .get(
            "categories",
            []
        )
        or
        []
    )


    output = []


    for category_group in categories:

        if not category_group:

            continue


        for category in str(
            category_group
        ).split("|"):

            category = (
                category
                .strip()
            )


            if category:

                output.append(
                    category
                )


    return output


# =========================================================
# SEMANTIC TEXT EXTRACTION
# =========================================================

def extract_semantic_text(
    node
):

    tags = (
        node
        .get(
            "text",
            {}
        )
        .get(
            "tags",
            []
        )
        or
        []
    )


    return " ".join(

        str(tag)
        for tag
        in tags
        if tag

    )


# =========================================================
# FEEL DATA
# =========================================================

def build_feel_lookup():
    """
    Reconstruct the exact global standardized Feel
    representation used by Method B.

    Returns:

        {
            (domain, source_id):
                {
                    "valence": z,
                    ...
                }
        }
    """

    print(
        "Loading Feel scores..."
    )


    df = pd.read_csv(

        FEEL_SCORES_PATH,

        dtype={
            "domain": "string",
            "source_id": "string"
        }

    )


    df["domain"] = (

        df["domain"]
        .astype("string")
        .str.strip()
        .str.lower()

    )


    df["source_id"] = (

        df["source_id"]
        .astype("string")
        .str.strip()

    )


    defined = (

        df[
            "is_semantically_defined"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1"
            ]
        )

    )


    defined_df = (

        df[
            defined
        ]
        .copy()

    )


    values = (

        defined_df[
            FEEL_DIMENSIONS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )

    )


    valid = (

        ~values
        .isna()
        .any(
            axis=1
        )

    )


    defined_df = (

        defined_df[
            valid
        ]
        .copy()

    )


    values = (

        values[
            valid
        ]
        .to_numpy(
            dtype=np.float64
        )

    )


    scaler = StandardScaler()


    scaled = scaler.fit_transform(
        values
    )


    lookup = {}


    domains = (
        defined_df[
            "domain"
        ]
        .astype(str)
        .tolist()
    )


    source_ids = (
        defined_df[
            "source_id"
        ]
        .astype(str)
        .tolist()
    )


    for index in range(
        len(defined_df)
    ):


        key = (

            domains[index],

            source_ids[index]

        )


        lookup[
            key
        ] = {

            dimension:
                float(
                    scaled[
                        index,
                        dimension_index
                    ]
                )

            for (
                dimension_index,
                dimension
            )
            in enumerate(
                FEEL_DIMENSIONS
            )

        }


    print(
        f"Feel lookup ready: "
        f"{len(lookup):,} entities."
    )


    return lookup


# =========================================================
# CLUSTER CENTROID
# =========================================================

def calculate_centroid(
    members
):

    x = np.mean(
        [
            float(
                node["position"]["x"]
            )
            for node in members
        ]
    )


    y = np.mean(
        [
            float(
                node["position"]["y"]
            )
            for node in members
        ]
    )


    return (
        float(x),
        float(y)
    )


# =========================================================
# CENTRAL REPRESENTATIVES
# =========================================================

def get_central_representatives(
    members,
    centroid,
    count
):

    cx, cy = centroid


    ranked = sorted(

        members,

        key=lambda node:

            (
                float(
                    node["position"]["x"]
                )
                -
                cx
            )
            ** 2

            +

            (
                float(
                    node["position"]["y"]
                )
                -
                cy
            )
            ** 2

    )


    return ranked[
        :count
    ]


# =========================================================
# POPULAR REPRESENTATIVES
# =========================================================

def get_popular_representatives(
    members,
    count_per_domain
):
    """
    Select popular representatives separately within
    each domain so incompatible raw popularity scales
    are never compared across domains.
    """

    by_domain = defaultdict(
        list
    )


    for node in members:

        by_domain[
            node.get(
                "domain",
                "unknown"
            )
        ].append(
            node
        )


    result = {}


    for (
        domain,
        domain_members
    ) in by_domain.items():


        ranked = sorted(

            domain_members,

            key=lambda node:

                get_popularity(
                    node
                )
                if
                get_popularity(
                    node
                )
                is not None

                else
                -np.inf,

            reverse=True

        )


        result[
            domain
        ] = ranked[
            :count_per_domain
        ]


    return result


# =========================================================
# DOMAIN COMPOSITION
# =========================================================

def get_domain_composition(
    members
):

    counts = Counter(

        node.get(
            "domain",
            "unknown"
        )

        for node
        in members

    )


    total = len(
        members
    )


    return {

        domain: {

            "count":
                count,

            "share":
                round(
                    count
                    /
                    total,
                    4
                )

        }

        for (
            domain,
            count
        )
        in counts.most_common()

    }


# =========================================================
# DOMINANT CATEGORIES
# =========================================================

def get_dominant_categories(
    members
):

    counts = Counter()


    for node in members:

        pass

        counts = Counter()


    for node in members:

        unique_categories = set(

            extract_categories(
                node
            )

        )


        counts.update(
            unique_categories
        )


    return [

        {
            "category":
                category,

            "entity_count":
                count,

            "share":
                round(
                    count
                    /
                    len(members),
                    4
                )

        }

        for (
            category,
            count
        )
        in counts.most_common(
            TOP_CATEGORY_COUNT
        )

    ]


# =========================================================
# DOMINANT SEMANTIC TERMS
# =========================================================

def get_dominant_terms(
    representative_nodes
):
    """
    Use document frequency rather than raw token frequency.

    If "Pixar" appears 50 times in one enormous MovieLens
    tag corpus, it still counts only once for that entity.
    """

    counts = Counter()


    for node in representative_nodes:

        terms = set(

            tokenize(

                extract_semantic_text(
                    node
                )

            )

        )


        counts.update(
            terms
        )


    return [

        {
            "term":
                term,

            "representative_count":
                count

        }

        for (
            term,
            count
        )
        in counts.most_common(
            TOP_TERM_COUNT
        )

    ]


# =========================================================
# MEAN FEEL PROFILE
# =========================================================

def calculate_cluster_feel_profile(
    members,
    feel_lookup
):

    values = []


    for node in members:

        domain = str(

            node.get(
                "domain",
                ""
            )

        ).strip().lower()


        source_id = normalize_id(

            node.get(
                "source_id"
            )
            or
            node.get(
                "id"
            )

        )


        if not source_id:

            continue


        profile = feel_lookup.get(

            (
                domain,
                source_id
            )

        )


        if profile is None:

            continue


        values.append(

            [
                profile[
                    dimension
                ]

                for dimension
                in FEEL_DIMENSIONS
            ]

        )


    if not values:

        return {

            "coverage_count":
                0,

            "coverage_share":
                0,

            "mean":
                None,

            "strongest_dimensions":
                []

        }


    matrix = np.asarray(

        values,

        dtype=np.float64

    )


    mean = matrix.mean(
        axis=0
    )


    mean_profile = {

        dimension:
            round(
                float(
                    mean[index]
                ),
                4
            )

        for (
            index,
            dimension
        )
        in enumerate(
            FEEL_DIMENSIONS
        )

    }


    ranked = sorted(

        mean_profile.items(),

        key=lambda item:

            abs(
                item[1]
            ),

        reverse=True

    )


    strongest = [

        {
            "dimension":
                dimension,

            "mean_z":
                value

        }

        for (
            dimension,
            value
        )
        in ranked[
            :7
        ]

    ]


    return {

        "coverage_count":
            len(values),

        "coverage_share":
            round(
                len(values)
                /
                len(members),
                4
            ),

        "mean":
            mean_profile,

        "strongest_dimensions":
            strongest

    }


# =========================================================
# SIMPLE NODE REPRESENTATION
# =========================================================

def summarize_node(
    node
):

    return {

        "id":
            normalize_id(
                node.get(
                    "id"
                )
            ),

        "source_id":
            normalize_id(
                node.get(
                    "source_id"
                )
            ),

        "title":
            node.get(
                "title"
            ),

        "domain":
            node.get(
                "domain"
            ),

        "popularity":
            get_popularity(
                node
            )

    }


# =========================================================
# ATLAS TYPE
# =========================================================

def get_atlas_type(
    atlas_id
):

    if atlas_id.endswith(
        "_feel"
    ):

        return "feel"


    if atlas_id in {

        "movies",
        "music",
        "restaurants",

    }:

        return "mono"


    return "general_cross_domain"


# =========================================================
# BUILD CLUSTER SUMMARY
# =========================================================

def build_cluster_summary(
    atlas_id,
    cluster_id,
    members,
    feel_lookup
):

    centroid = calculate_centroid(
        members
    )


    central = get_central_representatives(

        members,

        centroid,

        CENTRAL_REPRESENTATIVE_COUNT

    )


    popular = get_popular_representatives(

        members,

        POPULAR_REPRESENTATIVES_PER_DOMAIN

    )



    # Semantic terms are deliberately extracted only
    # from a compact representative population.

    term_nodes = list(
        central
    )


    for domain_nodes in popular.values():

        term_nodes.extend(
            domain_nodes
        )


    # Deduplicate representatives by atlas ID.

    unique_term_nodes = {}

    for node in term_nodes:

        node_id = normalize_id(
            node.get(
                "id"
            )
        )

        if node_id:

            unique_term_nodes[
                node_id
            ] = node


    existing_labels = Counter(

        get_existing_cluster_label(
            node
        )

        for node
        in members

        if get_existing_cluster_label(
            node
        )

    )


    existing_label = (

        existing_labels
        .most_common(
            1
        )[0][0]

        if existing_labels

        else None
    )


    return {

        "atlas_id":
            atlas_id,

        "atlas_type":
            get_atlas_type(
                atlas_id
            ),

        "cluster_id":
            cluster_id,

        "existing_label":
            existing_label,

        "size":
            len(members),

        "centroid": {

            "x":
                round(
                    centroid[0],
                    5
                ),

            "y":
                round(
                    centroid[1],
                    5
                )

        },

        "domain_composition":
            get_domain_composition(
                members
            ),

        "central_representatives": [

            summarize_node(
                node
            )

            for node
            in central

        ],

        "popular_representatives": {

            domain: [

                summarize_node(
                    node
                )

                for node
                in domain_nodes

            ]

            for (
                domain,
                domain_nodes
            )
            in popular.items()

        },

        "dominant_categories":
            get_dominant_categories(
                members
            ),

        "dominant_semantic_terms":
            get_dominant_terms(

                list(
                    unique_term_nodes.values()
                )

            ),

        "feel_profile":
            calculate_cluster_feel_profile(

                members,

                feel_lookup

            )

    }


# =========================================================
# NAMING INSTRUCTIONS
# =========================================================

def get_naming_instructions(
    atlas_type
):

    common = """
You are naming one region of a semantic cultural atlas.

Return a concise, interpretable region name and a one-sentence
description.

Rules:
- The name should normally contain 2 to 6 words.
- Be specific rather than generic.
- Avoid labels such as "Mixed", "General", "Various", "Other",
  or "Miscellaneous".
- Do not simply copy the existing label.
- Do not claim that every item in the region has exactly the
  same properties.
- Treat the region name as a post-hoc interpretive description,
  not as a ground-truth class.
- Use representative items, categories, semantic terms and
  cluster composition together.
""".strip()


    if atlas_type == "feel":

        return (
            common
            +
            """

This is a Shared Experiential / Feel atlas.

The region was formed primarily from experiential dimensions,
not from genre/category labels.

Prioritise the shared experiential character:
valence, activation, potency, tension, warmth, scale, tone,
familiarity, refinement, complexity, nostalgia, wonder and
tenderness.

Do NOT default to genre labels merely because several
representative items share a genre.

Prefer names such as:
- Warm Nostalgic Comfort
- Grand Dark Intensity
- Playful Familiar Energy
- Intimate Tender Reflection

when the evidence supports them.
"""
        )


    if atlas_type == "general_cross_domain":

        return (
            common
            +
            """

This is a cross-domain General Semantic atlas containing multiple
cultural domains.

Prefer a semantic theme that can sensibly describe the represented
domains rather than naming only one domain.

Concrete themes are welcome when supported by the evidence.
"""
        )


    return (
        common
        +
        """

This is a mono-domain semantic atlas.

Domain-specific terminology, genres and categories may be used
when they accurately distinguish this region from neighbouring
regions.
"""
    )


# =========================================================
# BUILD CLUSTER PROMPT
# =========================================================

def build_cluster_prompt(
    summary
):

    instructions = get_naming_instructions(

        summary[
            "atlas_type"
        ]

    )


    summary_json = json.dumps(

        summary,

        ensure_ascii=False,

        indent=2

    )


    return f"""
{instructions}

CLUSTER INFORMATION

{summary_json}

Return ONLY valid JSON in this exact form:

{{
  "cluster_id": {summary["cluster_id"]},
  "name": "Concise Region Name",
  "description": "One concise sentence describing the region."
}}
""".strip()


# =========================================================
# PREPARE ONE ATLAS
# =========================================================

def prepare_atlas(
    atlas_id,
    feel_lookup
):

    atlas_path = (

        FRONTEND_DATA_ROOT
        /
        atlas_id
        /
        "atlas.json"

    )


    if not atlas_path.exists():

        raise FileNotFoundError(

            f"Atlas file not found: "
            f"{atlas_path}"

        )


    print(
        "\n"
        "=================================================="
    )

    print(
        f"Preparing naming data: {atlas_id}"
    )

    print(
        "=================================================="
    )


    nodes = load_json(
        atlas_path
    )


    clusters = defaultdict(
        list
    )


    for node in nodes:

        cluster_id = get_cluster_id(
            node
        )


        if cluster_id is None:

            continue


        clusters[
            cluster_id
        ].append(
            node
        )


    summaries = []


    prompt_directory = (

        OUTPUT_ROOT
        /
        "prompts"
        /
        atlas_id

    )


    for cluster_id in sorted(
        clusters
    ):


        summary = build_cluster_summary(

            atlas_id,

            cluster_id,

            clusters[
                cluster_id
            ],

            feel_lookup

        )


        summaries.append(
            summary
        )


        prompt = build_cluster_prompt(
            summary
        )


        write_text(

            prompt_directory
            /
            f"cluster_{cluster_id}.txt",

            prompt

        )


    # -----------------------------------------------------
    # STRUCTURED NAMING INPUT
    # -----------------------------------------------------

    write_json(

        OUTPUT_ROOT
        /
        "inputs"
        /
        f"{atlas_id}.json",

        summaries

    )


    # -----------------------------------------------------
    # JSONL PROMPTS
    # -----------------------------------------------------

    jsonl_path = (

        OUTPUT_ROOT
        /
        "prompts"
        /
        f"{atlas_id}.jsonl"

    )


    jsonl_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        jsonl_path,
        "w",
        encoding="utf8"
    ) as file:


        for summary in summaries:


            record = {

                "atlas_id":
                    atlas_id,

                "cluster_id":
                    summary[
                        "cluster_id"
                    ],

                "prompt":
                    build_cluster_prompt(
                        summary
                    )

            }


            file.write(

                json.dumps(
                    record,
                    ensure_ascii=False
                )

                +

                "\n"

            )


    print(
        f"Clusters prepared: "
        f"{len(summaries):,}"
    )


    print(
        f"Input summary: "
        f"{OUTPUT_ROOT / 'inputs' / f'{atlas_id}.json'}"
    )


    print(
        f"Prompt folder: "
        f"{prompt_directory}"
    )


# =========================================================
# VALIDATE FINAL NAMES
# =========================================================

def validate_names(
    atlas_id,
    names
):

    if not isinstance(
        names,
        dict
    ):

        raise ValueError(

            "Final names file must be a JSON object "
            "keyed by cluster ID."

        )


    seen_names = Counter()


    for (
        cluster_id,
        record
    ) in names.items():


        if not isinstance(
            record,
            dict
        ):

            raise ValueError(

                f"Cluster {cluster_id} "
                "does not contain an object."

            )


        name = str(

            record.get(
                "name",
                ""
            )

        ).strip()


        description = str(

            record.get(
                "description",
                ""
            )

        ).strip()


        if not name:

            raise ValueError(

                f"Cluster {cluster_id} "
                "has no name."

            )


        if not description:

            raise ValueError(

                f"Cluster {cluster_id} "
                "has no description."

            )


        seen_names[
            name.lower()
        ] += 1


    duplicates = [

        name
        for (
            name,
            count
        )
        in seen_names.items()

        if count > 1

    ]


    if duplicates:

        print(
            "\nWARNING: duplicate names detected:"
        )


        for name in duplicates:

            print(
                f"  - {name}"
            )


    print(
        f"\nValidated names for "
        f"{atlas_id}: "
        f"{len(names):,} clusters."
    )


# =========================================================
# APPLY FINAL NAMES
# =========================================================

def apply_names(
    atlas_id,
    names_path
):

    atlas_directory = (

        FRONTEND_DATA_ROOT
        /
        atlas_id

    )


    atlas_path = (

        atlas_directory
        /
        "atlas.json"

    )


    regions_path = (

        atlas_directory
        /
        "regions.json"

    )


    names = load_json(
        names_path
    )


    validate_names(

        atlas_id,

        names

    )


    # =====================================================
    # BACKUPS
    # =====================================================

    backup_directory = (

        OUTPUT_ROOT
        /
        "backups"
        /
        atlas_id

    )


    backup_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    shutil.copy2(

        atlas_path,

        backup_directory
        /
        "atlas.json"

    )


    if regions_path.exists():

        shutil.copy2(

            regions_path,

            backup_directory
            /
            "regions.json"

        )


    print(
        f"Backup saved to: "
        f"{backup_directory}"
    )


    # =====================================================
    # UPDATE ATLAS NODES
    # =====================================================

    nodes = load_json(
        atlas_path
    )


    updated_nodes = 0


    for node in nodes:


        cluster_id = get_cluster_id(
            node
        )


        if cluster_id is None:

            continue


        record = names.get(
            str(
                cluster_id
            )
        )


        if not record:

            continue


        node.setdefault(
            "visual",
            {}
        )


        node[
            "visual"
        ][
            "cluster_label"
        ] = record[
            "name"
        ]


        updated_nodes += 1


    write_json(

        atlas_path,

        nodes

    )


    # =====================================================
    # UPDATE REGIONS
    # =====================================================

    updated_regions = 0


    if regions_path.exists():


        regions = load_json(
            regions_path
        )


        for region in regions:


            cluster_id = region.get(
                "id"
            )


            if cluster_id is None:

                continue


            record = names.get(
                str(
                    cluster_id
                )
            )


            if not record:

                continue


            region[
                "label"
            ] = record[
                "name"
            ]


            region[
                "description"
            ] = record[
                "description"
            ]


            updated_regions += 1


        write_json(

            regions_path,

            regions

        )


    print(
        f"Updated atlas nodes: "
        f"{updated_nodes:,}"
    )


    print(
        f"Updated regions: "
        f"{updated_regions:,}"
    )


    print(
        "\nNaming applied successfully."
    )


# =========================================================
# PREPARE COMMAND
# =========================================================

def run_prepare(
    atlas_ids
):

    feel_lookup = build_feel_lookup()


    for atlas_id in atlas_ids:

        prepare_atlas(

            atlas_id,

            feel_lookup

        )


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Prepare LLM cluster-naming inputs "
            "and optionally apply approved names."
        )

    )


    subparsers = parser.add_subparsers(

        dest="command",

        required=True

    )


    # =====================================================
    # PREPARE
    # =====================================================

    prepare_parser = subparsers.add_parser(

        "prepare",

        help=(
            "Build cluster summaries and "
            "LLM-ready prompts."
        )

    )


    prepare_parser.add_argument(

        "--atlas",

        nargs="+",

        choices=ATLAS_IDS,

        help=(
            "One or more atlas IDs. "
            "Use --all to prepare every atlas."
        )

    )


    prepare_parser.add_argument(

        "--all",

        action="store_true",

        help=(
            "Prepare all registered atlases."
        )

    )


    # =====================================================
    # APPLY
    # =====================================================

    apply_parser = subparsers.add_parser(

        "apply",

        help=(
            "Apply an approved cluster-name JSON "
            "to frontend atlas/region files."
        )

    )


    apply_parser.add_argument(

        "--atlas",

        required=True,

        choices=ATLAS_IDS

    )


    apply_parser.add_argument(

        "--names",

        required=True,

        type=Path,

        help=(
            "Approved JSON file containing "
            "cluster names/descriptions."
        )

    )


    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    args = parse_args()


    if args.command == "prepare":


        if args.all:

            atlas_ids = ATLAS_IDS


        elif args.atlas:

            atlas_ids = args.atlas


        else:

            raise ValueError(

                "Provide --atlas <id> "
                "or use --all."

            )


        run_prepare(
            atlas_ids
        )


    elif args.command == "apply":

        apply_names(

            atlas_id=
                args.atlas,

            names_path=
                args.names

        )