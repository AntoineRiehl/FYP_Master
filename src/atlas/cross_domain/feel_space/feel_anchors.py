# src/atlas/cross_domain/feel_space/feel_anchors.py

# ============================================================
# SHARED EXPERIENTIAL / FEEL SPACE
# ============================================================
#
# This module defines the semantic anchor system used by
# Method B of the cross-domain atlas.
#
#
# PURPOSE
# -------
#
# Rather than attempting to remove every domain-specific
# word from Movies, Music and Restaurants, Method B defines
# explicitly which semantic characteristics should matter.
#
# Every entity is represented according to a set of shared
# experiential dimensions such as:
#
#     valence
#     activation
#     tension
#     warmth
#     scale
#     refinement
#     nostalgia
#     wonder
#
# These dimensions are intended to describe the perceived
# experiential / affective character of an entity rather
# than its domain.
#
#
# IMPORTANT TERMINOLOGY
# ---------------------
#
# These scores should NOT be interpreted as:
#
#     "the emotion a user will definitely feel"
#
# They represent:
#
#     semantic similarity between the textual representation
#     of an entity and controlled experiential anchor phrases.
#
#
# REPRESENTATION
# --------------
#
# 10 bipolar dimensions:
#
#     score =
#         cosine(entity, high_anchor)
#         -
#         cosine(entity, low_anchor)
#
#
# 3 unipolar dimensions:
#
#     score =
#         cosine(entity, anchor)
#
#
# This produces:
#
#     23 anchor embeddings
#          ↓
#     13 experiential dimensions
#
#
# MODEL
# -----
#
# The anchors must be embedded using the same model as the
# base semantic and review representations:
#
#     sentence-transformers/all-MiniLM-L6-v2
#
#
# DESIGN PRINCIPLES
# -----------------
#
# - domain-neutral wording
# - same grammatical structure across opposing anchors
# - descriptive rather than evaluative wording
# - applicable to Movies, Music and Restaurants
# - limited number of interpretable dimensions
#
# ============================================================


# ============================================================
# MODEL
# ============================================================

FEEL_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# BIPOLAR AXES
# ============================================================
#
# For each bipolar axis:
#
#     score =
#         similarity(high_anchor)
#         -
#         similarity(low_anchor)
#
#
# Therefore:
#
#     positive score -> closer to high pole
#     negative score -> closer to low pole
#
#
# "high" and "low" refer ONLY to the mathematical direction
# of the resulting axis.
#
# They do not imply that one side is better than the other.
# ============================================================


BIPOLAR_AXES = {


    # ========================================================
    # 1. VALENCE
    # ========================================================
    #
    # The positive ↔ negative affective character of the
    # experience.
    # ========================================================

    "valence": {

        "high_label":
            "positive",

        "low_label":
            "negative",

        "high_anchor":
            (
                "An experience that feels joyful, uplifting, "
                "pleasant and emotionally positive."
            ),

        "low_anchor":
            (
                "An experience that feels sad, bleak, "
                "unpleasant and emotionally negative."
            ),

        "description":
            (
                "Overall positive versus negative affective "
                "character."
            ),

    },


    # ========================================================
    # 2. ACTIVATION
    # ========================================================
    #
    # Closely related to affective arousal.
    # ========================================================

    "activation": {

        "high_label":
            "energetic",

        "low_label":
            "calm",

        "high_anchor":
            (
                "An experience that feels energetic, lively, "
                "exciting and stimulating."
            ),

        "low_anchor":
            (
                "An experience that feels calm, peaceful, "
                "relaxed and tranquil."
            ),

        "description":
            (
                "Energetic and stimulating versus calm and "
                "tranquil experiential character."
            ),

    },


    # ========================================================
    # 3. POTENCY
    # ========================================================
    #
    # Captures force / impact rather than simple arousal.
    # ========================================================

    "potency": {

        "high_label":
            "powerful",

        "low_label":
            "gentle",

        "high_anchor":
            (
                "An experience that feels powerful, forceful, "
                "overwhelming and impactful."
            ),

        "low_anchor":
            (
                "An experience that feels gentle, subtle, "
                "delicate and restrained."
            ),

        "description":
            (
                "Powerful and forceful versus gentle and "
                "restrained experiential character."
            ),

    },


    # ========================================================
    # 4. TENSION
    # ========================================================

    "tension": {

        "high_label":
            "unsettling",

        "low_label":
            "comforting",

        "high_anchor":
            (
                "An experience that feels tense, unsettling, "
                "anxious and emotionally uneasy."
            ),

        "low_anchor":
            (
                "An experience that feels comforting, safe, "
                "reassuring and emotionally soothing."
            ),

        "description":
            (
                "Tense and unsettling versus comforting and "
                "reassuring experiential character."
            ),

    },


    # ========================================================
    # 5. WARMTH
    # ========================================================

    "warmth": {

        "high_label":
            "warm",

        "low_label":
            "cold",

        "high_anchor":
            (
                "An experience that feels warm, welcoming, "
                "affectionate and emotionally inviting."
            ),

        "low_anchor":
            (
                "An experience that feels cold, distant, "
                "detached and emotionally reserved."
            ),

        "description":
            (
                "Warm and welcoming versus cold and detached "
                "experiential character."
            ),

    },


    # ========================================================
    # 6. SCALE
    # ========================================================
    #
    # We define the positive mathematical direction as GRAND,
    # making increasingly positive values correspond to a
    # greater perceived experiential scale.
    # ========================================================

    "scale": {

        "high_label":
            "grand",

        "low_label":
            "intimate",

        "high_anchor":
            (
                "An experience that feels grand, epic, "
                "expansive and monumental."
            ),

        "low_anchor":
            (
                "An experience that feels intimate, personal, "
                "close and small-scale."
            ),

        "description":
            (
                "Grand and expansive versus intimate and "
                "small-scale experiential character."
            ),

    },


    # ========================================================
    # 7. TONE
    # ========================================================

    "tone": {

        "high_label":
            "playful",

        "low_label":
            "serious",

        "high_anchor":
            (
                "An experience that feels playful, humorous, "
                "light-hearted and fun."
            ),

        "low_anchor":
            (
                "An experience that feels serious, solemn, "
                "grave and emotionally weighty."
            ),

        "description":
            (
                "Playful and light-hearted versus serious and "
                "solemn experiential character."
            ),

    },


    # ========================================================
    # 8. FAMILIARITY
    # ========================================================
    #
    # Positive values correspond to familiarity rather than
    # novelty.
    #
    # Importantly, familiarity is NOT the same as popularity.
    # ========================================================

    "familiarity": {

        "high_label":
            "familiar",

        "low_label":
            "novel",

        "high_anchor":
            (
                "An experience that feels familiar, accessible, "
                "conventional and recognisable."
            ),

        "low_anchor":
            (
                "An experience that feels novel, unusual, "
                "experimental and unconventional."
            ),

        "description":
            (
                "Familiar and conventional versus novel and "
                "experimental experiential character."
            ),

    },


    # ========================================================
    # 9. REFINEMENT
    # ========================================================

    "refinement": {

        "high_label":
            "refined",

        "low_label":
            "raw",

        "high_anchor":
            (
                "An experience that feels polished, elegant, "
                "refined and carefully crafted."
            ),

        "low_anchor":
            (
                "An experience that feels raw, rough, gritty "
                "and unpolished."
            ),

        "description":
            (
                "Polished and refined versus raw and rough "
                "experiential character."
            ),

    },


    # ========================================================
    # 10. COMPLEXITY
    # ========================================================
    #
    # Neither side is intended as a quality judgement.
    # ========================================================

    "complexity": {

        "high_label":
            "complex",

        "low_label":
            "simple",

        "high_anchor":
            (
                "An experience that feels complex, layered, "
                "intricate and intellectually rich."
            ),

        "low_anchor":
            (
                "An experience that feels simple, direct, "
                "straightforward and minimal."
            ),

        "description":
            (
                "Complex and layered versus simple and direct "
                "experiential character."
            ),

    },

}


# ============================================================
# UNIPOLAR AXES
# ============================================================
#
# These concepts do not have sufficiently clean semantic
# opposites for a bipolar construction.
#
# Their score is simply:
#
#     cosine(entity_embedding, anchor_embedding)
#
# ============================================================


UNIPOLAR_AXES = {


    # ========================================================
    # 11. NOSTALGIA
    # ========================================================

    "nostalgia": {

        "label":
            "nostalgic",

        "anchor":
            (
                "An experience that feels nostalgic, "
                "reminiscent, sentimental and connected "
                "to the past."
            ),

        "description":
            (
                "Degree to which the experience conveys "
                "nostalgia, reminiscence or sentimentality."
            ),

    },


    # ========================================================
    # 12. WONDER / AWE
    # ========================================================

    "wonder": {

        "label":
            "wonder",

        "anchor":
            (
                "An experience that inspires wonder, awe, "
                "amazement and a sense of the extraordinary."
            ),

        "description":
            (
                "Degree to which the experience evokes "
                "wonder, awe or amazement."
            ),

    },


    # ========================================================
    # 13. TENDERNESS / ROMANCE
    # ========================================================

    "tenderness": {

        "label":
            "tender",

        "anchor":
            (
                "An experience that feels tender, romantic, "
                "affectionate and emotionally intimate."
            ),

        "description":
            (
                "Degree to which the experience conveys "
                "tenderness, romance or affection."
            ),

    },

}


# ============================================================
# FINAL DIMENSION ORDER
# ============================================================
#
# This order should remain stable once Method B is frozen.
#
# It determines the column order of the final 13-dimensional
# experiential representation.
# ============================================================


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


# ============================================================
# COUNTS
# ============================================================

N_BIPOLAR_AXES = len(
    BIPOLAR_AXES
)


N_UNIPOLAR_AXES = len(
    UNIPOLAR_AXES
)


N_FEEL_DIMENSIONS = len(
    FEEL_DIMENSIONS
)


N_ANCHORS = (

    2
    *
    N_BIPOLAR_AXES

    +

    N_UNIPOLAR_AXES

)


# ============================================================
# GET ALL ANCHOR TEXTS
# ============================================================


def get_anchor_texts():
    """
    Return all 23 anchor sentences in a stable order.

    Returns
    -------
    list[str]
        Anchor sentences.

    Order
    -----
    For every bipolar axis:

        high anchor
        low anchor

    followed by every unipolar anchor.
    """

    anchors = []


    for dimension in FEEL_DIMENSIONS:

        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            anchors.append(
                axis[
                    "high_anchor"
                ]
            )


            anchors.append(
                axis[
                    "low_anchor"
                ]
            )


        elif dimension in UNIPOLAR_AXES:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            anchors.append(
                axis[
                    "anchor"
                ]
            )


    return anchors


# ============================================================
# GET NAMED ANCHORS
# ============================================================


def get_named_anchors():
    """
    Return all anchors with explicit stable names.

    Returns
    -------
    list[dict]

    Example
    -------

    {
        "anchor_name": "valence__positive",
        "dimension": "valence",
        "type": "bipolar_high",
        "label": "positive",
        "text": "..."
    }
    """

    anchors = []


    for dimension in FEEL_DIMENSIONS:


        # ====================================================
        # BIPOLAR
        # ====================================================

        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            anchors.append({

                "anchor_name":
                    (
                        f"{dimension}__"
                        f"{axis['high_label']}"
                    ),

                "dimension":
                    dimension,

                "type":
                    "bipolar_high",

                "label":
                    axis[
                        "high_label"
                    ],

                "text":
                    axis[
                        "high_anchor"
                    ],

            })


            anchors.append({

                "anchor_name":
                    (
                        f"{dimension}__"
                        f"{axis['low_label']}"
                    ),

                "dimension":
                    dimension,

                "type":
                    "bipolar_low",

                "label":
                    axis[
                        "low_label"
                    ],

                "text":
                    axis[
                        "low_anchor"
                    ],

            })


        # ====================================================
        # UNIPOLAR
        # ====================================================

        elif dimension in UNIPOLAR_AXES:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            anchors.append({

                "anchor_name":
                    dimension,

                "dimension":
                    dimension,

                "type":
                    "unipolar",

                "label":
                    axis[
                        "label"
                    ],

                "text":
                    axis[
                        "anchor"
                    ],

            })


    return anchors


# ============================================================
# GET DIMENSION METADATA
# ============================================================


def get_dimension_metadata():
    """
    Return metadata describing all 13 experiential
    dimensions.

    Useful later for:

        - diagnostic output
        - metadata JSON
        - frontend labels
        - report tables
    """

    dimensions = []


    for dimension in FEEL_DIMENSIONS:


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            dimensions.append({

                "dimension":
                    dimension,

                "type":
                    "bipolar",

                "high_label":
                    axis[
                        "high_label"
                    ],

                "low_label":
                    axis[
                        "low_label"
                    ],

                "description":
                    axis[
                        "description"
                    ],

            })


        else:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            dimensions.append({

                "dimension":
                    dimension,

                "type":
                    "unipolar",

                "label":
                    axis[
                        "label"
                    ],

                "description":
                    axis[
                        "description"
                    ],

            })


    return dimensions


# ============================================================
# VALIDATION
# ============================================================


def validate_feel_anchors():
    """
    Validate the internal consistency of the anchor
    configuration.

    Raises
    ------
    ValueError
        If the configuration is inconsistent.

    Returns
    -------
    True
        If validation succeeds.
    """

    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    if N_BIPOLAR_AXES != 10:

        raise ValueError(

            "Expected exactly 10 bipolar experiential axes, "
            f"found {N_BIPOLAR_AXES}."

        )


    if N_UNIPOLAR_AXES != 3:

        raise ValueError(

            "Expected exactly 3 unipolar experiential axes, "
            f"found {N_UNIPOLAR_AXES}."

        )


    if N_FEEL_DIMENSIONS != 13:

        raise ValueError(

            "Expected exactly 13 experiential dimensions, "
            f"found {N_FEEL_DIMENSIONS}."

        )


    if N_ANCHORS != 23:

        raise ValueError(

            "Expected exactly 23 semantic anchors, "
            f"found {N_ANCHORS}."

        )


    # ========================================================
    # DIMENSION UNIQUENESS
    # ========================================================

    if (

        len(
            FEEL_DIMENSIONS
        )

        !=

        len(
            set(
                FEEL_DIMENSIONS
            )
        )

    ):

        raise ValueError(

            "FEEL_DIMENSIONS contains duplicate names."

        )


    # ========================================================
    # EVERY CONFIGURED AXIS MUST APPEAR IN FINAL ORDER
    # ========================================================

    configured_dimensions = (

        set(
            BIPOLAR_AXES.keys()
        )

        |

        set(
            UNIPOLAR_AXES.keys()
        )

    )


    ordered_dimensions = set(
        FEEL_DIMENSIONS
    )


    if (

        configured_dimensions

        !=

        ordered_dimensions

    ):

        missing = (

            configured_dimensions
            -
            ordered_dimensions

        )


        unknown = (

            ordered_dimensions
            -
            configured_dimensions

        )


        raise ValueError(

            "Dimension configuration mismatch.\n"
            f"Missing from FEEL_DIMENSIONS: {sorted(missing)}\n"
            f"Unknown dimensions: {sorted(unknown)}"

        )


    # ========================================================
    # NO EMPTY ANCHORS
    # ========================================================

    named_anchors = (
        get_named_anchors()
    )


    for anchor in named_anchors:

        if not str(
            anchor[
                "text"
            ]
        ).strip():

            raise ValueError(

                "Empty semantic anchor detected: "
                f"{anchor['anchor_name']}"

            )


    # ========================================================
    # UNIQUE ANCHOR NAMES
    # ========================================================

    anchor_names = [

        anchor[
            "anchor_name"
        ]

        for anchor
        in named_anchors

    ]


    if (

        len(
            anchor_names
        )

        !=

        len(
            set(
                anchor_names
            )
        )

    ):

        raise ValueError(

            "Duplicate anchor names detected."

        )


    # ========================================================
    # UNIQUE ANCHOR SENTENCES
    # ========================================================

    anchor_texts = [

        anchor[
            "text"
        ]

        for anchor
        in named_anchors

    ]


    if (

        len(
            anchor_texts
        )

        !=

        len(
            set(
                anchor_texts
            )
        )

    ):

        raise ValueError(

            "Duplicate anchor sentences detected."

        )


    return True


# ============================================================
# OPTIONAL HUMAN-READABLE SUMMARY
# ============================================================


def print_feel_space_summary():
    """
    Print a concise description of the complete experiential
    space.

    Useful when debugging or documenting Method B.
    """

    validate_feel_anchors()


    print()

    print(
        "=" * 65
    )

    print(
        "SHARED EXPERIENTIAL / FEEL SPACE"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Embedding model:      "
        f"{FEEL_EMBEDDING_MODEL}"
    )


    print(
        f"Bipolar dimensions:   "
        f"{N_BIPOLAR_AXES}"
    )


    print(
        f"Unipolar dimensions:  "
        f"{N_UNIPOLAR_AXES}"
    )


    print(
        f"Final dimensions:     "
        f"{N_FEEL_DIMENSIONS}"
    )


    print(
        f"Semantic anchors:     "
        f"{N_ANCHORS}"
    )


    print()


    for index, dimension in enumerate(

        FEEL_DIMENSIONS,

        start=1

    ):


        if dimension in BIPOLAR_AXES:

            axis = BIPOLAR_AXES[
                dimension
            ]


            print(

                f"{index:>2}. "
                f"{dimension:<12} | "
                f"{axis['high_label']} "
                f"<-> "
                f"{axis['low_label']}"

            )


        else:

            axis = UNIPOLAR_AXES[
                dimension
            ]


            print(

                f"{index:>2}. "
                f"{dimension:<12} | "
                f"{axis['label']} "
                f"(unipolar)"

            )


    print()


# ============================================================
# ENTRY POINT
# ============================================================
#
# This module is primarily intended to be imported.
#
# Running it directly performs a useful configuration check.
# ============================================================


if __name__ == "__main__":

    print_feel_space_summary()