#src/atlas/cross_domain/domain_vocabulary.py

"""
Domain-specific vocabulary used by the cross-domain semantic pipeline.

The purpose of this module is to identify words that describe the
TYPE / DOMAIN of an item rather than its semantic atmosphere.

For example:

    "science fiction movie with a dark atmosphere"

should ideally contribute:

    "science fiction dark atmosphere"

rather than allowing the word "movie" to become an important
semantic feature.

This vocabulary is intentionally conservative.

We do NOT want to remove every word associated with a domain.
Words such as "jazz", "guitar", "sushi", etc. can still carry
meaningful semantic or cultural information.

The lists below therefore focus primarily on generic structural
vocabulary such as "movie", "artist", "restaurant", "album", etc.
"""


# =========================================================
# MOVIE VOCABULARY
# =========================================================

MOVIE_VOCABULARY = {

    # Generic medium / object terms
    "movie",
    "movies",
    "film",
    "films",
    "cinema",

    # People involved in production
    "actor",
    "actors",
    "actress",
    "actresses",
    "director",
    "directors",
    "producer",
    "producers",
    "screenwriter",
    "screenwriters",
    "writer",
    "writers",
    "cast",

    # Narrative / structural terminology
    "character",
    "characters",
    "plot",
    "story",
    "stories",
    "narrative",
    "narratives",
    "screenplay",
    "screenplays",
    "script",
    "scripts",

    # Television / episodic terminology
    "show",
    "shows",
    "series",
    "episode",
    "episodes",
    "season",
    "seasons",

    # Film-related terminology
    "documentary",
    "documentaries",
    "animation",
    "animated",
    "sequel",
    "sequels",
    "prequel",
    "prequels",
    "remake",
    "remakes",
    "adaptation",
    "adaptations",

}


# =========================================================
# MUSIC VOCABULARY
# =========================================================

MUSIC_VOCABULARY = {

    # Generic medium / object terms
    "music",
    "song",
    "songs",
    "track",
    "tracks",
    "record",
    "records",
    "recording",
    "recordings",

    # People / entities
    "artist",
    "artists",
    "band",
    "bands",
    "singer",
    "singers",
    "vocalist",
    "vocalists",
    "musician",
    "musicians",
    "composer",
    "composers",

    # Releases
    "album",
    "albums",
    "single",
    "singles",
    "ep",
    "eps",

    # Performance terminology
    "concert",
    "concerts",
    "live",
    "performance",
    "performances",
    "performer",
    "performers",

    # Music industry terminology
    "label",
    "labels",
    "discography",
    "discographies",

    # Generic musical terminology
    "vocal",
    "vocals",
    "instrument",
    "instruments",

}


# =========================================================
# RESTAURANT VOCABULARY
# =========================================================

RESTAURANT_VOCABULARY = {

    # Generic establishment terms
    "restaurant",
    "restaurants",
    "dining",
    "eatery",
    "eateries",
    "cafe",
    "cafes",
    "café",
    "cafés",
    "bar",
    "bars",

    # Food-service terminology
    "food",
    "meal",
    "meals",
    "dish",
    "dishes",
    "menu",
    "menus",
    "chef",
    "chefs",

    # Service terminology
    "waiter",
    "waiters",
    "waitress",
    "waitresses",
    "staff",
    "service",
    "table",
    "tables",
    "reservation",
    "reservations",

    # Generic establishment terminology
    "diner",
    "diners",
    "brunch",
    "takeout",
    "takeaway",
    "delivery",

}


# =========================================================
# DOMAIN VOCABULARY REGISTRY
# =========================================================

DOMAIN_VOCABULARY = {

    "movies": MOVIE_VOCABULARY,

    "music": MUSIC_VOCABULARY,

    "restaurants": RESTAURANT_VOCABULARY,

}


# =========================================================
# GET VOCABULARY FOR DOMAINS
# =========================================================

def get_domain_vocabulary(
    domains
):
    """
    Return the combined domain-specific vocabulary for
    a collection of domains.

    Parameters
    ----------
    domains : list[str]
        Domains participating in the cross-domain atlas.

        Example:
            ["movies", "music"]

    Returns
    -------
    set[str]
        Combined vocabulary for all requested domains.
    """

    vocabulary = set()

    for domain in domains:

        domain = domain.lower().strip()

        if domain not in DOMAIN_VOCABULARY:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: "
                f"{list(DOMAIN_VOCABULARY.keys())}"
            )

        vocabulary.update(
            DOMAIN_VOCABULARY[domain]
        )

    return vocabulary


# =========================================================
# CHECK WHETHER A WORD IS DOMAIN-SPECIFIC
# =========================================================

def is_domain_specific(
    word,
    domains
):
    """
    Check whether a word belongs to the vocabulary
    associated with one or more domains.

    Parameters
    ----------
    word : str
        Word to check.

    domains : list[str]
        Domains participating in the atlas.

    Returns
    -------
    bool
        True if the word is considered domain-specific.
    """

    vocabulary = get_domain_vocabulary(
        domains
    )

    return (
        word.lower().strip()
        in vocabulary
    )