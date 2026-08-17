#data/data_scraping/Critique_Brainz_Scraping.py

# ============================================================
# CritiqueBrainz -> Music Artist Reviews
#
# Purpose:
#   1. Download CritiqueBrainz reviews
#   2. Keep artist reviews directly
#   3. Resolve album/release-group reviews to their artist
#   4. Match those artists against your MusicBrainz/Last.fm dataset
#   5. Save checkpoints so the script can safely resume
#
# The script is designed to survive:
#   - MusicBrainz 503 errors
#   - rate limiting
#   - temporary network failures
#   - interruptions / Ctrl+C
#   - stopping and restarting the script
#
# ============================================================

import requests
import pandas as pd
import json
import time
import os
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

CRITIQUEBRAINZ_URL = "https://critiquebrainz.org/ws/1/review/"
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2"

INPUT_CSV = "music_data.csv"

OUTPUT_REVIEWS = "critiquebrainz_all_reviews.json"
OUTPUT_MATCHED = "critiquebrainz_matched_reviews.csv"

# Checkpoint files
CHECKPOINT_REVIEWS = "critiquebrainz_checkpoint.json"
CHECKPOINT_CACHE = "musicbrainz_artist_cache.json"
FAILED_LOOKUPS = "musicbrainz_failed_lookups.json"

# How many CritiqueBrainz reviews to retrieve per API request
CB_LIMIT = 50

# Save progress every N processed reviews
CHECKPOINT_EVERY = 50

# Only retrieve English reviews
LANGUAGE = "en"

# Maximum number of attempts for temporary API errors
MAX_RETRIES = 5

# Base delay between retries
RETRY_BASE_DELAY = 5

# Delay between normal MusicBrainz requests.
#
# MusicBrainz asks API users to be polite and avoid hammering
# the service. Keeping this around 1 second is intentional.
MUSICBRAINZ_DELAY = 1.1

# User-Agent is important when using MusicBrainz.
# Replace this with your own project/contact information if desired.
USER_AGENT = (
    "FYP-MultiDomainAtlas/1.0 "
    "(Data Science MSc project; riehl.antoine@gmail.com)"
)


# ============================================================
# CREATE SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/json"
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path, default):
    """Load JSON file if it exists, otherwise return default."""

    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"Could not load {path}: {e}")
        print("Starting with default value.")

        return default


def save_json(path, data):
    """Safely save JSON."""

    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Atomic replacement
    os.replace(temp_path, path)


def request_with_retry(
    url,
    params=None,
    max_retries=MAX_RETRIES,
    delay=RETRY_BASE_DELAY,
    description=""
):
    """
    Request a URL with retry handling.

    Temporary errors:
        429
        500
        502
        503
        504

    Permanent errors are not repeatedly retried.
    """

    for attempt in range(1, max_retries + 1):

        try:

            response = session.get(
                url,
                params=params,
                timeout=30
            )

            # Successful request
            if response.status_code == 200:
                return response.json()

            # Rate limit
            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = delay * attempt
                else:
                    wait = delay * attempt

                print(
                    f"\nRate limited while requesting "
                    f"{description}."
                )

                print(
                    f"Waiting {wait:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )

                time.sleep(wait)

                continue

            # Temporary server problems
            if response.status_code in {
                500,
                502,
                503,
                504
            }:

                wait = delay * attempt

                print(
                    f"\nTemporary MusicBrainz error "
                    f"{response.status_code} "
                    f"while requesting {description}."
                )

                print(
                    f"Waiting {wait:.1f}s "
                    f"(attempt {attempt}/{max_retries})..."
                )

                time.sleep(wait)

                continue

            # Other HTTP error
            print(
                f"\nPermanent HTTP error "
                f"{response.status_code} "
                f"while requesting {description}."
            )

            return None

        except requests.RequestException as e:

            wait = delay * attempt

            print(
                f"\nNetwork error while requesting "
                f"{description}: {e}"
            )

            print(
                f"Waiting {wait:.1f}s "
                f"(attempt {attempt}/{max_retries})..."
            )

            time.sleep(wait)

    print(
        f"\nFAILED after {max_retries} attempts: "
        f"{description}"
    )

    return None


# ============================================================
# LOAD YOUR MUSIC DATASET
# ============================================================

print("\n==============================================")
print("LOADING MUSIC DATASET")
print("==============================================\n")

music_df = pd.read_csv(INPUT_CSV)

music_df["mbid"] = (
    music_df["mbid"]
    .astype(str)
    .str.strip()
)

print(
    f"Loaded {len(music_df):,} music entities."
)

print(
    f"Unique MBIDs: "
    f"{music_df['mbid'].nunique():,}"
)


# ------------------------------------------------------------
# Create MBID -> artist lookup
# ------------------------------------------------------------

artist_lookup = {}

for _, row in music_df.iterrows():

    mbid = str(row["mbid"]).strip()

    artist_name = row.get("artist_lastfm")

    if pd.isna(artist_name):
        artist_name = row.get("artist_mb")

    if pd.isna(artist_name):
        artist_name = None

    artist_lookup[mbid] = artist_name


music_mbids = set(artist_lookup.keys())


# ============================================================
# LOAD / DOWNLOAD CRITIQUEBRAINZ REVIEWS
# ============================================================

print("\n==============================================")
print("CRITIQUEBRAINZ REVIEWS")
print("==============================================\n")


checkpoint = load_json(
    CHECKPOINT_REVIEWS,
    {
        "offset": 0,
        "reviews": []
    }
)

all_reviews = checkpoint.get(
    "reviews",
    []
)

offset = checkpoint.get(
    "offset",
    0
)

print(
    f"Checkpoint contains "
    f"{len(all_reviews):,} reviews."
)

print(
    f"Resuming CritiqueBrainz download "
    f"from offset {offset:,}."
)


while True:

    params = {
        "limit": CB_LIMIT,
        "offset": offset,
        "language": LANGUAGE
    }

    data = request_with_retry(
        CRITIQUEBRAINZ_URL,
        params=params,
        description=f"CritiqueBrainz offset {offset}"
    )

    if data is None:

        print(
            "\nCritiqueBrainz request failed."
        )

        print(
            "Saving current progress before stopping."
        )

        save_json(
            CHECKPOINT_REVIEWS,
            {
                "offset": offset,
                "reviews": all_reviews
            }
        )

        break

    reviews = data.get(
        "reviews",
        []
    )

    total = data.get(
        "count",
        0
    )

    if not reviews:
        break

    all_reviews.extend(reviews)

    offset += len(reviews)

    print(
        f"Downloaded "
        f"{len(all_reviews):,} / {total:,} "
        f"CritiqueBrainz reviews"
    )

    # Save CritiqueBrainz download checkpoint
    save_json(
        CHECKPOINT_REVIEWS,
        {
            "offset": offset,
            "reviews": all_reviews
        }
    )

    if offset >= total:
        break

    time.sleep(0.5)


print(
    f"\nTotal CritiqueBrainz reviews available: "
    f"{len(all_reviews):,}"
)


# ============================================================
# SAVE COMPLETE CRITIQUEBRAINZ DATA
# ============================================================

save_json(
    OUTPUT_REVIEWS,
    all_reviews
)

print(
    f"Saved raw reviews to: "
    f"{OUTPUT_REVIEWS}"
)


# ============================================================
# LOAD MUSICBRAINZ CACHE
# ============================================================

print("\n==============================================")
print("LOADING MUSICBRAINZ CACHE")
print("==============================================\n")


artist_cache = load_json(
    CHECKPOINT_CACHE,
    {}
)

failed_lookups = load_json(
    FAILED_LOOKUPS,
    []
)

failed_lookup_set = set(
    failed_lookups
)

print(
    f"Cached MusicBrainz lookups: "
    f"{len(artist_cache):,}"
)

print(
    f"Previously failed lookups: "
    f"{len(failed_lookup_set):,}"
)


# ============================================================
# FUNCTIONS FOR MUSICBRAINZ RESOLUTION
# ============================================================

def get_release_group_artist(release_group_mbid):
    """
    Resolve a MusicBrainz release-group MBID
    to its main artist MBID and artist name.
    """

    cache_key = (
        f"release_group:{release_group_mbid}"
    )

    # Already cached
    if cache_key in artist_cache:

        return artist_cache[cache_key]

    # Previously failed
    if cache_key in failed_lookup_set:

        return None

    url = (
        f"{MUSICBRAINZ_URL}/release-group/"
        f"{release_group_mbid}"
    )

    params = {
        "fmt": "json",
        "inc": "artist-credits"
    }

    data = request_with_retry(
        url,
        params=params,
        description=(
            f"release-group "
            f"{release_group_mbid}"
        )
    )

    # Request failed
    if data is None:

        failed_lookup_set.add(
            cache_key
        )

        failed_lookups.append(
            cache_key
        )

        save_json(
            FAILED_LOOKUPS,
            failed_lookups
        )

        return None

    # --------------------------------------------------------
    # Extract artist
    # --------------------------------------------------------

    artist_credit = data.get(
        "artist-credit",
        []
    )

    if not artist_credit:

        result = None

    else:

        artist = artist_credit[0].get(
            "artist",
            {}
        )

        artist_mbid = artist.get(
            "id"
        )

        artist_name = artist.get(
            "name"
        )

        if artist_mbid:

            result = {
                "artist_mbid": artist_mbid,
                "artist_name": artist_name
            }

        else:

            result = None

    # Cache result
    artist_cache[cache_key] = result

    save_json(
        CHECKPOINT_CACHE,
        artist_cache
    )

    time.sleep(
        MUSICBRAINZ_DELAY
    )

    return result


# ============================================================
# PROCESS REVIEWS
# ============================================================

print("\n==============================================")
print("PROCESSING REVIEWS")
print("==============================================\n")


# ------------------------------------------------------------
# Resume checkpoint for processed reviews
# ------------------------------------------------------------

processing_checkpoint = load_json(
    "review_processing_checkpoint.json",
    {
        "index": 0,
        "processed": []
    }
)

start_index = processing_checkpoint.get(
    "index",
    0
)

processed_reviews = processing_checkpoint.get(
    "processed",
    []
)

print(
    f"Resuming review processing from "
    f"{start_index:,} / {len(all_reviews):,}"
)

print(
    f"Already processed: "
    f"{len(processed_reviews):,}"
)


# ------------------------------------------------------------
# Main processing loop
# ------------------------------------------------------------

for i in range(
    start_index,
    len(all_reviews)
):

    review = all_reviews[i]

    print(
        f"\rProcessing review "
        f"{i + 1:,}/{len(all_reviews):,}",
        end=""
    )

    entity_id = review.get(
        "entity_id"
    )

    entity_type = review.get(
        "entity_type"
    )

    if not entity_id:

        continue

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    revision = review.get(
        "last_revision",
        {}
    )

    text = revision.get(
        "text"
    )

    rating = revision.get(
        "rating"
    )

    if not text:

        continue

    # --------------------------------------------------------
    # Determine artist
    # --------------------------------------------------------

    artist_mbid = None
    artist_name = None

    # --------------------------------------------------------
    # DIRECT ARTIST REVIEW
    # --------------------------------------------------------

    if entity_type == "artist":

        artist_mbid = entity_id

        # The review itself is an artist review.
        #
        # We do not necessarily need another API call here.
        #
        # We can obtain the artist name later through
        # matching / optional MusicBrainz enrichment.

    # --------------------------------------------------------
    # RELEASE GROUP / ALBUM REVIEW
    # --------------------------------------------------------

    elif entity_type in {
        "release_group",
        "release"
    }:

        result = None

        if entity_type == "release_group":

            result = get_release_group_artist(
                entity_id
            )

        else:

            # ------------------------------------------------
            # Release -> release-group -> artist
            # ------------------------------------------------

            release_cache_key = (
                f"release:{entity_id}"
            )

            if (
                release_cache_key
                in artist_cache
            ):

                result = artist_cache[
                    release_cache_key
                ]

            elif (
                release_cache_key
                not in failed_lookup_set
            ):

                url = (
                    f"{MUSICBRAINZ_URL}/release/"
                    f"{entity_id}"
                )

                params = {
                    "fmt": "json",
                    "inc": "release-groups+artist-credits"
                }

                data = request_with_retry(
                    url,
                    params=params,
                    description=(
                        f"release "
                        f"{entity_id}"
                    )
                )

                if data is None:

                    failed_lookup_set.add(
                        release_cache_key
                    )

                    failed_lookups.append(
                        release_cache_key
                    )

                    save_json(
                        FAILED_LOOKUPS,
                        failed_lookups
                    )

                    result = None

                else:

                    release_group = data.get(
                        "release-group",
                        {}
                    )

                    artist_credit = (
                        release_group.get(
                            "artist-credit",
                            []
                        )
                    )

                    if artist_credit:

                        artist = (
                            artist_credit[0]
                            .get("artist", {})
                        )

                        mbid = artist.get(
                            "id"
                        )

                        name = artist.get(
                            "name"
                        )

                        if mbid:

                            result = {
                                "artist_mbid": mbid,
                                "artist_name": name
                            }

                        else:

                            result = None

                    else:

                        result = None

                    artist_cache[
                        release_cache_key
                    ] = result

                    save_json(
                        CHECKPOINT_CACHE,
                        artist_cache
                    )

                    time.sleep(
                        MUSICBRAINZ_DELAY
                    )

            else:

                result = None

        if result:

            artist_mbid = result.get(
                "artist_mbid"
            )

            artist_name = result.get(
                "artist_name"
            )

    # --------------------------------------------------------
    # OTHER ENTITY TYPES
    # --------------------------------------------------------

    else:

        # Unknown entity type.
        # Keep the review in the raw dataset,
        # but don't attempt to attach it to an artist.

        continue

    # --------------------------------------------------------
    # MATCH AGAINST OUR MUSIC DATASET
    # --------------------------------------------------------

    matched = False

    matched_artist_name = None

    if artist_mbid:

        if artist_mbid in music_mbids:

            matched = True

            matched_artist_name = (
                artist_lookup.get(
                    artist_mbid
                )
            )

    # --------------------------------------------------------
    # Save processed review
    # --------------------------------------------------------

    processed_reviews.append({

        "review_id": review.get(
            "id"
        ),

        "original_entity_id": entity_id,

        "original_entity_type": entity_type,

        "artist_mbid": artist_mbid,

        "artist_name_musicbrainz": artist_name,

        "artist_name_dataset": matched_artist_name,

        "matched": matched,

        "language": review.get(
            "language"
        ),

        "created": review.get(
            "created"
        ),

        "rating": rating,

        "text": text,

        "info_url": review.get(
            "info_url"
        )
    })

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if (
        (i + 1) % CHECKPOINT_EVERY == 0
        or i == len(all_reviews) - 1
    ):

        save_json(
            "review_processing_checkpoint.json",
            {
                "index": i + 1,
                "processed": processed_reviews
            }
        )

        save_json(
            CHECKPOINT_CACHE,
            artist_cache
        )

        save_json(
            FAILED_LOOKUPS,
            failed_lookups
        )

        print(
            f"\nCheckpoint saved at "
            f"{i + 1:,} reviews."
        )


# ============================================================
# SAVE FINAL PROCESSED DATA
# ============================================================

print("\n\n==============================================")
print("SAVING FINAL DATA")
print("==============================================\n")


save_json(
    OUTPUT_REVIEWS,
    processed_reviews
)


# ============================================================
# CREATE MATCHED DATAFRAME
# ============================================================

matched_reviews = [
    review
    for review in processed_reviews
    if review.get("matched")
]


matched_df = pd.DataFrame(
    matched_reviews
)


if len(matched_df) > 0:

    matched_df.to_csv(
        OUTPUT_MATCHED,
        index=False,
        encoding="utf-8"
    )

else:

    # Still create an empty CSV with the expected columns
    matched_df = pd.DataFrame(
        columns=[
            "review_id",
            "original_entity_id",
            "original_entity_type",
            "artist_mbid",
            "artist_name_musicbrainz",
            "artist_name_dataset",
            "matched",
            "language",
            "created",
            "rating",
            "text",
            "info_url"
        ]
    )

    matched_df.to_csv(
        OUTPUT_MATCHED,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# SUMMARY
# ============================================================

unique_matched_artists = (
    matched_df["artist_mbid"].nunique()
    if len(matched_df) > 0
    else 0
)


direct_artist_reviews = sum(
    1
    for r in processed_reviews
    if r.get("original_entity_type")
    == "artist"
)


album_reviews = sum(
    1
    for r in processed_reviews
    if r.get("original_entity_type")
    in {
        "release",
        "release_group"
    }
)


print("\n==============================================")
print("CRITIQUEBRAINZ SUMMARY")
print("==============================================")

print(
    f"Total downloaded reviews:     "
    f"{len(all_reviews):,}"
)

print(
    f"Processed reviews:             "
    f"{len(processed_reviews):,}"
)

print(
    f"Direct artist reviews:         "
    f"{direct_artist_reviews:,}"
)

print(
    f"Album/release reviews:          "
    f"{album_reviews:,}"
)

print(
    f"Matched reviews:                "
    f"{len(matched_reviews):,}"
)

print(
    f"Unique matched artists:         "
    f"{unique_matched_artists:,}"
)

print(
    f"Your music entities:             "
    f"{len(music_df):,}"
)

print(
    f"MusicBrainz cache entries:      "
    f"{len(artist_cache):,}"
)

print(
    f"Failed API lookups:             "
    f"{len(failed_lookups):,}"
)

print("\nOutput files:")
print(
    f"  - {OUTPUT_REVIEWS}"
)

print(
    f"  - {OUTPUT_MATCHED}"
)

print(
    f"  - {CHECKPOINT_REVIEWS}"
)

print(
    f"  - {CHECKPOINT_CACHE}"
)

print(
    f"  - {FAILED_LOOKUPS}"
)

print(
    "\nDone."
)
