#data/data_scraping/Critique_Brainz_Scraping_old.py

"""
build_critique_brainz_dataset.py

Downloads English textual music reviews from CritiqueBrainz,
resolves reviewed release groups to their artists through MusicBrainz,
and matches those artists against the existing music atlas dataset.

INPUT:
    music_data.csv

Expected important column:
    mbid

OUTPUT:
    data/critique_brainz/
        critiquebrainz_reviews_raw.csv
        critiquebrainz_reviews_matched.csv
        critiquebrainz_artist_reviews.csv
        critiquebrainz_release_groups.csv
        critiquebrainz_match_report.txt
"""

import os
import time
import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_MUSIC_CSV = "music_data.csv"

OUTPUT_DIR = "scraped_data/critique_brainz"

CRITIQUEBRAINZ_API = "https://critiquebrainz.org/ws/1/review/"
MUSICBRAINZ_API = "https://musicbrainz.org/ws/2/release-group/"

# CritiqueBrainz API allows a maximum of 50.
PAGE_SIZE = 50

# We only want English reviews for the first version.
LANGUAGE = "en"

# MusicBrainz asks API users to identify themselves.
# Change this to your own project information/email.
USER_AGENT = (
    "FYP-MusicAtlas/1.0 "
    "(academic MSc Data Science project; contact: riehl.antoine@gmail.com)"
)

# MusicBrainz asks clients to stay around 1 request/second.
MUSICBRAINZ_DELAY = 1.1

# Save progress periodically so that an interrupted run
# does not necessarily lose everything.
SAVE_EVERY = 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalise_mbid(value):
    """
    Normalise a MusicBrainz ID for safe comparison.
    """
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if not value:
        return None

    return value


def get_json(session, url, params=None, retries=5):
    """
    GET a JSON response with basic retry handling.
    """

    for attempt in range(retries):

        try:
            response = session.get(
                url,
                params=params,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()

            # Rate limiting
            if response.status_code == 429:
                wait_time = 10 * (attempt + 1)

                print(
                    f"Rate limited (429). "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            # Temporary server problems
            if response.status_code >= 500:
                wait_time = 5 * (attempt + 1)

                print(
                    f"Server error {response.status_code}. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

        except requests.RequestException as error:

            wait_time = 5 * (attempt + 1)

            print(
                f"Request failed: {error}. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Could not retrieve data from {url}"
    )


# ============================================================
# STEP 1 — LOAD YOUR EXISTING MUSIC ATLAS
# ============================================================

def load_music_dataset():

    print("\n" + "=" * 70)
    print("STEP 1 — Loading existing music dataset")
    print("=" * 70)

    df = pd.read_csv(INPUT_MUSIC_CSV)

    if "mbid" not in df.columns:
        raise ValueError(
            "Your CSV does not contain an 'mbid' column."
        )

    df["mbid"] = df["mbid"].apply(normalise_mbid)

    # Remove rows without a valid MBID
    df = df[df["mbid"].notna()].copy()

    # Keep only unique artists
    df = df.drop_duplicates(subset=["mbid"])

    print(f"Artists in existing dataset: {len(df):,}")

    return df


# ============================================================
# STEP 2 — DOWNLOAD CRITIQUEBRAINZ REVIEWS
# ============================================================

def download_critiquebrainz_reviews():

    print("\n" + "=" * 70)
    print("STEP 2 — Downloading CritiqueBrainz reviews")
    print("=" * 70)

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    })

    all_reviews = []

    offset = 0
    total = None

    while True:

        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "entity_type": "musicbrainz",
            "review_type": "review",
            "language": LANGUAGE,
            "sort": "published_on",
            "sort_order": "desc"
        }

        data = get_json(
            session,
            CRITIQUEBRAINZ_API,
            params=params
        )

        reviews = data.get("reviews", [])

        if total is None:
            total = data.get("count", 0)

            print(
                f"CritiqueBrainz reports "
                f"{total:,} matching reviews."
            )

        if not reviews:
            break

        for review in reviews:

            # We only care about music entities.
            # entity_type should identify what is being reviewed.
            entity_type = review.get("entity_type")

            # Keep the review even if rating is missing.
            # Text is our main target.
            review_text = review.get("text")

            if not review_text:
                continue

            all_reviews.append({
                "review_id": review.get("id"),
                "entity_id": normalise_mbid(
                    review.get("entity_id")
                ),
                "entity_type": entity_type,
                "review_text": review_text,
                "rating": review.get("rating"),
                "language": review.get("language"),
                "created": review.get("created"),
                "last_updated": review.get("last_updated"),
                "source": review.get("source"),
                "source_url": review.get("source_url"),
                "review_popularity": review.get("popularity"),
                "votes_positive": (
                    review.get("votes", {})
                    .get("positive", 0)
                ),
                "votes_negative": (
                    review.get("votes", {})
                    .get("negative", 0)
                ),
                "license": (
                    review.get("license", {})
                    .get("id")
                )
            })

        offset += PAGE_SIZE

        print(
            f"Downloaded {len(all_reviews):,} reviews "
            f"(offset {offset:,}/{total:,})"
        )

        # Be polite to CritiqueBrainz.
        time.sleep(0.2)

        # Save intermediate progress.
        if len(all_reviews) % SAVE_EVERY < PAGE_SIZE:

            temp_df = pd.DataFrame(all_reviews)

            temp_path = os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_reviews_raw_progress.csv"
            )

            temp_df.to_csv(
                temp_path,
                index=False,
                encoding="utf-8"
            )

    df_reviews = pd.DataFrame(all_reviews)

    print(
        f"\nFinished downloading "
        f"{len(df_reviews):,} textual reviews."
    )

    return df_reviews


# ============================================================
# STEP 3 — KEEP RELEASE-GROUP REVIEWS
# ============================================================

def filter_release_group_reviews(df_reviews):

    print("\n" + "=" * 70)
    print("STEP 3 — Selecting release-group reviews")
    print("=" * 70)

    print("\nReview entity types found:")

    print(
        df_reviews["entity_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    # For your current artist atlas, release groups are
    # the most useful starting point because they correspond
    # to album-level entities.
    df_release = df_reviews[
        df_reviews["entity_type"] == "release_group"
    ].copy()

    print(
        f"\nRelease-group reviews: "
        f"{len(df_release):,}"
    )

    return df_release


# ============================================================
# STEP 4 — RESOLVE RELEASE GROUP → ARTIST USING MUSICBRAINZ
# ============================================================

def get_release_group_metadata(
    session,
    release_group_mbid
):

    params = {
        "inc": "artists",
        "fmt": "json"
    }

    url = (
        f"{MUSICBRAINZ_API}"
        f"{release_group_mbid}"
    )

    data = get_json(
        session,
        url,
        params=params
    )

    artists = data.get("artist-credit", [])

    artist_results = []

    for artist_credit in artists:

        artist = artist_credit.get("artist", {})

        artist_mbid = normalise_mbid(
            artist.get("id")
        )

        artist_name = artist.get("name")

        if artist_mbid:

            artist_results.append({
                "artist_mbid": artist_mbid,
                "artist_name": artist_name
            })

    return {
        "release_group_mbid": release_group_mbid,
        "album_title": data.get("title"),
        "first_release_date": data.get(
            "first-release-date"
        ),
        "artist_results": artist_results
    }


def resolve_release_groups(df_release):

    print("\n" + "=" * 70)
    print("STEP 4 — Resolving release groups through MusicBrainz")
    print("=" * 70)

    unique_release_groups = (
        df_release["entity_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique release groups to resolve: "
        f"{len(unique_release_groups):,}"
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    })

    resolved_rows = []

    for index, release_group_mbid in enumerate(
        unique_release_groups,
        start=1
    ):

        try:

            metadata = get_release_group_metadata(
                session,
                release_group_mbid
            )

            artists = metadata["artist_results"]

            # An album can have multiple credited artists.
            # We therefore create one row per artist credit.
            if artists:

                for artist in artists:

                    resolved_rows.append({
                        "release_group_mbid": (
                            release_group_mbid
                        ),
                        "album_title": (
                            metadata["album_title"]
                        ),
                        "first_release_date": (
                            metadata["first_release_date"]
                        ),
                        "artist_mbid": (
                            artist["artist_mbid"]
                        ),
                        "artist_name_mb": (
                            artist["artist_name"]
                        )
                    })

            else:

                resolved_rows.append({
                    "release_group_mbid": (
                        release_group_mbid
                    ),
                    "album_title": (
                        metadata["album_title"]
                    ),
                    "first_release_date": (
                        metadata["first_release_date"]
                    ),
                    "artist_mbid": None,
                    "artist_name_mb": None
                })

        except Exception as error:

            print(
                f"\nWARNING: Could not resolve "
                f"{release_group_mbid}: {error}"
            )

        if index % 10 == 0 or index == 1:

            print(
                f"Resolved "
                f"{index:,}/{len(unique_release_groups):,}"
            )

        # MusicBrainz requests should be spaced out.
        time.sleep(MUSICBRAINZ_DELAY)

    df_mapping = pd.DataFrame(resolved_rows)

    print(
        f"\nResolved mappings: "
        f"{len(df_mapping):,}"
    )

    return df_mapping


# ============================================================
# STEP 5 — MATCH REVIEWS TO YOUR EXISTING ARTISTS
# ============================================================

def match_reviews_to_existing_artists(
    df_reviews,
    df_release_mapping,
    df_music
):

    print("\n" + "=" * 70)
    print("STEP 5 — Matching reviews to existing music atlas")
    print("=" * 70)

    # Merge reviews with release-group → artist mapping.
    df = df_reviews.merge(
        df_release_mapping,
        left_on="entity_id",
        right_on="release_group_mbid",
        how="left"
    )

    # Match MusicBrainz artist IDs against your existing atlas.
    df = df.merge(
        df_music[
            [
                "mbid",
                "artist_mb",
                "artist_lastfm",
                "listeners_lastfm",
                "scrobbles_lastfm"
            ]
        ],
        left_on="artist_mbid",
        right_on="mbid",
        how="left"
    )

    # An exact MBID match is the only match we accept here.
    df["matched_to_atlas"] = df["mbid"].notna()

    matched = df[
        df["matched_to_atlas"]
    ].copy()

    print(
        f"Total release-group reviews: "
        f"{len(df):,}"
    )

    print(
        f"Reviews matched to existing artists: "
        f"{len(matched):,}"
    )

    print(
        f"Reviews not matched: "
        f"{len(df) - len(matched):,}"
    )

    unique_artists = matched["mbid"].nunique()

    print(
        f"Existing atlas artists enriched: "
        f"{unique_artists:,}"
    )

    return df, matched


# ============================================================
# STEP 6 — CREATE ARTIST-LEVEL REVIEW CORPUS
# ============================================================

def create_artist_review_corpus(df_matched):

    print("\n" + "=" * 70)
    print("STEP 6 — Creating artist-level review corpus")
    print("=" * 70)

    # Remove accidental duplicate review rows.
    df_matched = df_matched.drop_duplicates(
        subset=["review_id"]
    ).copy()

    # Group all album reviews belonging to an artist.
    grouped = (
        df_matched
        .groupby(
            [
                "mbid",
                "artist_mb",
                "artist_lastfm"
            ],
            dropna=False
        )
        .agg(
            review_count=(
                "review_id",
                "count"
            ),
            average_review_rating=(
                "rating",
                "mean"
            ),
            total_positive_votes=(
                "votes_positive",
                "sum"
            ),
            total_negative_votes=(
                "votes_negative",
                "sum"
            ),
            review_text=(
                "review_text",
                lambda texts: "\n\n".join(
                    str(text).strip()
                    for text in texts
                    if pd.notna(text)
                )
            ),
            albums_reviewed=(
                "album_title",
                lambda titles: " | ".join(
                    sorted(
                        set(
                            str(title)
                            for title in titles
                            if pd.notna(title)
                        )
                    )
                )
            )
        )
        .reset_index()
    )

    print(
        f"Artist-level records created: "
        f"{len(grouped):,}"
    )

    return grouped


# ============================================================
# STEP 7 — SAVE EVERYTHING
# ============================================================

def save_outputs(
    df_raw,
    df_release_mapping,
    df_all_matches,
    df_matched,
    df_artist
):

    print("\n" + "=" * 70)
    print("STEP 7 — Saving datasets")
    print("=" * 70)

    ensure_output_directory()

    paths = {

        "raw":
            os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_reviews_raw.csv"
            ),

        "release_groups":
            os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_release_groups.csv"
            ),

        "all_matches":
            os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_reviews_with_artists.csv"
            ),

        "matched":
            os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_reviews_matched.csv"
            ),

        "artist":
            os.path.join(
                OUTPUT_DIR,
                "critiquebrainz_artist_reviews.csv"
            )
    }

    df_raw.to_csv(
        paths["raw"],
        index=False,
        encoding="utf-8"
    )

    df_release_mapping.to_csv(
        paths["release_groups"],
        index=False,
        encoding="utf-8"
    )

    df_all_matches.to_csv(
        paths["all_matches"],
        index=False,
        encoding="utf-8"
    )

    df_matched.to_csv(
        paths["matched"],
        index=False,
        encoding="utf-8"
    )

    df_artist.to_csv(
        paths["artist"],
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # MATCH REPORT
    # --------------------------------------------------------

    report_path = os.path.join(
        OUTPUT_DIR,
        "critiquebrainz_match_report.txt"
    )

    total_reviews = len(df_all_matches)

    matched_reviews = len(df_matched)

    match_percentage = (
        matched_reviews / total_reviews * 100
        if total_reviews
        else 0
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CRITIQUEBRAINZ MATCH REPORT\n"
        )

        f.write(
            "============================\n\n"
        )

        f.write(
            f"Total textual reviews downloaded: "
            f"{len(df_raw):,}\n"
        )

        f.write(
            f"Release-group reviews: "
            f"{len(df_all_matches):,}\n"
        )

        f.write(
            f"Reviews matched to existing atlas: "
            f"{matched_reviews:,}\n"
        )

        f.write(
            f"Match rate: "
            f"{match_percentage:.2f}%\n"
        )

        f.write(
            f"Existing artists enriched: "
            f"{df_artist['mbid'].nunique():,}\n"
        )

        f.write("\n")

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Matching was performed using exact "
            "MusicBrainz artist MBIDs only. "
            "No fuzzy artist-name matching was used.\n"
        )

    print("\nFiles created:")

    for name, path in paths.items():

        print(
            f"  {name}: {path}"
        )

    print(
        f"  report: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    ensure_output_directory()

    print("\n")
    print("=" * 70)
    print("CRITIQUEBRAINZ MUSIC REVIEW DATASET BUILDER")
    print("=" * 70)

    # 1. Existing music atlas
    df_music = load_music_dataset()

    # 2. CritiqueBrainz reviews
    df_reviews = download_critiquebrainz_reviews()

    # 3. Save raw data immediately
    raw_path = os.path.join(
        OUTPUT_DIR,
        "critiquebrainz_reviews_raw.csv"
    )

    df_reviews.to_csv(
        raw_path,
        index=False,
        encoding="utf-8"
    )

    # 4. Release-group reviews
    df_release = filter_release_group_reviews(
        df_reviews
    )

    # 5. Resolve release groups to artists
    df_release_mapping = resolve_release_groups(
        df_release
    )

    # 6. Match against existing atlas
    (
        df_all_matches,
        df_matched
    ) = match_reviews_to_existing_artists(
        df_release,
        df_release_mapping,
        df_music
    )

    # 7. Create artist-level corpus
    df_artist = create_artist_review_corpus(
        df_matched
    )

    # 8. Save everything
    save_outputs(
        df_reviews,
        df_release_mapping,
        df_all_matches,
        df_matched,
        df_artist
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"Total execution time: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "\nYour main file for the NLP pipeline is:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "critiquebrainz_artist_reviews.csv"
        )
    )


if __name__ == "__main__":
    main()