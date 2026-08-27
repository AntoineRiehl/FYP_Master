// frontend/src/types/atlas.ts


// =====================================================
// Reviews
// =====================================================

export type Review = {

    review_id?: string | null;

    text: string;

    source?: string | null;

    rating?: number | null;

    date?: string | null;

};


// =====================================================
// Textual features
// =====================================================

export type TextFeatures = {

    tags: string[];

    categories: string[];

    reviews: Review[];

};


// =====================================================
// Metadata
// =====================================================

export type MediaMetadata = {

    /*
     * Some existing exports contain a numeric year,
     * while others contain a string such as "1995".
     *
     * The frontend accepts both rather than requiring
     * old atlas bundles to be rebuilt.
     */

    year?:
        number
        |
        string
        |
        null;

    country?: string | null;

    language?: string | null;


    // Movie

    director?: string | null;

    actors?: string[];


    // Music

    artist?: string | null;

    album?: string | null;


    // Restaurant

    address?: string | null;

    city?: string | null;

    latitude?: number | null;

    longitude?: number | null;

};


// =====================================================
// Statistics
// =====================================================

export type Statistics = {

    rating?: number | null;

    rating_count?: number | null;

    popularity?: number | null;

};


// =====================================================
// Position
// =====================================================

export type AtlasPosition = {

    x: number;

    y: number;

};


// =====================================================
// Visual information
// =====================================================

export type AtlasVisual = {

    size: number;

    cluster?: number | null;

    cluster_label?: string | null;

};


// =====================================================
// Enrichment
// =====================================================

export type AtlasEnrichment = {

    review_count?: number;

    reviews_used_for_embedding?: number;

    has_review_embedding?: boolean;

};


// =====================================================
// Main Atlas item
// =====================================================

export type AtlasNode = {

    id: string;

    source_id?: string | null;

    title: string;

    domain: string;


    metadata: MediaMetadata;

    text: TextFeatures;

    statistics: Statistics;


    position: AtlasPosition;

    visual: AtlasVisual;


    enrichment?: AtlasEnrichment;

};


// =====================================================
// Region
// =====================================================

export type RegionNode = {

    id: number;

    label: string;

    x: number;

    y: number;

    size: number;

    item_count: number;


    description?: string | null;

    color?: string | null;

};


// =====================================================
// Landmark
// =====================================================

export type LandmarkNode = {

    id: string;

    label: string;

    x: number;

    y: number;


    importance: number;

    cluster?: number | null;


    description?: string | null;

};


// =====================================================
// Feature configuration metadata
// =====================================================

export type FeatureConfig = {

    name: string;

    use_tags: boolean;

    use_categories: boolean;

    use_reviews: boolean;

    use_metadata: boolean;

    use_statistics: boolean;

    use_images: boolean;

    use_external_embeddings: boolean;

};


// =====================================================
// Bundle metadata
// =====================================================

export type AtlasMetadata = {

    domain: string;

    feature_config: FeatureConfig;

    metadata: Record<string, unknown>;

};


// =====================================================
// FEEL PROFILE
// =====================================================

/*
 * These are the 13 standardized experiential dimensions
 * exported by build_frontend_item_details.py.
 *
 * Values are global z-scores calculated across the full
 * semantically defined Movies + Music + Restaurants
 * population.
 */

export type FeelProfile = {

    valence: number;

    activation: number;

    potency: number;

    tension: number;

    warmth: number;

    scale: number;

    tone: number;

    familiarity: number;

    refinement: number;

    complexity: number;

    nostalgia: number;

    wonder: number;

    tenderness: number;

};


// =====================================================
// ITEM SEMANTIC DETAILS
// =====================================================

export type ItemSemanticDetails = {

    source?: string | null;

    has_base_semantics: boolean;

    has_review_semantics: boolean;

};


// =====================================================
// ITEM REVIEW SUMMARY
// =====================================================

export type ItemReviewSummary = {

    /*
     * Number of prepared reviews available for this
     * entity in the frontend-enrichment source data.
     */

    available_count: number;


    /*
     * Number of reviews used when producing the pooled
     * review embedding.
     */

    used_for_embedding: number;


    /*
     * Maximum number of review samples exposed to the
     * frontend for manual inspection.
     */

    sampled_count: number;

};


// =====================================================
// ITEM PROFILE
// =====================================================

export type ItemProfile = {

    /*
     * False for entities excluded from the Feel space
     * because neither usable base semantics nor usable
     * review semantics were available.
     */

    feel_defined: boolean;


    /*
     * Null when feel_defined is false.
     */

    feel:
        FeelProfile
        |
        null;


    semantic:
        ItemSemanticDetails;


    reviews:
        ItemReviewSummary;

};


// =====================================================
// ITEM DETAIL REVIEW
// =====================================================

export type ItemDetailReview = {

    review_id?: string | null;

    text: string;

    rating?: number | null;

    date?: string | null;

    source?: string | null;

};


// =====================================================
// REVIEW DETAIL BUNDLE
// =====================================================

export type ItemReviewBundle = {

    available_count: number;

    sampled_count: number;

    reviews: ItemDetailReview[];

};


// =====================================================
// ITEM DETAILS MANIFEST
// =====================================================

export type ItemDetailsManifest = {

    schema_version: number;

    generated_at_utc: string;


    sharding: {

        shard_count: number;

        algorithm: string;

        profile_pattern: string;

        review_pattern: string;

    };


    feel: {

        available_on_all_atlases: boolean;

        representation: string;

        dimensions: string[];


        scaling: {

            method: string;

            fit_population: string;

            mean:
                Record<
                    string,
                    number
                >;

            scale:
                Record<
                    string,
                    number
                >;

        };

    };


    reviews: {

        sample_size_per_entity: number;

        sampling: string;

        note: string;

    };


    domains:

        Record<
            string,
            {

                entity_count: number;

                feel_defined_count: number;

                entities_with_reviews: number;

                prepared_review_count: number;

            }
        >;

};


// =====================================================
// Visualisation
// =====================================================

/**
 * Determines which semantic/property dimension is used
 * to colour atlas nodes.
 *
 * Domain:
 *     Original source domain of the item.
 *
 * Cluster:
 *     Semantic cluster determined by the atlas pipeline.
 *
 * Category:
 *     Domain-specific category assigned to the item.
 *
 * Feel:
 *     Experiential / Feel-based colouring.
 */
export type ColorMode =
    | "domain"
    | "cluster"
    | "category"
    | "feel";


// =====================================================
// Atlas filters
// =====================================================

/*
 * Legacy/general atlas filter type.
 *
 * The active richer filter implementation currently
 * lives in types/filters.ts.
 */

export type AtlasFilters = {

    domains: string[];

    categories: string[];

};


// =====================================================
// Complete exported bundle
// =====================================================

export type AtlasData = {

    atlas: AtlasNode[];

    landmarks: LandmarkNode[];

    regions: RegionNode[];

    metadata?: AtlasMetadata;

};