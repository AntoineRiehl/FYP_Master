// frontend/src/types/atlas.ts


// =====================================================
// Reviews
// =====================================================

export type Review = {

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

    year?: number | null;

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


    enrichment?: Record<string, unknown>;

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
 */
export type ColorMode =
    | "domain"
    | "cluster"
    | "category";


// =====================================================
// Atlas filters
// =====================================================

/**
 * Filters applied to the currently loaded atlas.
 *
 * Empty arrays mean "show everything".
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