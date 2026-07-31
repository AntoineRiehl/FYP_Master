//frontend/src/types/atlas.ts

export type AtlasNode = {
    id: number;

    label: string;

    umap_x: number;
    umap_y: number;

    cluster: number;
    cluster_label?: string;

    visual_size: number;

    popularity?: number;
    category?: string;

    rating?: number;
    description?: string;
};

export type RegionNode = {
    cluster: number;

    umap_x: number;
    umap_y: number;

    cluster_label: string;

    visual_size: number;

    movie_count?: number;
    artist_count?: number;
    restaurant_count?: number;
};

export type AtlasData = {
    atlas: AtlasNode[];
    landmarks: AtlasNode[];
    regions: RegionNode[];
};