//frontend/src/types/movie.ts

export type MovieNode = {
  movieId: number;
  title: string;

  umap_x: number;
  umap_y: number;

  cluster: number;
  cluster_label?: string;

  macro_genre: string;

  rating_count: number;
  weighted_rating: number;

  visual_size: number;
};