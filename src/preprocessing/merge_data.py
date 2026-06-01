def compute_movie_stats(ratings):

    movie_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std")
    ).reset_index()

    return movie_stats


def merge_movies(movie_stats, movies):

    merged = movie_stats.merge(
        movies,
        on="movieId",
        how="left"
    )

    return merged