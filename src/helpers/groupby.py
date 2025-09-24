from typing import Union, List
import pandas as pd


def summarize_by_group(
    data: pd.DataFrame,
    group_by: str,
    metrics: Union[str, List[str]],
    func: str = "mean"
) -> pd.DataFrame:
    """
    Group data and calculate summary statistics

    Examples:
    summarize_by_group(movies, 'genres', 'imdb_score')
    summarize_by_group(movies, 'director_name', ['imdb_score', 'gross'], 'mean')
    summarize_by_group(olympics, 'NOC', 'Medal', 'count')
    summarize_by_group(spotify, 'track_genre', ['danceability', 'energy'])
    """

    # Handle single metric vs list of metrics
    if isinstance(metrics, str):
        metrics = [metrics]

    # Handle the pesky comma-separated genres
    if group_by == "genres":  # Movies dataset specific
        # Split genres and explode so each row represents one genre
        exploded = data.copy()
        exploded["genres"] = exploded["genres"].str.split("|")
        exploded = exploded.explode("genres")
        grouped = exploded.groupby("genres")[metrics]
    else:
        grouped = data.groupby(group_by)[metrics]

    # Apply the aggregation function
    if func == "mean":
        result = grouped.mean()
    elif func == "count":
        result = grouped.size().to_frame("count")
    elif func == "sum":
        result = grouped.sum()
    else:
        result = grouped.agg(func)

    return result.sort_values(metrics[0], ascending=False)
