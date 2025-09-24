from typing import Union, List
import pandas as pd


def get_top_n(
    data: pd.DataFrame,
    sort_by: Union[str, List[str]],
    n: int = 10,
    direction: str = "top"
) -> pd.DataFrame:
    """
    Get top or bottom N rows sorted by specified column(s)

    Examples:
    get_top_n(movies, 'imdb_score', n=5)  # Top 5 highest rated
    get_top_n(movies, 'imdb_score', n=5, direction='bottom')  # Bottom 5
    get_top_n(movies, ['imdb_score', 'gross'], n=10)  # Sort by score, then gross
    get_top_n(olympics, 'Year', n=20, columns=['Name', 'Sport', 'Medal'])  # Custom columns
    """

    # Handle single sort column vs multiple
    if isinstance(sort_by, str):
        sort_by = [sort_by]

    # Determine sort order
    ascending = True if direction == "bottom" else False

    # Sort the data
    sorted_data = data.sort_values(sort_by, ascending=ascending)

    # Select top N
    result = sorted_data.head(n)

    return result
