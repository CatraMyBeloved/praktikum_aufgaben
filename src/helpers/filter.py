from typing import Any, Union, List
import pandas as pd


def filter_data(data: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """
    Universal filtering using actual column names as kwargs
    Strings use 'contains' matching, other types use exact/list matching

    filter_data(movies, title_year=2010, imdb_score_min=7.0)
    filter_data(movies, genres='Action')  # Movies containing 'Action'
    filter_data(olympics, Year=2008, Sport='Swimming')  # Events containing 'Swimming'
    filter_data(spotify, popularity_min=70, track_genre='pop')
    """
    filtered = data.copy()

    for key, value in filters.items():
        if key.endswith("_min"):
            col = key.replace("_min", "")
            filtered = filtered[filtered[col] >= value]
        elif key.endswith("_max"):
            col = key.replace("_max", "")
            filtered = filtered[filtered[col] <= value]
        elif isinstance(value, list):  # Multiple values
            filtered = filtered[filtered[key].isin(value)]
        elif isinstance(value, str):  # String contains match
            filtered = filtered[filtered[key].str.contains(value, na=False)]
        else:  # Exact match for numbers, etc.
            filtered = filtered[filtered[key] == value]

    return filtered
