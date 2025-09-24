import pandas as pd
import os
from typing import Optional


def load_dataset(dataset: str, filtered: bool = True) -> pd.DataFrame:
    """
    Load dataset from the data directory

    Args:
        dataset: Name of dataset ('movies', 'olympics', 'spotify')
        filtered: Whether to load filtered version (default True)

    Returns:
        pd.DataFrame: Loaded dataset
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    dataset_configs = {
        'movies': {
            'original': 'movie_metadata.csv',
            'filtered': 'filtered_movie_metadata_no_na.csv'
        },
        'olympics': {
            'original': 'athlete_events.csv',
            'filtered': 'filtered_athlete_events.csv'
        },
        'spotify': {
            'original': 'spotify_data.csv',
            'filtered': 'filtered_spotify_data.csv'
        }
    }

    if dataset not in dataset_configs:
        available = ', '.join(dataset_configs.keys())
        raise ValueError(f"Dataset '{dataset}' not found. Available: {available}")

    filename = dataset_configs[dataset]['filtered' if filtered else 'original']
    filepath = os.path.join(base_path, dataset, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    return pd.read_csv(filepath)


def load_noc_regions() -> pd.DataFrame:
    """Load NOC regions data for Olympics dataset"""
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'olympics')
    filepath = os.path.join(base_path, 'noc_regions.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NOC regions file not found: {filepath}")

    return pd.read_csv(filepath)