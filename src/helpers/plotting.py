from typing import Optional, Union, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(
    data: pd.DataFrame,
    plot_type: str,
    x: str,
    y: Optional[str] = None,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    add_trend: bool = False
) -> None:
    """
    Universal plotting function with seaborn styling

    Examples:
    plot(movies, 'distribution', x='imdb_score')
    plot(movies, 'distribution', x='imdb_score', group_by='genres')  # Split by genre
    plot(movies, 'relationship', x='budget', y='gross')
    plot(movies, 'relationship', x='budget', y='gross', group_by='genres')  # Color by genre
    """

    plt.figure(figsize=(12, 6))
    sns.set_palette("husl")  # Nice, colorful palette

    if plot_type == "distribution":
        if data[x].dtype in ["int64", "float64"]:
            if group_by:
                sns.histplot(data=data, x=x, hue=group_by, multiple="dodge")
            else:
                sns.histplot(data=data, x=x)
        else:
            if group_by:
                sns.countplot(data=data, x=x, hue=group_by)
            else:
                sns.countplot(data=data, x=x)
            plt.xticks(rotation=45, ha="right")

    elif plot_type == "relationship":

        if group_by:

            sns.scatterplot(data=data, x=x, y=y, hue=group_by, alpha=0.7)

            if add_trend:
                # Add trend lines for each group

                sns.regplot(
                    data=data,
                    x=x,
                    y=y,
                    scatter=False,
                    color="black",
                    line_kws={"linestyle": "--", "alpha": 0.8},
                )

        else:
            sns.scatterplot(data=data, x=x, y=y, alpha=0.7)
            if add_trend:
                sns.regplot(
                    data=data,
                    x=x,
                    y=y,
                    scatter=False,
                    color="red",
                    line_kws={"linewidth": 2},
                )

    plt.title(title or f"{plot_type.title()} of {x}" + (f" vs {y}" if y else ""))
    plt.tight_layout()
    plt.show()
