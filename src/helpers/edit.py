from typing import Any
import pandas as pd


def add_columns(data: pd.DataFrame, **new_cols: str) -> pd.DataFrame:
    """
    Create new columns using simple expressions

    Examples:
    add_columns(movies,
               decade="title_year // 10 * 10",
               is_long="duration > 120",
               profit="gross - budget")
    """
    result = data.copy()

    # Create safe namespace with just the dataframe columns
    safe_dict = {col: result[col] for col in result.columns}
    safe_dict.update({"__builtins__": {}})  # Remove dangerous builtins

    for col_name, expression in new_cols.items():
        try:
            result[col_name] = eval(expression, safe_dict)
        except Exception as e:
            print(f"Error creating column '{col_name}': {e}")

    return result
