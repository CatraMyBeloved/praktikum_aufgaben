from typing import Any
import pandas as pd
import numpy as np


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
    safe_dict.update({"__builtins__": {}, "np": np})  # Add numpy for where function

    for col_name, expression in new_cols.items():
        try:
            # Handle conditional expressions that use if/else
            if " if " in expression and " else " in expression:
                # Parse the conditional expression manually
                parts = expression.split(" if ")
                if len(parts) == 2:
                    true_val = parts[0].strip().strip("'\"")
                    condition_and_false = parts[1].split(" else ")
                    if len(condition_and_false) == 2:
                        condition = condition_and_false[0].strip()
                        false_val = condition_and_false[1].strip().strip("'\"")

                        # Evaluate condition
                        condition_result = eval(condition, safe_dict)
                        result[col_name] = np.where(condition_result, true_val, false_val)
                        continue

            # Standard expression evaluation
            result[col_name] = eval(expression, safe_dict)
        except Exception as e:
            print(f"Error creating column '{col_name}': {e}")

    return result
