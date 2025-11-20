import pandas as pd

def filter_probability_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to keep rows where the 'probability' column is
    between 0.1 and 0.9 (inclusive).

    Args:
        df: The input pandas DataFrame. It must contain a 'probability' column.

    Returns:
        A new DataFrame with rows filtered based on the probability range.
    """
    if 'probability' not in df.columns:
        raise ValueError("DataFrame must contain a 'probability' column.")

    filtered_df = df[(df['probability'] >= 0.1) & (df['probability'] <= 0.9)]
    return filtered_df