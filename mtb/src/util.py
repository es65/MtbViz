import pandas as pd
import io


def df_info_to_file(
    df: pd.DataFrame,
    file_name: str = "df_info.txt",
    encoding: str = "utf-8",
    include_desc_stats: bool = True,
    include_nulls: bool = True,
    include_sample: bool = True,
    sample_rows: int = 0,
    max_cols: int = None,  # Set to None to show all columns
) -> None:
    """
    Write comprehensive DataFrame information to a file, showing all columns regardless of width.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze
        file_name (str): Output file name
        encoding (str): File encoding
        include_desc_stats (bool): Include descriptive statistics
        include_nulls (bool): Include null value counts
        include_sample (bool): Include a sample of the DataFrame
        sample_rows (int): Number of rows to include in the sample
        max_cols (int): Maximum number of columns to display (None shows all columns)
    """
    with open(file_name, "w", encoding=encoding) as f:
        # Write basic DataFrame info
        f.write(f"DataFrame Information\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)\n")
        f.write(
            f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n\n"
        )

        # Save original display settings
        original_max_columns = pd.get_option("display.max_columns")
        original_max_rows = pd.get_option("display.max_rows")
        original_width = pd.get_option("display.width")

        try:
            # Set display options to show all columns
            pd.set_option("display.max_columns", None if max_cols is None else max_cols)
            pd.set_option("display.max_rows", None)
            pd.set_option("display.width", 1000)

            # Write standard info using buffer
            buffer = io.StringIO()
            df.info(buf=buffer, verbose=True, show_counts=True)
            f.write(f"Column Info:\n{'-' * 80}\n")
            f.write(buffer.getvalue())
            f.write("\n\n")

            # Write null values count if requested
            if include_nulls:
                null_counts = df.isna().sum()
                if null_counts.sum() > 0:
                    f.write(f"Null Value Counts:\n{'-' * 80}\n")
                    # Filter to only show columns with nulls
                    null_cols = null_counts[null_counts > 0]
                    if len(null_cols) > 0:
                        f.write(null_cols.to_string())
                    else:
                        f.write("No null values found.")
                    f.write("\n\n")

            # Write descriptive statistics if requested
            if include_desc_stats:
                f.write(f"Descriptive Statistics:\n{'-' * 80}\n")
                desc_stats = df.describe(include="all").T
                f.write(desc_stats.to_string())
                f.write("\n\n")

            # Write sample rows if requested
            if include_sample and sample_rows > 0:
                f.write(f"DataFrame Sample ({sample_rows} rows):\n{'-' * 80}\n")
                sample = df.head(sample_rows).to_string()
                f.write(sample)
                f.write("\n\n")

        finally:
            # Restore original display settings
            pd.set_option("display.max_columns", original_max_columns)
            pd.set_option("display.max_rows", original_max_rows)
            pd.set_option("display.width", original_width)


def trim_df(
    df: pd.DataFrame,
    x_range: tuple[float, float] = (0, 1000),
    x_col: str = "elapsed_seconds",
) -> pd.DataFrame:
    return df[(df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1])]
