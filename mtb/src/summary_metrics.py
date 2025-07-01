from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
import numpy as np
from haversine import haversine, Unit

import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import time
import json
import sys

from mtb.src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config vars:
min_consecutive = config.jump_min_consecutive_points


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process motion data and compute summary metrics"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the 5Hz parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/summaries",
        help="Directory to save processed files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def calculate_distance_haversine_mi(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    smooth: bool = True,
    window_size: int = 21,
) -> float:

    df = df.copy()

    def haversine_series(lat1, lon1, lat2, lon2):
        return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

    if smooth:
        df[lat_col] = df[lat_col].rolling(window_size, center=True).mean()
        df[lon_col] = df[lon_col].rolling(window_size, center=True).mean()

    df["lat_shift"] = df[lat_col].shift()
    df["lon_shift"] = df[lon_col].shift()
    df["step_dist_haversine"] = df.apply(
        lambda row: (
            haversine_series(
                row["lat_shift"], row["lon_shift"], row[lat_col], row[lon_col]
            )
            if pd.notnull(row["lat_shift"])
            else 0
        ),
        axis=1,
    )
    return df["step_dist_haversine"].fillna(0).cumsum().iloc[-1] / config.mile_to_m


def calculate_elevation_gain_ft(
    df: pd.DataFrame,
    y_col: str = "altitude",
    window_size: int = 25,
    threshold: float = 0.0,
) -> float:
    """
    Calculate the total elevation gain from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing elevation data.
        y_col (str): Column name for elevation data.
        threshold (float): Minimum elevation change to count as gain.

    Returns:
        float: Total elevation gain.
    """

    df = df[[y_col]].copy()
    df[y_col] = df[y_col].rolling(window=window_size, center=True).mean()
    df["diff_alt"] = df[y_col].diff()
    return df["diff_alt"].where(df["diff_alt"] > threshold).sum() / config.ft_to_m


def calculate_duration_hr(df: pd.DataFrame, x_col: str = "elapsed_seconds") -> float:
    return round((df[x_col].max() - df[x_col].min()) / 3600, 3)


def generate_moving_df(
    df: pd.DataFrame,
    x_col: str = "elapsed_seconds",
    y_col: str = "speed",
    threshold: float = 0.0,
) -> pd.DataFrame:

    df = df.copy()

    df["diff_secs"] = df[x_col].diff()

    df = df[df[y_col] > threshold].copy()

    df["elapsed_seconds_moving"] = df["diff_secs"].fillna(0).cumsum()

    return df


def calculate_duration_moving_hr(
    df: pd.DataFrame,
    x_col: str = "elapsed_seconds",
    y_col: str = "speed",
    threshold: float = 0.0,
) -> float:
    """
    Calculate the duration in hours when the bike is moving (speed > threshold).

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for elapsed time in seconds
        y_col (str): Column name for speed
        threshold (float): Minimum speed to consider as moving

    Returns:
        float: Duration in hours when moving
    """
    df = df.copy()
    df["diff_secs"] = df[x_col].diff()
    # Only sum time differences when speed is above threshold
    moving_time = df[df[y_col] > threshold]["diff_secs"].fillna(0).sum()
    return round(moving_time / 3600, 3)


def calculate_avg_speed_moving_mph(
    df_moving: pd.DataFrame, x_col: str = "elapsed_seconds", y_col: str = "speed"
) -> float:
    """
    Calculate the average speed. Use moving df.

    Parameters:
        df_moving (pd.DataFrame): Movement only DataFrame
        x_col (str): Column name time used for weighting
        y_col (str): Column name for speed

    Returns:
        float: Average speed when moving
    """

    df_moving = df_moving.copy()
    df_moving["diff_secs"] = df_moving[x_col].diff()
    df_moving["y_weighted"] = df_moving[y_col] * df_moving["diff_secs"]

    return (
        df_moving["y_weighted"].sum() / df_moving["diff_secs"].sum() * config.mps_to_mph
    )


def extract_jump_events(
    df: pd.DataFrame,
    jump_col: str = "jump",
    time_col: str = "elapsed_seconds",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    speed_col: str = "speed",
    min_consecutive: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Must use smoothed data for this function, not raw 20 Hz data!

    Important Note: Uses speed to determine distance, which is low resolution due to limitations of GPS.
    For steep jumps, vertical velocity comes at expense of horizontal velocity, so this methodology
    will overestimate distance. For future update, try integrating acceleration to get better estimate of
    horizontal velocity and distance.

    Minor Note: Consider modifying so time deltas calculated from DatetimeIndex
    and not from elapsed_seconds.

    Extract jump events from the dataframe by identifying consecutive jump points.

    Parameters:
        df (pd.DataFrame): Input DataFrame with jump indicators
        jump_col (str): Column name containing jump indicators (1 for jump, 0 for no jump)
        time_col (str): Column name for time data (seconds)
        lat_col (str): Column name for latitude
        lon_col (str): Column name for longitude
        speed_col (str): Column name for speed
        min_consecutive (int): Minimum number of consecutive jump points to qualify as a jump event

    Returns:
        Optional[pd.DataFrame]: DataFrame with extracted jump events and their properties,
                              or None if no jump events found

    Raises:
        ValueError: If required columns are missing from DataFrame
    """
    # Check required columns exist
    required_cols = [jump_col, time_col]
    optional_cols = [lat_col, lon_col, speed_col]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")

    # Check which optional columns exist
    available_cols = {col: col in df.columns for col in optional_cols}

    # Create a copy with only necessary columns
    df_work = df.copy()

    # Create a group identifier for consecutive jump points
    # This trick identifies runs of consecutive 1s by looking at when the values change
    df_work["jump_change"] = df_work[jump_col].diff().ne(0).cumsum()

    # Group by this identifier to find consecutive sequences
    jump_groups = df_work[df_work[jump_col] == 1].groupby("jump_change")

    # Initialize a list to store jump event data
    jump_events = []

    # Process each jump group
    logger.info("Processing jump groups...")
    for _, group in jump_groups:
        # Skip if fewer than min_consecutive points
        if len(group) < min_consecutive:
            continue

        # Calculate basic properties
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        airtime_s = end_time - start_time

        # Initialize event data
        event = {
            "elapsed_seconds_ride": start_time,
            "airtime_s": airtime_s,
            "points": len(group),
        }

        if available_cols[speed_col]:
            speed_avg = group[speed_col].mean()
            event["speed_mph"] = speed_avg * 2.23694  # Convert m/s to mph
            event["distance_ft"] = (
                airtime_s * event["speed_mph"] * 1.4667
            )  # Convert mph to ft/s

        # Add location data if available (just the starting point)
        if available_cols[lat_col] and available_cols[lon_col]:
            event["latitude"] = group[lat_col].iloc[0]
            event["longitude"] = group[lon_col].iloc[0]

        # Use DataFrame's DatetimeIndex for timestamps if available
        if isinstance(df.index, pd.DatetimeIndex):
            event["datetime"] = group.index[0]

        # Add event to the list
        jump_events.append(event)

    if not jump_events:
        logger.warning("No jump events found.")
        return None

    # Create the jumps DataFrame
    jumps_df = pd.DataFrame(jump_events)

    # Sort by start time
    jumps_df = jumps_df.sort_values("elapsed_seconds_ride").reset_index(drop=True)

    # Log jump statistics
    logger.info(f"Extracted {len(jumps_df)} jump events")
    if "airtime_s" in jumps_df.columns:
        logger.info(f"Average airtime: {jumps_df['airtime_s'].mean():.2f}s")
        logger.info(f"Max airtime: {jumps_df['airtime_s'].max():.2f}s")

    return jumps_df


def calculate_intensity_scores(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the average overall and cornering intensity.
    Assumes evenly spaced datetime index.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        Tuple[float, float]: Average overall and cornering intensity
    """
    cols = ["ar", "ah_e"]

    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")

    avg_overall_intensity = df["ar"].mean()
    avg_cornering_intensity = np.abs(df["ah_e"]).mean()

    return avg_overall_intensity, avg_cornering_intensity


def calculate_top_3_events(
    df_moving: pd.DataFrame,
    cols: List[str] = [
        "elapsed_seconds",
        "elapsed_seconds_moving",
        "latitude",
        "longitude",
    ],
) -> Tuple:
    """
    Calculate the top 3 landings, boosts, and corners.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data for movement only

    Returns:
        List[Dict[str: float]]: List of top 3 landings, boosts, and corners
    """

    missing_cols = [col for col in cols if col not in df_moving.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")

    top_3_boosts = df_moving.nlargest(3, "az_e")[cols + ["az_e"]]
    top_3_landings = df_moving.nsmallest(3, "az_e")[cols + ["az_e"]]
    top_3_corners = df_moving.loc[df_moving["ah_e"].abs().nlargest(3).index][
        cols + ["ah_e"]
    ]

    top_3_boosts = [dict(top_3_boosts.iloc[i]) for i in range(top_3_boosts.shape[0])]
    top_3_landings = [
        dict(top_3_landings.iloc[i]) for i in range(top_3_landings.shape[0])
    ]
    top_3_corners = [dict(top_3_corners.iloc[i]) for i in range(top_3_corners.shape[0])]

    return top_3_boosts, top_3_landings, top_3_corners


def calculate_top_3_jumps(
    jump_events_df: pd.DataFrame,
    cols: List[str] = ["elapsed_seconds_ride", "latitude", "longitude"],
) -> Dict[str, float]:
    """
    Calculate the top 3 jumps based on airtime_s. May not capture longest distance jumps.

    Args:
        jump_events_df (pd.DataFrame): DataFrame containing jump events
        cols (List[str]): Columns to include in the output

    Returns:
        Dict[str, float]: Dictionary containing the top 3 jumps with selected columns
    """

    jump_cols = ["airtime_s", "distance_ft"]
    cols = cols + jump_cols

    missing_cols = [col for col in cols if col not in jump_events_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")

    top_3_jumps = jump_events_df.nlargest(3, "airtime_s")[cols]
    top_3_jumps = [dict(top_3_jumps.iloc[i]) for i in range(top_3_jumps.shape[0])]

    return top_3_jumps


def main(args: argparse.Namespace) -> None:

    start_time = time.perf_counter()

    summary_metrics = {}

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input file name and create output paths
    input_path = Path(args.input)
    file_name = input_path.stem
    output_path = output_dir / f"{file_name}.json"
    jumps_output_path = output_dir / f"{file_name}_jumps.parquet"

    df = pd.read_parquet(input_path)
    df_moving = generate_moving_df(df)

    # Process jumps
    logger.info("Extracting jump events...")
    jump_events_df = extract_jump_events(
        df_moving,
        jump_col="jump",
        time_col="elapsed_seconds",
        min_consecutive=min_consecutive,
    )

    if jump_events_df is not None:
        logger.info(f"Saving jump events to {jumps_output_path}")
        jump_events_df.to_parquet(jumps_output_path)
        longest_jump = jump_events_df["distance_ft"].max()
        total_jump_airtime = jump_events_df["airtime_s"].sum()
        total_jump_distance = jump_events_df["distance_ft"].sum()
        summary_metrics["longest_jump_ft"] = longest_jump
        summary_metrics["total_jump_airtime_s"] = total_jump_airtime
        summary_metrics["total_jump_distance_s"] = total_jump_distance
        top_3_jumps = calculate_top_3_jumps(jump_events_df)
        summary_metrics["top_3_jumps"] = top_3_jumps
    else:
        logger.warning("No jump events found")

    # Calculate all other summary metrics
    summary_metrics["duration_hr"] = calculate_duration_hr(df)
    summary_metrics["duration_moving_hr"] = calculate_duration_moving_hr(df)
    summary_metrics["distance_haversine"] = calculate_distance_haversine_mi(df)
    summary_metrics["elevation_gain"] = calculate_elevation_gain_ft(df)
    summary_metrics["avg_speed_mph"] = calculate_avg_speed_moving_mph(df_moving)

    avg_overall_intensity, avg_cornering_intensity = calculate_intensity_scores(
        df_moving
    )
    summary_metrics["avg_overall_intensity"] = avg_overall_intensity
    summary_metrics["avg_cornering_intensity"] = avg_cornering_intensity

    top_3_boosts, top_3_landings, top_3_corners = calculate_top_3_events(df_moving)
    summary_metrics["top_3_boosts"] = top_3_boosts
    summary_metrics["top_3_landings"] = top_3_landings
    summary_metrics["top_3_corners"] = top_3_corners

    logger.info(f"Saving summary metrics to {output_path}")
    with open(output_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

    return None


if __name__ == "__main__":

    try:
        args = parse_arguments()
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
