"""
Process sensor data from interim.

- Coordinate system transformations
- Lead/lag and other ML features
- Jump detection and event extraction

Example usage:

Basic:
python build_features.py --input path/to/input.parquet

With custom parameters:
python build_features.py --input path/to/input.parquet \
    --output-dir path/to/output \
    --verbose

"""

import io
import argparse
import logging
from typing import Dict, List, Tuple, Union, Optional
import time
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

from mtb.src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
magnitude_mapper = config.magnitude_mapper
earth_to_travel_mapper = config.earth_to_travel_mapper
lead_lag_cols = config.lead_lag_cols
threshold = config.jump_z_accel_threshold


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and build features from sensor data"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/processed",
        help="Directory to save processed files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    return args


def add_magnitude_cols(
    df: pd.DataFrame, col_map: dict[str, str], inplace: bool = True
) -> pd.DataFrame:
    """
    Add magnitude columns to the DataFrame based on the provided column mapping.

    Parameters:
        df (pd.DataFrame): Input DataFrame with sensor data.
        col_map (dict): Mapping of sensor columns to their respective magnitude names.
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added magnitude columns.
    """
    if not inplace:
        df = df.copy()

    for cols, mag_col in col_map.items():
        df[mag_col] = np.sqrt(df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2)

    if not inplace:
        return df


def transform_to_world_frame(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Transform accelerometer and gyroscope data from device coordinates to world coordinates using quaternions.

    Note: Sensor Logger uses the x-north-z-vertical reference frame.

    Reference Frame Suffixes:
    - device: iPhone (x=up, y=right, z=normal to screen)
    - e: earth (x=north, y=east, z=up)

    Parameters:
        df (pd.DataFrame): Input DataFrame with sensor data.
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with transformed accelerometer and gyroscope data in world coordinates.

    Raises:
        ValueError: If any of the required columns are missing in the DataFrame.
    """
    # Define required columns
    quat_cols = ["qx", "qy", "qz", "qw"]
    acc_dev_cols = ["ax_device", "ay_device", "az_device"]
    acc_uc_dev_cols = ["ax_uc_device", "ay_uc_device", "az_uc_device"]
    gyr_dev_cols = ["gx_device", "gy_device", "gz_device"]

    # Check for required columns
    required_cols = quat_cols + acc_dev_cols + acc_uc_dev_cols + gyr_dev_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not inplace:
        df = df.copy()

    try:
        # Extract quaternion components
        quaternions = df[quat_cols].values

        # Extract sensor data in device coordinates
        acc_device = df[acc_dev_cols].values
        acc_uc_device = df[acc_uc_dev_cols].values
        gyr_device = df[gyr_dev_cols].values

        # Create Rotation objects from quaternions
        rotations = R.from_quat(quaternions)

        # Rotate sensor data to the world frame
        acc_e = rotations.apply(acc_device)
        acc_uc_e = rotations.apply(acc_uc_device)
        gyr_e = rotations.apply(gyr_device)

        # Add transformed data to DataFrame
        df[["ax_e", "ay_e", "az_e"]] = acc_e
        df[["ax_uc_e", "ay_uc_e", "az_uc_e"]] = acc_uc_e
        df[["gx_e", "gy_e", "gz_e"]] = gyr_e

        df["ah_e"] = np.sqrt(df["ax_e"] ** 2 + df["ay_e"] ** 2)
        df["ah_uc_e"] = np.sqrt(df["ax_uc_e"] ** 2 + df["ay_uc_e"] ** 2)

        return df if not inplace else None

    except Exception as e:
        raise RuntimeError(f"Error during coordinate transformation: {str(e)}")


def transform_to_travel_frame(
    df: pd.DataFrame,
    bearing_col: str = "bearing",
    col_mappings: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    include_z: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    NOTE: DON'T USE! BASED ON HIGHLY INACCURATE BEARING DATA FROM GPS DATA.

    Transform data from earth-relative frame to direction-of-travel-relative frame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to transform.
        bearing_col (str): Column name containing bearing in degrees (0-360°, where 0° is North).
        col_mappings (dict): Dictionary mapping input column names to output column names.
                           Format: {('input_x', 'input_y'[, 'input_z']): ('output_x', 'output_y'[, 'output_z'])}
                           If None, defaults to {('ax_e', 'ay_e'): ('ax_t', 'ay_t')}
        include_z (bool): Whether to include z-axis transformations.
                         If True, col_mappings must include z components.
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with transformed data columns added.

    Raises:
        ValueError: If required columns are missing or invalid column mappings are provided.
    """
    if not inplace:
        df = df.copy()

    # Check if bearing column exists
    if bearing_col not in df.columns:
        raise ValueError(f"Bearing column '{bearing_col}' not found in DataFrame")

    # Get bearing values and check for NaN values
    bearing_degs = df[bearing_col].values
    nan_mask = np.isnan(bearing_degs)
    if np.any(nan_mask):
        logger.warning(
            f"{np.sum(nan_mask)} NaN values found in bearing column. "
            f"Transformations for these rows will result in NaN."
        )

    # Set default column mappings if not provided
    if col_mappings is None:
        if include_z:
            col_mappings = {("ax_e", "ay_e", "az_e"): ("ax_t", "ay_t", "az_t")}
        else:
            col_mappings = {("ax_e", "ay_e"): ("ax_t", "ay_t")}

    try:
        # Convert bearing to radians and calculate sine and cosine values
        theta = np.deg2rad(bearing_degs)
        cos_theta = np.cos(-theta)  # Negative angle to align x with forward direction
        sin_theta = np.sin(-theta)

        # Process each set of columns
        for input_cols, output_cols in col_mappings.items():
            # Validate input columns
            if include_z and len(input_cols) != 3:
                raise ValueError(
                    f"When include_z=True, input columns must contain 3 elements: {input_cols}"
                )
            if not include_z and len(input_cols) < 2:
                raise ValueError(
                    f"Input columns must contain at least 2 elements: {input_cols}"
                )

            # Validate output columns
            if include_z and len(output_cols) != 3:
                raise ValueError(
                    f"When include_z=True, output columns must contain 3 elements: {output_cols}"
                )
            if not include_z and len(output_cols) < 2:
                raise ValueError(
                    f"Output columns must contain at least 2 elements: {output_cols}"
                )

            # Check if input columns exist
            missing_cols = [col for col in input_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Input columns not found in DataFrame: {missing_cols}"
                )

            # Extract input data
            x_values = df[input_cols[0]].values
            y_values = df[input_cols[1]].values

            # Transform x and y values using rotation matrix
            # [x_t]   = [cos(-θ) -sin(-θ)] [x_e]
            # [y_t]     [sin(-θ)  cos(-θ)] [y_e]
            x_transformed = cos_theta * x_values - sin_theta * y_values
            y_transformed = sin_theta * x_values + cos_theta * y_values

            # Assign transformed values to output columns
            df[output_cols[0]] = x_transformed
            df[output_cols[1]] = y_transformed

            # Handle z values if include_z is True
            if include_z and len(input_cols) >= 3 and len(output_cols) >= 3:
                z_values = df[input_cols[2]].values
                # Z-axis is not affected by bearing rotation
                df[output_cols[2]] = z_values

        return df if not inplace else None

    except Exception as e:
        raise RuntimeError(f"Error during travel frame transformation: {str(e)}")


def freq_str_to_seconds(freq_str: str) -> float:
    """
    Convert a Pandas frequency string to total seconds as a float.

    Parameters:
        freq_str (str): Frequency string (e.g., '200ms', '5T', '1D')

    Returns:
        float: Total seconds represented by the frequency
    """
    try:
        td = pd.to_timedelta(freq_str)
        return td.total_seconds()
    except ValueError as e:
        logger.error(f"Error converting frequency string: {e}")
        return None


def add_lead_lag_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    lead_periods: Optional[Union[List[int], int]] = None,
    lag_periods: Optional[Union[List[int], int]] = None,
    lead_time: Optional[Union[float, List[float]]] = None,
    lag_time: Optional[Union[float, List[float]]] = None,
    dropna: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add leading and lagging features to the DataFrame with DatetimeIndex.

    Parameters:
        df (pd.DataFrame): Input DataFrame with time series data (must have DatetimeIndex).
        cols (List[str], optional): List of column names to create lead/lag features for.
                                   If None, use all numeric columns.
        lead_periods (int or List[int], optional): Number of leading periods to shift.
                                                 If List, will create multiple lead columns.
        lag_periods (int or List[int], optional): Number of lagging periods to shift.
                                                If List, will create multiple lag columns.
        lead_time (float or List[float], optional): Time in seconds for leading features.
                                                   Will create features for all periods within this time.
        lag_time (float or List[float], optional): Time in seconds for lagging features.
                                                  Will create features for all periods within this time.
        dropna (bool): Whether to drop rows with NaN values after adding features.
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added lead/lag features.

    Raises:
        TypeError: If DataFrame index is not a DatetimeIndex
        ValueError: If sampling frequency cannot be determined and time-based features are requested
        ValueError: If invalid periods or times are provided
    """
    # Validate DataFrame index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    # Validate periods and times
    if lead_periods is not None:
        if isinstance(lead_periods, (list, tuple)):
            if not all(isinstance(p, int) and p > 0 for p in lead_periods):
                raise ValueError("All lead periods must be positive integers")
        elif not isinstance(lead_periods, int) or lead_periods <= 0:
            raise ValueError(
                "Lead periods must be a positive integer or list of positive integers"
            )

    if lag_periods is not None:
        if isinstance(lag_periods, (list, tuple)):
            if not all(isinstance(p, int) and p > 0 for p in lag_periods):
                raise ValueError("All lag periods must be positive integers")
        elif not isinstance(lag_periods, int) or lag_periods <= 0:
            raise ValueError(
                "Lag periods must be a positive integer or list of positive integers"
            )

    if lead_time is not None:
        if isinstance(lead_time, (list, tuple)):
            if not all(isinstance(t, (int, float)) and t > 0 for t in lead_time):
                raise ValueError("All lead times must be positive numbers")
        elif not isinstance(lead_time, (int, float)) or lead_time <= 0:
            raise ValueError(
                "Lead time must be a positive number or list of positive numbers"
            )

    if lag_time is not None:
        if isinstance(lag_time, (list, tuple)):
            if not all(isinstance(t, (int, float)) and t > 0 for t in lag_time):
                raise ValueError("All lag times must be positive numbers")
        elif not isinstance(lag_time, (int, float)) or lag_time <= 0:
            raise ValueError(
                "Lag time must be a positive number or list of positive numbers"
            )

    if not inplace:
        df = df.copy()

    try:
        # If no columns specified, use all numeric columns
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()

        # Calculate the time difference between consecutive samples
        time_diff = pd.Series(df.index).diff().median()

        # Estimate the sampling frequency (samples per second)
        if pd.notna(time_diff) and time_diff.total_seconds() > 0:
            sampling_freq = 1.0 / time_diff.total_seconds()
        else:
            sampling_freq = None
            if (
                lead_time is not None or lag_time is not None
            ) and sampling_freq is None:
                raise ValueError(
                    "Cannot determine sampling frequency from index. "
                    "Please use lead_periods/lag_periods instead of lead_time/lag_time."
                )

        # Convert time-based parameters to period-based
        if lead_time is not None:
            if isinstance(lead_time, (int, float)):
                # Calculate how many periods correspond to the specified time
                periods = int(round(lead_time * sampling_freq))
                lead_periods = list(range(1, periods + 1))
            elif isinstance(lead_time, list):
                # Process each time value
                lead_periods = []
                for t in lead_time:
                    periods = int(round(t * sampling_freq))
                    lead_periods.extend(list(range(1, periods + 1)))
                # Remove duplicates and sort
                lead_periods = sorted(list(set(lead_periods)))

        if lag_time is not None:
            if isinstance(lag_time, (int, float)):
                # Calculate how many periods correspond to the specified time
                periods = int(round(lag_time * sampling_freq))
                lag_periods = list(range(1, periods + 1))
            elif isinstance(lag_time, list):
                # Process each time value
                lag_periods = []
                for t in lag_time:
                    periods = int(round(t * sampling_freq))
                    lag_periods.extend(list(range(1, periods + 1)))
                # Remove duplicates and sort
                lag_periods = sorted(list(set(lag_periods)))

        # Process leading features
        if lead_periods is not None:
            if isinstance(lead_periods, int):
                lead_periods = [lead_periods]

            logger.info(
                f"Adding {len(lead_periods)} lead features for {len(cols)} columns..."
            )
            for period in lead_periods:
                for col in cols:
                    df[f"{col}_lead{period}"] = df[col].shift(-period)

        # Process lagging features
        if lag_periods is not None:
            if isinstance(lag_periods, int):
                lag_periods = [lag_periods]

            logger.info(
                f"Adding {len(lag_periods)} lag features for {len(cols)} columns..."
            )
            for period in lag_periods:
                for col in cols:
                    df[f"{col}_lag{period}"] = df[col].shift(period)

        # Drop rows with NaN values if requested
        if dropna:
            original_shape = df.shape
            df = df.dropna()
            dropped_rows = original_shape[0] - df.shape[0]
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN values")

        return df if not inplace else None

    except Exception as e:
        raise RuntimeError(f"Error during lead/lag feature generation: {str(e)}")


def identify_jumps(
    df: pd.DataFrame,
    accel_col: str = "az_uc_e",
    threshold: float = 1.5,
    new_col: str = "jump",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Identify potential jumps in motion data by detecting when vertical
    acceleration is close to zero (free fall).

    Parameters:
        df (pd.DataFrame): Input DataFrame containing acceleration data
        accel_col (str): Column name for vertical acceleration (typically 'az_e')
        threshold (float): Threshold value around zero to identify as a jump
        new_col (str): Name of the new column to create
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the new 'jump' column (1 for jump, 0 for no jump)

    Raises:
        ValueError: If acceleration column is not found in DataFrame or threshold is invalid
    """
    # Validate inputs
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    if accel_col not in df.columns:
        raise ValueError(f"Acceleration column '{accel_col}' not found in DataFrame")

    # Create a copy if not modifying in place
    if not inplace:
        df = df.copy()

    try:
        # Create the jump column
        # 1 where absolute value of acceleration is below threshold (near-zero = free fall)
        # 0 elsewhere
        df[new_col] = (df[accel_col].abs() < threshold).astype(int)

        # Log jump statistics
        jump_count = df[new_col].sum()
        total_points = len(df)
        jump_percentage = (jump_count / total_points) * 100
        logger.info(
            f"Identified {jump_count} potential jump points ({jump_percentage:.1f}% of data)"
        )

        return df if not inplace else None

    except Exception as e:
        raise RuntimeError(f"Error during jump identification: {str(e)}")


def main(args: argparse.Namespace) -> None:
    """
    Main function to process sensor data and build features.

    Parameters:
        args (argparse.Namespace): Command line arguments
    """
    start_time = time.perf_counter()

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(args.input)
        file_name = input_path.stem
        output_path = output_dir / f"{file_name}.parquet"

        logger.info(f"Loading data from {input_path}")
        try:
            df = pd.read_parquet(input_path)
        except Exception as e:
            logger.error(f"Error loading parquet file: {e}")
            raise

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        required_cols = ["ax_device", "ay_device", "az_device", "qx", "qy", "qz", "qw"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.empty:
            raise ValueError("No valid data remaining after dropping NaN values")

        logger.info(f"DataFrame shape after dropping NaN: {df.shape}")

        # Process data
        logger.info("Adding magnitude columns...")
        add_magnitude_cols(df, magnitude_mapper, inplace=True)

        logger.info("Transforming to world frame...")
        transform_to_world_frame(df, inplace=True)

        logger.info("Identifying jumps...")
        identify_jumps(df, threshold=threshold, inplace=True)

        logger.info("Adding lead/lag features...")
        add_lead_lag_features(
            df,
            cols=lead_lag_cols,
            lead_periods=[1, 2, 3],
            lag_periods=[1, 2, 3],
            dropna=False,
            inplace=True,
        )

        logger.info(f"Saving processed data to {output_path}")
        df.to_parquet(output_path)

        if args.verbose:
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame index type: {type(df.index)}")
            logger.info(f"DataFrame columns:\n{df.columns}")

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        args = parse_arguments()
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
