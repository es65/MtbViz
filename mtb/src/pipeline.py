"""
Data Processing Pipeline

Raw CSVs -> processed parquet files and json summary metrics

This module provides:
- DataLoader: Load and preprocess raw Sensor Logger data
- FeatureBuilder: Transform data and build features
- MetricsCalculator: Calculate summary metrics and extract events
- Pipeline: Orchestrate the entire processing workflow
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from haversine import haversine, Unit

from mtb.src import config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:

    # Physical constants
    g: float = config.g
    mps_to_mph: float = config.mps_to_mph
    mile_to_km: float = config.mile_to_km
    mile_to_ft: float = config.mile_to_ft
    mile_to_m: float = config.mile_to_m
    ft_to_m: float = config.ft_to_m

    # Data loading settings
    timezone: str = config.DEFAULT_TIMEZONE
    downsample_freq: int = config.DEFAULT_DOWNSAMPLE_FREQ
    bin_data: bool = config.DEFAULT_BIN_DATA
    bin_width_ms: int = config.DEFAULT_BIN_WIDTH_MS
    dropna_resampled: bool = config.DEFAULT_DROPNA_RESAMPLED

    # Sensor Logger file settings
    sl_file_names: List[str] = field(default_factory=lambda: config.sl_file_names)
    sl_cols_maps: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: config.sl_cols_maps
    )
    interp_cols: List[str] = field(default_factory=lambda: config.interp_cols)

    # Feature building settings
    add_lead_lag: bool = config.DEFAULT_ADD_LEAD_LAG
    lead_periods: List[int] = field(default_factory=lambda: config.DEFAULT_LEAD_PERIODS)
    lag_periods: List[int] = field(default_factory=lambda: config.DEFAULT_LAG_PERIODS)
    magnitude_mapper: Dict[Tuple[str, str, str], str] = field(
        default_factory=lambda: config.magnitude_mapper
    )
    earth_to_travel_mapper: Dict[Tuple[str, str, str], Tuple[str, str, str]] = field(
        default_factory=lambda: config.earth_to_travel_mapper
    )
    lead_lag_cols: List[str] = field(default_factory=lambda: config.lead_lag_cols)

    # Jump detection settings
    jump_threshold: float = config.jump_z_accel_threshold
    jump_min_consecutive: int = config.jump_min_consecutive_points

    # Summary metrics settings
    distance_smooth_window: int = config.DEFAULT_DISTANCE_SMOOTH_WINDOW
    elevation_smooth_window: int = config.DEFAULT_ELEVATION_SMOOTH_WINDOW
    elevation_threshold: float = config.DEFAULT_ELEVATION_THRESHOLD
    speed_threshold: float = config.DEFAULT_SPEED_THRESHOLD

    # Output settings
    save_intermediate: bool = config.DEFAULT_SAVE_INTERMEDIATE
    verbose: bool = config.DEFAULT_VERBOSE

    # Default directories
    raw_dir: str = config.DEFAULT_RAW_DIR
    processed_dir: str = config.DEFAULT_PROCESSED_DIR
    intermediate_dir: str = config.DEFAULT_INTERMEDIATE_DIR


class DataValidationError(Exception):

    pass


class ProcessingError(Exception):

    pass


class BaseProcessor(ABC):
    """Base class for all data processors."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, *args, **kwargs):
        """Process the data. Must be implemented by subclasses....but currently not consistently lol."""
        pass

    def validate_input(self, df: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate that DataFrame has required columns."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

    def log_processing_time(self, start_time: float, operation: str) -> None:
        elapsed = time.perf_counter() - start_time
        self.logger.info(f"{operation} completed in {elapsed:.2f} seconds")


class DataLoader(BaseProcessor):
    """Load and preprocess raw Sensor Logger data."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.g = config.g
        self.file_names = config.sl_file_names
        self.cols_maps = config.sl_cols_maps
        self.interp_cols = config.interp_cols

    def load_sl_csv(
        self,
        file: Union[str, Any],
        file_name: str,
        sep: str = ",",
    ) -> pd.DataFrame:
        """Load data from a single CSV file from Sensor Logger app."""
        try:
            name = file_name.split("/")[-1].split(".")[0]

            df = pd.read_csv(file, sep=sep)
            if self.config.verbose:
                self.logger.info(f"{name} df shape: {df.shape}")

            # Convert time to datetime
            df["datetime_PT"] = pd.to_datetime(
                df["time"], unit="ns", utc=True
            ).dt.tz_convert(self.config.timezone)
            df.set_index("datetime_PT", inplace=True)

            # Drop unnecessary columns
            df.drop(["time", "seconds_elapsed"], axis=1, inplace=True)

            # Rename columns according to mapping
            if name in self.cols_maps:
                df.rename(columns=self.cols_maps[name], inplace=True)

            return df

        except Exception as e:
            raise ProcessingError(f"Error loading CSV file {file_name}: {str(e)}")

    def bin_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group sensor readings into time bins and aggregate."""
        try:
            # Create time bins
            df["time_bin"] = df.index.floor(f"{self.config.bin_width_ms}ms")

            # Group by time bin and aggregate
            agg_dict = {
                col: "mean" for col in df.select_dtypes(include=np.number).columns
            }

            binned_df = df.groupby("time_bin").agg(agg_dict)
            return binned_df

        except Exception as e:
            raise ProcessingError(f"Error binning sensor data: {str(e)}")

    def process_single_ride(
        self, input_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a single ride from directory or zip file."""
        start_time = time.perf_counter()
        input_path = Path(input_path)

        self.logger.info(f"Processing ride: {input_path.name}")

        df = pd.DataFrame()

        try:
            # Load data from directory or zip file
            if input_path.is_dir():
                df = self._load_from_directory(input_path)
            elif input_path.suffix.lower() == ".zip":
                df = self._load_from_zip(input_path)
            else:
                raise DataValidationError(
                    f"Input path must be a directory or zip file: {input_path}"
                )

            if df.empty:
                raise DataValidationError("No valid data found in input files")

            # Add elapsed seconds
            df.insert(0, "elapsed_seconds", (df.index - df.index[0]).total_seconds())

            # Interpolate missing values
            for col in [col for col in self.interp_cols if col in df.columns]:
                df[col] = df[col].interpolate(method="time", axis=0)

            # Convert uncalibrated accelerometer data from g to m/s^2
            accel_cols = ["ax_uc_device", "ay_uc_device", "az_uc_device"]
            for col in accel_cols:
                if col in df.columns:
                    df[col] = df[col] * self.g

            # Bin data if requested
            if self.config.bin_data:
                df = self.bin_sensor_data(df)

            # Resample to target frequency
            freq_ms = int(1000 / self.config.downsample_freq)
            df_resampled = df.resample(f"{freq_ms}ms").mean()

            # Drop NaN values if requested
            if self.config.dropna_resampled:
                df_resampled.dropna(subset=["ax_device", "gx_device"], inplace=True)

            self.log_processing_time(start_time, "Data loading")

            if self.config.verbose:
                self.logger.info(f"Raw df shape: {df.shape}")
                self.logger.info(f"Resampled df shape: {df_resampled.shape}")

            return df, df_resampled

        except Exception as e:
            raise ProcessingError(f"Error processing ride {input_path}: {str(e)}")

    def _load_from_directory(self, dir_path: Path) -> pd.DataFrame:
        """Load data from directory containing CSV files."""
        df = pd.DataFrame()

        for file_path in dir_path.glob("*.csv"):
            name = file_path.stem
            if name in self.file_names:
                df_load = self.load_sl_csv(file_path, str(file_path))

                if df.empty:
                    df = df_load
                else:
                    df = df.merge(df_load, how="outer", on="datetime_PT")

        return df

    def _load_from_zip(self, zip_path: Path) -> pd.DataFrame:
        """Load data from zip file containing CSV files."""
        import zipfile

        df = pd.DataFrame()

        with zipfile.ZipFile(zip_path, "r") as z:
            for filename in z.namelist():
                if filename.endswith(".csv") and Path(filename).stem in self.file_names:
                    with z.open(filename) as f:
                        df_load = self.load_sl_csv(f, filename)

                        if df.empty:
                            df = df_load
                        else:
                            df = df.merge(df_load, how="outer", on="datetime_PT")

        return df

    def process(self, input_path: Path) -> pd.DataFrame:
        """Required by BaseProcessor. Might remove or reworking naming."""
        raise NotImplementedError("Use process_single_ride method instead")


class FeatureBuilder(BaseProcessor):
    """Build features from preprocessed sensor data."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.magnitude_mapper = config.magnitude_mapper
        self.earth_to_travel_mapper = config.earth_to_travel_mapper
        self.lead_lag_cols = config.lead_lag_cols

    def add_magnitude_cols(
        self, df: pd.DataFrame, inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Add magnitude columns to the DataFrame."""
        try:
            if not inplace:
                df = df.copy()

            for cols, mag_col in self.magnitude_mapper.items():
                if all(col in df.columns for col in cols):
                    df[mag_col] = np.sqrt(
                        df[cols[0]] ** 2 + df[cols[1]] ** 2 + df[cols[2]] ** 2
                    )

            return df if not inplace else None

        except Exception as e:
            raise ProcessingError(f"Error adding magnitude columns: {str(e)}")

    def transform_to_world_frame(
        self, df: pd.DataFrame, inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """Transform accelerometer and gyroscope data to world coordinates."""
        try:
            # Define required columns ... should prob move to config
            quat_cols = ["qx", "qy", "qz", "qw"]
            acc_dev_cols = ["ax_device", "ay_device", "az_device"]
            acc_uc_dev_cols = ["ax_uc_device", "ay_uc_device", "az_uc_device"]
            gyr_dev_cols = ["gx_device", "gy_device", "gz_device"]

            required_cols = quat_cols + acc_dev_cols + acc_uc_dev_cols + gyr_dev_cols
            self.validate_input(df, required_cols)

            if not inplace:
                df = df.copy()

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

            # Add horizontal acceleration magnitudes
            df["ah_e"] = np.sqrt(df["ax_e"] ** 2 + df["ay_e"] ** 2)
            df["ah_uc_e"] = np.sqrt(df["ax_uc_e"] ** 2 + df["ay_uc_e"] ** 2)

            return df if not inplace else None

        except Exception as e:
            raise ProcessingError(f"Error transforming to world frame: {str(e)}")

    def identify_jumps(
        self,
        df: pd.DataFrame,
        accel_col: str = "az_uc_e",
        new_col: str = "jump",
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Identify potential jumps in motion data."""
        try:
            if accel_col not in df.columns:
                raise DataValidationError(
                    f"Acceleration column '{accel_col}' not found"
                )

            if not inplace:
                df = df.copy()

            # Create jump column (1 where acceleration is near zero = free fall)
            df[new_col] = (df[accel_col].abs() < self.config.jump_threshold).astype(int)

            # Log jump statistics
            jump_count = df[new_col].sum()
            total_points = len(df)
            jump_percentage = (jump_count / total_points) * 100
            self.logger.info(
                f"Identified {jump_count} potential jump points ({jump_percentage:.1f}% of data)"
            )

            return df if not inplace else None

        except Exception as e:
            raise ProcessingError(f"Error identifying jumps: {str(e)}")

    def add_lead_lag_features(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        dropna: bool = False,
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Add leading and lagging features to the DataFrame."""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise DataValidationError("DataFrame index must be a DatetimeIndex")

            if not inplace:
                df = df.copy()

            # Use specified columns or default
            if cols is None:
                cols = self.lead_lag_cols

            # Add leading features
            if self.config.lead_periods:
                self.logger.info(
                    f"Adding {len(self.config.lead_periods)} lead features..."
                )
                for period in self.config.lead_periods:
                    for col in cols:
                        if col in df.columns:
                            df[f"{col}_lead{period}"] = df[col].shift(-period)

            # Add lagging features
            if self.config.lag_periods:
                self.logger.info(
                    f"Adding {len(self.config.lag_periods)} lag features..."
                )
                for period in self.config.lag_periods:
                    for col in cols:
                        if col in df.columns:
                            df[f"{col}_lag{period}"] = df[col].shift(period)

            # Drop rows with NaN values if requested
            if dropna:
                original_shape = df.shape
                df = df.dropna()
                dropped_rows = original_shape[0] - df.shape[0]
                if dropped_rows > 0:
                    self.logger.warning(f"Dropped {dropped_rows} rows with NaN values")

            return df if not inplace else None

        except Exception as e:
            raise ProcessingError(f"Error adding lead/lag features: {str(e)}")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and build features."""
        start_time = time.perf_counter()

        try:
            # Validate required columns
            required_cols = [
                "ax_device",
                "ay_device",
                "az_device",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
            self.validate_input(df, required_cols)

            if df.empty:
                raise DataValidationError("Input DataFrame is empty")

            self.logger.info(f"Processing DataFrame with shape: {df.shape}")

            # Add magnitude columns
            self.logger.info("Adding magnitude columns...")
            self.add_magnitude_cols(df, inplace=True)

            # Transform to world frame
            self.logger.info("Transforming to world frame...")
            self.transform_to_world_frame(df, inplace=True)

            # Identify jumps
            self.logger.info("Identifying jumps...")
            self.identify_jumps(df, inplace=True)

            # Add lead/lag features if requested
            if self.config.add_lead_lag:
                self.logger.info("Adding lead/lag features...")
                self.add_lead_lag_features(df, inplace=True)

            self.log_processing_time(start_time, "Feature building")

            return df

        except Exception as e:
            raise ProcessingError(f"Error in feature building: {str(e)}")


class MetricsCalculator(BaseProcessor):
    """Calculate summary metrics and extract events."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)

    def generate_moving_df(
        self,
        df: pd.DataFrame,
        x_col: str = "elapsed_seconds",
        y_col: str = "speed",
    ) -> pd.DataFrame:
        """Generate DataFrame with only moving data."""
        try:
            df = df.copy()
            df["diff_secs"] = df[x_col].diff()
            df = df[df[y_col] > self.config.speed_threshold].copy()
            df["elapsed_seconds_moving"] = df["diff_secs"].fillna(0).cumsum()
            return df

        except Exception as e:
            raise ProcessingError(f"Error generating moving DataFrame: {str(e)}")

    def calculate_distance_haversine_mi(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        smooth: bool = True,
    ) -> float:
        """Calculate distance using Haversine formula."""
        try:
            df = df.copy()

            def haversine_series(lat1, lon1, lat2, lon2):
                return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

            if smooth:
                df[lat_col] = (
                    df[lat_col]
                    .rolling(self.config.distance_smooth_window, center=True)
                    .mean()
                )
                df[lon_col] = (
                    df[lon_col]
                    .rolling(self.config.distance_smooth_window, center=True)
                    .mean()
                )

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
            return (
                df["step_dist_haversine"].fillna(0).cumsum().iloc[-1] / config.mile_to_m
            )

        except Exception as e:
            raise ProcessingError(f"Error calculating distance: {str(e)}")

    def calculate_elevation_gain_ft(
        self,
        df: pd.DataFrame,
        y_col: str = "altitude",
    ) -> float:
        """Calculate total elevation gain."""
        try:
            df = df[[y_col]].copy()
            df[y_col] = (
                df[y_col]
                .rolling(window=self.config.elevation_smooth_window, center=True)
                .mean()
            )
            df["diff_alt"] = df[y_col].diff()
            return (
                df["diff_alt"]
                .where(df["diff_alt"] > self.config.elevation_threshold)
                .sum()
                / config.ft_to_m
            )

        except Exception as e:
            raise ProcessingError(f"Error calculating elevation gain: {str(e)}")

    def calculate_duration_hr(
        self, df: pd.DataFrame, x_col: str = "elapsed_seconds"
    ) -> float:
        """Calculate total duration in hours."""
        try:
            return round((df[x_col].max() - df[x_col].min()) / 3600, 3)
        except Exception as e:
            raise ProcessingError(f"Error calculating duration: {str(e)}")

    def calculate_duration_moving_hr(
        self,
        df: pd.DataFrame,
        x_col: str = "elapsed_seconds",
        y_col: str = "speed",
    ) -> float:
        """Calculate duration when moving."""
        try:
            df = df.copy()
            df["diff_secs"] = df[x_col].diff()
            moving_time = (
                df[df[y_col] > self.config.speed_threshold]["diff_secs"].fillna(0).sum()
            )
            return round(moving_time / 3600, 3)
        except Exception as e:
            raise ProcessingError(f"Error calculating moving duration: {str(e)}")

    def calculate_avg_speed_moving_mph(
        self,
        df_moving: pd.DataFrame,
        x_col: str = "elapsed_seconds",
        y_col: str = "speed",
    ) -> float:
        """Calculate average speed when moving."""
        try:
            df_moving = df_moving.copy()
            df_moving["diff_secs"] = df_moving[x_col].diff()
            df_moving["y_weighted"] = df_moving[y_col] * df_moving["diff_secs"]

            return (
                df_moving["y_weighted"].sum()
                / df_moving["diff_secs"].sum()
                * config.mps_to_mph
            )
        except Exception as e:
            raise ProcessingError(f"Error calculating average speed: {str(e)}")

    def calculate_intensity_scores(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate intensity scores."""
        try:
            cols = ["ar", "ah_e"]
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Required columns not found: {missing_cols}")

            avg_overall_intensity = df["ar"].mean()
            avg_cornering_intensity = np.abs(df["ah_e"]).mean()

            return avg_overall_intensity, avg_cornering_intensity

        except Exception as e:
            raise ProcessingError(f"Error calculating intensity scores: {str(e)}")

    def extract_jump_events(
        self,
        df: pd.DataFrame,
        jump_col: str = "jump",
        time_col: str = "elapsed_seconds",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        speed_col: str = "speed",
    ) -> Optional[pd.DataFrame]:
        """Extract jump events from the DataFrame."""
        try:
            # Check required columns exist
            required_cols = [jump_col, time_col]
            optional_cols = [lat_col, lon_col, speed_col]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"Required columns not found: {missing_cols}")

            # Check which optional columns exist
            available_cols = {col: col in df.columns for col in optional_cols}

            # Create a copy with only necessary columns
            df_work = df.copy()

            # Create a group identifier for consecutive jump points
            df_work["jump_change"] = df_work[jump_col].diff().ne(0).cumsum()

            # Group by this identifier to find consecutive sequences
            jump_groups = df_work[df_work[jump_col] == 1].groupby("jump_change")

            # Initialize a list to store jump event data
            jump_events = []

            # Process each jump group
            for _, group in jump_groups:
                # Skip if fewer than min_consecutive points
                if len(group) < self.config.jump_min_consecutive:
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

                # Add location data if available
                if available_cols[lat_col] and available_cols[lon_col]:
                    event["latitude"] = group[lat_col].iloc[0]
                    event["longitude"] = group[lon_col].iloc[0]

                # Use DataFrame's DatetimeIndex for timestamps if available
                if isinstance(df.index, pd.DatetimeIndex):
                    event["datetime"] = group.index[0]

                # Add event to the list
                jump_events.append(event)

            if not jump_events:
                self.logger.warning("No jump events found.")
                return None

            # Create the jumps DataFrame
            jumps_df = pd.DataFrame(jump_events)
            jumps_df = jumps_df.sort_values("elapsed_seconds_ride").reset_index(
                drop=True
            )

            # Log jump statistics
            self.logger.info(f"Extracted {len(jumps_df)} jump events")
            if "airtime_s" in jumps_df.columns:
                self.logger.info(
                    f"Average airtime: {jumps_df['airtime_s'].mean():.2f}s"
                )
                self.logger.info(f"Max airtime: {jumps_df['airtime_s'].max():.2f}s")

            return jumps_df

        except Exception as e:
            raise ProcessingError(f"Error extracting jump events: {str(e)}")

    def calculate_top_3_events(
        self,
        df_moving: pd.DataFrame,
        cols: List[str] = [
            "elapsed_seconds",
            "elapsed_seconds_moving",
            "latitude",
            "longitude",
        ],
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Calculate top 3 landings, boosts, and corners."""
        try:
            missing_cols = [col for col in cols if col not in df_moving.columns]
            if missing_cols:
                raise DataValidationError(f"Required columns not found: {missing_cols}")

            top_3_boosts = df_moving.nlargest(3, "az_e")[cols + ["az_e"]]
            top_3_landings = df_moving.nsmallest(3, "az_e")[cols + ["az_e"]]
            top_3_corners = df_moving.loc[df_moving["ah_e"].abs().nlargest(3).index][
                cols + ["ah_e"]
            ]

            top_3_boosts = [
                dict(top_3_boosts.iloc[i]) for i in range(top_3_boosts.shape[0])
            ]
            top_3_landings = [
                dict(top_3_landings.iloc[i]) for i in range(top_3_landings.shape[0])
            ]
            top_3_corners = [
                dict(top_3_corners.iloc[i]) for i in range(top_3_corners.shape[0])
            ]

            return top_3_boosts, top_3_landings, top_3_corners

        except Exception as e:
            raise ProcessingError(f"Error calculating top 3 events: {str(e)}")

    def calculate_top_3_jumps(
        self,
        jump_events_df: pd.DataFrame,
        cols: List[str] = ["elapsed_seconds_ride", "latitude", "longitude"],
    ) -> List[Dict]:
        """Calculate top 3 jumps based on airtime."""
        try:
            jump_cols = ["airtime_s", "distance_ft"]
            cols = cols + jump_cols

            missing_cols = [col for col in cols if col not in jump_events_df.columns]
            if missing_cols:
                raise DataValidationError(f"Required columns not found: {missing_cols}")

            top_3_jumps = jump_events_df.nlargest(3, "airtime_s")[cols]
            top_3_jumps = [
                dict(top_3_jumps.iloc[i]) for i in range(top_3_jumps.shape[0])
            ]

            return top_3_jumps

        except Exception as e:
            raise ProcessingError(f"Error calculating top 3 jumps: {str(e)}")

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process DataFrame and calculate all metrics."""
        start_time = time.perf_counter()

        try:
            summary_metrics = {}

            # Generate moving DataFrame
            df_moving = self.generate_moving_df(df)

            # Extract jump events
            self.logger.info("Extracting jump events...")
            jump_events_df = self.extract_jump_events(df_moving)

            if jump_events_df is not None:
                longest_jump = jump_events_df["distance_ft"].max()
                total_jump_airtime = jump_events_df["airtime_s"].sum()
                total_jump_distance = jump_events_df["distance_ft"].sum()
                summary_metrics["longest_jump_ft"] = longest_jump
                summary_metrics["total_jump_airtime_s"] = total_jump_airtime
                summary_metrics["total_jump_distance_s"] = total_jump_distance
                top_3_jumps = self.calculate_top_3_jumps(jump_events_df)
                summary_metrics["top_3_jumps"] = top_3_jumps
            else:
                self.logger.warning("No jump events found")

            # Calculate all other summary metrics
            summary_metrics["duration_hr"] = self.calculate_duration_hr(df)
            summary_metrics["duration_moving_hr"] = self.calculate_duration_moving_hr(
                df
            )
            summary_metrics["distance_haversine"] = (
                self.calculate_distance_haversine_mi(df)
            )
            summary_metrics["elevation_gain"] = self.calculate_elevation_gain_ft(df)
            summary_metrics["avg_speed_mph"] = self.calculate_avg_speed_moving_mph(
                df_moving
            )

            avg_overall_intensity, avg_cornering_intensity = (
                self.calculate_intensity_scores(df_moving)
            )
            summary_metrics["avg_overall_intensity"] = avg_overall_intensity
            summary_metrics["avg_cornering_intensity"] = avg_cornering_intensity

            top_3_boosts, top_3_landings, top_3_corners = self.calculate_top_3_events(
                df_moving
            )
            summary_metrics["top_3_boosts"] = top_3_boosts
            summary_metrics["top_3_landings"] = top_3_landings
            summary_metrics["top_3_corners"] = top_3_corners

            self.log_processing_time(start_time, "Metrics calculation")

            return {
                "summary_metrics": summary_metrics,
                "jump_events_df": jump_events_df,
            }

        except Exception as e:
            raise ProcessingError(f"Error in metrics calculation: {str(e)}")


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize processors
        self.data_loader = DataLoader(config)
        self.feature_builder = FeatureBuilder(config)
        self.metrics_calculator = MetricsCalculator(config)

    def process_ride(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Process a single ride through the entire pipeline."""
        start_time = time.perf_counter()
        input_path = Path(input_path)

        if output_dir is None:
            output_dir = Path(self.config.processed_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if output files already exist
        processed_output_path = output_dir / f"{input_path.stem}.parquet"
        summary_output_path = output_dir / f"{input_path.stem}.json"

        if processed_output_path.exists() and not overwrite:
            self.logger.info(f"Output file already exists: {processed_output_path}")
            self.logger.info(
                "Skipping processing. Use --overwrite to force reprocessing."
            )
            return {
                "raw_df": None,
                "resampled_df": None,
                "processed_df": None,
                "summary_metrics": None,
                "jump_events_df": None,
                "output_files": {
                    "raw": None,
                    "resampled": None,
                    "processed": processed_output_path,
                    "summary": summary_output_path,
                    "jumps": (
                        output_dir / f"{input_path.stem}_jumps.parquet"
                        if (output_dir / f"{input_path.stem}_jumps.parquet").exists()
                        else None
                    ),
                },
                "skipped": True,
            }

        try:
            self.logger.info(f"Starting pipeline for: {input_path.name}")

            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data...")
            raw_df, resampled_df = self.data_loader.process_single_ride(input_path)

            # Save intermediate results if requested
            if self.config.save_intermediate:
                raw_output_path = output_dir / f"{input_path.stem}_raw.parquet"
                resampled_output_path = (
                    output_dir
                    / f"{input_path.stem}_{self.config.downsample_freq}Hz.parquet"
                )

                raw_df.to_parquet(raw_output_path)
                resampled_df.to_parquet(resampled_output_path)
                self.logger.info(f"Saved intermediate files to {output_dir}")

            # Step 2: Build features
            self.logger.info("Step 2: Building features...")
            processed_df = self.feature_builder.process(resampled_df)

            # Save processed data
            processed_df.to_parquet(processed_output_path)

            # Step 3: Calculate metrics
            self.logger.info("Step 3: Calculating metrics...")
            metrics_results = self.metrics_calculator.process(processed_df)

            # Save results
            with open(summary_output_path, "w") as f:
                import json

                json.dump(metrics_results["summary_metrics"], f, indent=4)

            if metrics_results["jump_events_df"] is not None:
                jumps_output_path = output_dir / f"{input_path.stem}_jumps.parquet"
                metrics_results["jump_events_df"].to_parquet(jumps_output_path)

            # Log completion
            elapsed_time = time.perf_counter() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

            return {
                "raw_df": raw_df,
                "resampled_df": resampled_df,
                "processed_df": processed_df,
                "summary_metrics": metrics_results["summary_metrics"],
                "jump_events_df": metrics_results["jump_events_df"],
                "output_files": {
                    "raw": raw_output_path if self.config.save_intermediate else None,
                    "resampled": (
                        resampled_output_path if self.config.save_intermediate else None
                    ),
                    "processed": processed_output_path,
                    "summary": summary_output_path,
                    "jumps": (
                        jumps_output_path
                        if metrics_results["jump_events_df"] is not None
                        else None
                    ),
                },
                "skipped": False,
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def process_multiple_rides(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process multiple rides from a directory."""
        input_dir = Path(input_dir)

        if output_dir is None:
            output_dir = Path(self.config.processed_dir)
        else:
            output_dir = Path(output_dir)

        results = []

        # Find all zip files and directories
        input_paths = list(input_dir.glob("*.zip")) + [
            p for p in input_dir.iterdir() if p.is_dir()
        ]

        if not input_paths:
            self.logger.warning(f"No valid input files found in {input_dir}")
            return results

        self.logger.info(f"Found {len(input_paths)} rides to process")

        for input_path in input_paths:
            try:
                result = self.process_ride(input_path, output_dir, overwrite)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {input_path}: {str(e)}")
                continue

        # Count successful and skipped rides
        successful = sum(1 for r in results if not r.get("skipped", False))
        skipped = sum(1 for r in results if r.get("skipped", False))

        self.logger.info(
            f"Successfully processed {successful} rides, skipped {skipped} rides (already exist)"
        )
        return results
