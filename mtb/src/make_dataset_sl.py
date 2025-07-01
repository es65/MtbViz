"""

Make dataset from Sensor Logger app csv files. Accepts directory or .zip file

Example usage:

# Process all files
python make_dataset_sl.py --many

# Process a specific file
python make_dataset_sl.py --input /path/to/file.zip

"""

import argparse
import glob
import os
import zipfile
import pandas as pd
import numpy as np
import time
from typing import Tuple
from mtb.src import config

########################    Config   ##########################################

g = config.g  # m/s^2
file_names = config.sl_file_names
cols_maps = config.sl_cols_maps
interp_cols = config.interp_cols

########################    Functions   ##########################################


def is_csv_file(file_name: str) -> bool:
    return file_name.lower().endswith(".csv")


def load_sl_csv(
    file,
    file_name: str,
    col_maps: dict[str, dict[str, str]],
    sep: str = ",",
    timezone: str = "America/Los_Angeles",
    verbose: bool = False,
) -> pd.DataFrame:
    """Loads data from a single CSV file from Sensor Logger app to a dataframe.
    - Sets index to datetime in Pacific Time by default.
    - Drops 'time' and 'seconds_elapsed' columns.
    - Renames columns according to col_maps.

    Parameters:
        file: str or file-like object
        file_name: str.

    Returns:
        pd.DataFrame: df with timeseries data from single file.
    """

    name = file_name.split("/")[-1].split(".")[0]

    df = pd.read_csv(file, sep=sep)
    if verbose:
        print(f"{name} df shape: {df.shape}")
    df["datetime_PT"] = pd.to_datetime(df["time"], unit="ns", utc=True).dt.tz_convert(
        timezone
    )
    df.set_index("datetime_PT", inplace=True)
    df.drop(["time", "seconds_elapsed"], axis=1, inplace=True)
    if name in col_maps:
        df.rename(columns=col_maps[name], inplace=True)

    return df


def bin_sensor_data(df, bin_width_ms=50):
    """
    Group sensor readings into time bins and aggregate.

    Must first interpolate any low frequency data before applying binning (eg GPS).

    Parameters:
        df (pd.DataFrame): DataFrame with DatetimeIndex
        bin_width_ms (int): Width of time bins in milliseconds

    Returns:
        pd.DataFrame: DataFrame with binned data
    """
    # Create time bins
    df["time_bin"] = df.index.floor(f"{bin_width_ms}ms")

    # Group by time bin and aggregate
    # Choose appropriate aggregation for each column
    agg_dict = {col: "mean" for col in df.select_dtypes(include=np.number).columns}

    # Aggregate categorical/object columns differently if needed
    # for col in df.select_dtypes(include=['object', 'category']).columns:
    #     agg_dict[col] = 'first'  # or 'last', 'mode', etc.

    binned_df = df.groupby("time_bin").agg(agg_dict)

    return binned_df


def process_single_ride(
    dir: str,
    file_names: list[str],
    bin_data: bool = True,
    downsample_freq: int = 5,
    dropna_resampled_df: bool = True,
    save_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calls load_sl_csv() to load all csv files from a directory or zip file.
    - Merges all dataframes on 'datetime_PT' index.
    - Interpolates missing values in interp_cols.
    - Converts uncalibrated accelerometer data from g to m/s^2.
    - Resamples to downsample_freq Hz.
    - All calculated columns generated in build_features.py!
    - Saves raw and resampled dataframes to parquet files.
    - Returns raw and resampled dataframes.

    Raises:
        ValueError: If provided path is neither a valid ZIP file nor a directory.

    Parameters:
        dir: str
            Directory or zip file containing Sensor Logger CSV files.
        file_names: list[str]
            List of file names to load from the directory or zip file.
        bin_sensor_data: bool
            If True, bins the sensor data into time bins.
        downsample_freq: int
            Frequency in Hz to downsample the data.
        dropna_resampled_df: bool
            If True, drops rows with NaN values in the resampled dataframe.
        save_path: str
            Path to save the raw and resampled dataframes as parquet files.

    Returns:
        df: pd.DataFrame
            Dataframe with all data from all csv files in directory or zip file.
        df_resampled: pd.DataFrame
            Resampled dataframe with downsample_freq Hz.
    """

    print(f"Processing ride: {os.path.basename(dir)}")
    start_time_sub = time.perf_counter()

    ride_name = os.path.splitext(dir)[-2].split("/")[-1]

    df = pd.DataFrame()

    if os.path.isdir(dir):

        if dir[-1] == "/":
            files = glob.glob(dir + "*.csv")
        else:
            files = glob.glob(dir + "/" + "*.csv")

        df = pd.DataFrame()

        for file in files:

            name = file.split("/")[-1].split(".")[0]

            if name in file_names:
                df_load = load_sl_csv(file, file_name=file, col_maps=cols_maps)

                if df.size == 0:
                    df = df_load
                else:
                    df = df.merge(df_load, how="outer", on="datetime_PT")

    elif zipfile.is_zipfile(dir):

        with zipfile.ZipFile(dir, "r") as z:
            for filename in z.namelist():
                if filename.endswith(".csv") and filename.split(".")[0] in file_names:
                    with z.open(filename) as f:
                        df_load = load_sl_csv(f, filename, cols_maps)
                        if df.size == 0:
                            df = df_load
                        else:
                            df = df.merge(df_load, how="outer", on="datetime_PT")

    else:
        raise ValueError("Provided path is neither a valid ZIP file nor a directory.")

    df.insert(0, "elapsed_seconds", (df.index - df.index[0]).total_seconds())

    for col in [col for col in interp_cols if col in df.columns]:
        df[col] = df[col].interpolate(method="time", axis=0)

    # uncalibrated accelerometer data are in g from sensor logger app - convert to m/s^2
    df["ax_uc_device"] = df["ax_uc_device"] * g
    df["ay_uc_device"] = df["ay_uc_device"] * g
    df["az_uc_device"] = df["az_uc_device"] * g

    freq_ms = int(1000 / downsample_freq)
    df_resampled = df.resample(f"{freq_ms}ms").mean()

    if bin_data:
        df = bin_sensor_data(df)

    if dropna_resampled_df:
        df_resampled.dropna(subset=["ax_device", "gx_device"], inplace=True)

    if save_path is not None:
        df.to_parquet(
            save_path.rstrip("/") + "/" + ride_name + "_" + "raw" + ".parquet"
        )
        df_resampled.to_parquet(
            save_path.rstrip("/")
            + "/"
            + ride_name
            + "_"
            + f"{downsample_freq}Hz"
            + ".parquet"
        )

    print(f"...Processed in {time.perf_counter() - start_time_sub:.2f} seconds.")
    print(f"...Shape of raw df: {df.shape}")
    print(f"...Shape of resampled df: {df_resampled.shape}")

    return df, df_resampled


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process Sensor Logger data files")
    parser.add_argument(
        "--many", action="store_true", help="Process all directories in data folder"
    )
    parser.add_argument(
        "--input", type=str, help="Path to a specific zip or directory to process"
    )
    parser.add_argument(
        "--freq", type=int, default=5, help="Downsample frequency in Hz"
    )
    parser.add_argument(
        "--save_path", type=str, help="Custom save path for output files"
    )
    return parser.parse_args()


def get_default_paths():
    """Get default paths for input and output files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(script_dir, "../data/raw/sl/")
    default_output_dir = os.path.join(script_dir, "../data/interim/")
    return default_input_dir, default_output_dir


def main():
    """Main function to process sensor logger data."""
    # Parse arguments or use defaults
    args = parse_arguments()
    input_dir, output_dir = get_default_paths()

    # Override defaults with any command line arguments
    process_many = args.many
    downsample_freq = args.freq
    save_path = args.save_path if args.save_path else output_dir

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    start_time = time.perf_counter()

    if process_many:
        # Process all rides in the input directory
        rides = glob.glob(os.path.join(input_dir, "*"))
        for ride in rides:
            process_single_ride(
                ride, file_names, downsample_freq=downsample_freq, save_path=save_path
            )
        print(f"All rides processed in {time.perf_counter() - start_time:.2f} seconds.")
    else:
        # Process a single ride
        if args.input:
            input_path = args.input
        else:
            raise ValueError("Please provide a valid input path using --input.")

        process_single_ride(
            input_path, file_names, downsample_freq=downsample_freq, save_path=save_path
        )


if __name__ == "__main__":

    main()
