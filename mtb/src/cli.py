#!/usr/bin/env python3
"""
Command-line interface for MtbViz data processing pipeline.

"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from mtb.src.pipeline import (
    Pipeline,
    ProcessingConfig,
    DataValidationError,
    ProcessingError,
)
from mtb.src import config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("mtbviz_pipeline.log"),
        ],
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Mountain Bike Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Process a single ride
  process --input {config.DEFAULT_RAW_DIR}/ride.zip --output {config.DEFAULT_PROCESSED_DIR}
  
  # Process all rides in a directory
  process --input {config.DEFAULT_RAW_DIR} --output {config.DEFAULT_PROCESSED_DIR} --batch
  
  # Process with custom settings
  process --input ride.zip --downsample-freq 10 --no-lead-lag --verbose
  
  # Process with custom jump detection
  process --input ride.zip --jump-threshold 2.0 --jump-min-consecutive 3
  
  # Force reprocessing of existing files
  process --input ride.zip --overwrite
  
  # Batch process with overwrite
  process --input {config.DEFAULT_RAW_DIR} --batch --overwrite
        """,
    )

    # Input/output arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input path: zip file, directory, or folder containing rides",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=config.DEFAULT_PROCESSED_DIR,
        help=f"Output directory for processed files (default: {config.DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all rides in input directory (if input is a directory)",
    )

    # Data loading settings
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/Los_Angeles",
        help="Timezone for datetime conversion (default: America/Los_Angeles)",
    )
    parser.add_argument(
        "--downsample-freq",
        type=int,
        default=5,
        help="Downsample frequency in Hz (default: 5)",
    )
    parser.add_argument(
        "--no-bin-data", action="store_true", help="Disable sensor data binning"
    )
    parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=50,
        help="Bin width in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--no-dropna",
        action="store_true",
        help="Do not drop NaN values from resampled data",
    )

    # Feature building settings
    parser.add_argument(
        "--no-lead-lag", action="store_true", help="Disable lead/lag feature generation"
    )
    parser.add_argument(
        "--lead-periods",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Lead periods for feature generation (default: 1 2 3)",
    )
    parser.add_argument(
        "--lag-periods",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Lag periods for feature generation (default: 1 2 3)",
    )

    # Jump detection settings
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=None,
        help="Jump detection threshold (default: from config)",
    )
    parser.add_argument(
        "--jump-min-consecutive",
        type=int,
        default=None,
        help="Minimum consecutive points for jump detection (default: from config)",
    )

    # Summary metrics settings
    parser.add_argument(
        "--distance-smooth-window",
        type=int,
        default=21,
        help="Distance smoothing window size (default: 21)",
    )
    parser.add_argument(
        "--elevation-smooth-window",
        type=int,
        default=25,
        help="Elevation smoothing window size (default: 25)",
    )
    parser.add_argument(
        "--elevation-threshold",
        type=float,
        default=0.0,
        help="Minimum elevation change to count as gain (default: 0.0)",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.0,
        help="Speed threshold for moving calculations (default: 0.0)",
    )

    # Output settings
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Do not save intermediate processing files",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite existing output files if they already exist",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser


def create_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """Create ProcessingConfig from command line arguments."""
    config = ProcessingConfig()

    # Data loading settings
    config.timezone = args.timezone
    config.downsample_freq = args.downsample_freq
    config.bin_data = not args.no_bin_data
    config.bin_width_ms = args.bin_width_ms
    config.dropna_resampled = not args.no_dropna

    # Feature building settings
    config.add_lead_lag = not args.no_lead_lag
    config.lead_periods = args.lead_periods
    config.lag_periods = args.lag_periods

    # Jump detection settings
    if args.jump_threshold is not None:
        config.jump_threshold = args.jump_threshold
    if args.jump_min_consecutive is not None:
        config.jump_min_consecutive = args.jump_min_consecutive

    # Summary metrics settings
    config.distance_smooth_window = args.distance_smooth_window
    config.elevation_smooth_window = args.elevation_smooth_window
    config.elevation_threshold = args.elevation_threshold
    config.speed_threshold = args.speed_threshold

    # Output settings
    config.save_intermediate = not args.no_save_intermediate
    config.verbose = args.verbose

    return config


def validate_input_path(input_path: str) -> Path:
    """Validate and return input path."""
    path = Path(input_path)

    if not path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if path.is_file() and path.suffix.lower() != ".zip":
        raise ValueError(f"Input file must be a zip file: {input_path}")

    return path


def process_single_ride(args: argparse.Namespace) -> None:
    """Process a single ride."""
    try:
        # Validate input
        input_path = validate_input_path(args.input)

        # Create config
        config = create_config_from_args(args)

        # Setup logging
        setup_logging(config.verbose)
        logger = logging.getLogger(__name__)

        # Create pipeline
        pipeline = Pipeline(config)

        # Process ride
        logger.info(f"Processing single ride: {input_path}")
        result = pipeline.process_ride(input_path, args.output, args.overwrite)

        if result.get("skipped", False):
            logger.info("Ride was skipped (output already exists)")
        else:
            logger.info("Processing completed successfully!")
            logger.info(f"Output files:")
            for file_type, file_path in result["output_files"].items():
                if file_path:
                    logger.info(f"  {file_type}: {file_path}")

    except (ValueError, DataValidationError, ProcessingError) as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


def process_multiple_rides(args: argparse.Namespace) -> None:
    """Process multiple rides."""
    try:
        # Validate input
        input_path = validate_input_path(args.input)

        if not input_path.is_dir():
            raise ValueError("Input must be a directory for batch processing")

        # Create config
        config = create_config_from_args(args)

        # Setup logging
        setup_logging(config.verbose)
        logger = logging.getLogger(__name__)

        # Create pipeline
        pipeline = Pipeline(config)

        # Process rides
        logger.info(f"Processing multiple rides from: {input_path}")
        results = pipeline.process_multiple_rides(
            input_path, args.output, args.overwrite
        )

        # Count successful and skipped rides
        successful = sum(1 for r in results if not r.get("skipped", False))
        skipped = sum(1 for r in results if r.get("skipped", False))

        logger.info(
            f"Batch processing completed! Processed {successful} rides, skipped {skipped} rides"
        )

    except (ValueError, DataValidationError, ProcessingError) as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Process the ride(s) based on batch flag
    if args.batch:
        process_multiple_rides(args)
    else:
        process_single_ride(args)


if __name__ == "__main__":
    main()
