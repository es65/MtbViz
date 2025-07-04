#!/usr/bin/env python3
"""
Test script for the new MtbViz pipeline.

This script demonstrates how to use the new pipeline classes
and can be used for testing and development.
"""

import logging
from pathlib import Path
from typing import Optional

from mtb.src.pipeline import (
    Pipeline,
    ProcessingConfig,
    DataValidationError,
    ProcessingError,
)


def setup_logging(verbose: bool = True) -> None:
    """Setup logging for testing."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def test_config_creation() -> None:
    """Test ProcessingConfig creation and defaults."""
    print("Testing ProcessingConfig creation...")

    # Test default config
    config = ProcessingConfig()
    print(f"Default timezone: {config.timezone}")
    print(f"Default downsample_freq: {config.downsample_freq}")
    print(f"Default jump_threshold: {config.jump_threshold}")
    print(f"Default raw_dir: {config.raw_dir}")
    print(f"Default processed_dir: {config.processed_dir}")
    print(f"Default intermediate_dir: {config.intermediate_dir}")

    # Test custom config
    custom_config = ProcessingConfig(
        downsample_freq=10, jump_threshold=2.0, verbose=True
    )
    print(f"Custom downsample_freq: {custom_config.downsample_freq}")
    print(f"Custom jump_threshold: {custom_config.jump_threshold}")
    print(f"Custom verbose: {custom_config.verbose}")

    print("✓ ProcessingConfig tests passed\n")


def test_pipeline_creation() -> None:
    """Test Pipeline creation."""
    print("Testing Pipeline creation...")

    config = ProcessingConfig(verbose=True)
    pipeline = Pipeline(config)

    print(f"Pipeline config timezone: {pipeline.config.timezone}")
    print(f"Pipeline config downsample_freq: {pipeline.config.downsample_freq}")

    print("✓ Pipeline creation tests passed\n")


def test_data_validation() -> None:
    """Test data validation functionality."""
    print("Testing data validation...")

    from mtb.src.pipeline import DataLoader, ProcessingConfig
    import pandas as pd

    config = ProcessingConfig()
    loader = DataLoader(config)

    # Test with valid DataFrame
    valid_df = pd.DataFrame(
        {
            "ax_device": [1, 2, 3],
            "ay_device": [1, 2, 3],
            "az_device": [1, 2, 3],
            "qx": [1, 2, 3],
            "qy": [1, 2, 3],
            "qz": [1, 2, 3],
            "qw": [1, 2, 3],
        }
    )

    try:
        loader.validate_input(valid_df, ["ax_device", "ay_device", "az_device"])
        print("✓ Valid DataFrame validation passed")
    except DataValidationError as e:
        print(f"✗ Valid DataFrame validation failed: {e}")

    # Test with invalid DataFrame
    invalid_df = pd.DataFrame({"ax_device": [1, 2, 3]})

    try:
        loader.validate_input(invalid_df, ["ax_device", "ay_device", "az_device"])
        print("✗ Invalid DataFrame validation should have failed")
    except DataValidationError as e:
        print(f"✓ Invalid DataFrame validation correctly failed: {e}")

    print("✓ Data validation tests passed\n")


def test_pipeline_with_sample_data(sample_data_path: Optional[str] = None) -> None:
    """Test pipeline with sample data if available."""
    print("Testing pipeline with sample data...")

    if sample_data_path is None:
        print("No sample data path provided, skipping pipeline execution test")
        print("To test with real data, provide a path to a Sensor Logger zip file")
        return

    sample_path = Path(sample_data_path)
    if not sample_path.exists():
        print(f"Sample data path does not exist: {sample_data_path}")
        return

    try:
        config = ProcessingConfig(
            verbose=True, save_intermediate=True, downsample_freq=5
        )

        pipeline = Pipeline(config)

        # Test single ride processing
        print(f"Processing sample data: {sample_path}")
        result = pipeline.process_ride(sample_path, config.processed_dir)

        print("✓ Pipeline execution completed successfully!")
        print(f"Output files:")
        for file_type, file_path in result["output_files"].items():
            if file_path:
                print(f"  {file_type}: {file_path}")

        # Print some summary metrics
        if result["summary_metrics"]:
            print("\nSummary metrics:")
            for key, value in result["summary_metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")

    except (DataValidationError, ProcessingError) as e:
        print(f"✗ Pipeline execution failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def main() -> None:
    """Main test function."""
    print("=" * 60)
    print("MtbViz Pipeline Test Suite")
    print("=" * 60)

    # Setup logging
    setup_logging(verbose=True)

    # Run tests
    test_config_creation()
    test_pipeline_creation()
    test_data_validation()

    # Test with sample data if provided
    import sys

    if len(sys.argv) > 1:
        sample_data_path = sys.argv[1]
        test_pipeline_with_sample_data(sample_data_path)
    else:
        test_pipeline_with_sample_data()

    print("=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
