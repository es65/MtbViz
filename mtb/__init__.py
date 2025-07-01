"""
Mountain Bike Data Tracker and Visualizer

A Python package for processing and analyzing mountain bike sensor data.
"""

from . import src

# Expose key modules for easier access
from .src import config, build_features, summary_metrics, make_dataset_sl

__version__ = "0.1.0"
__author__ = "Evan Sims"

# Convenience imports for common functionality
__all__ = ["config", "build_features", "summary_metrics", "make_dataset_sl", "src"]
