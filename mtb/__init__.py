"""
Mountain Bike Data Tracker and Visualizer

A Python package for processing and analyzing mountain bike sensor data.
"""

from . import src

# Expose key modules for easier access
from mtb.src import config, pipeline, cli

__version__ = "0.1.0"
__author__ = "Evan Sims"

# Convenience imports for common functionality
__all__ = ["config", "pipeline", "cli", "src"]
