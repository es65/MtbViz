#!/usr/bin/env python3
"""
Test script to verify all required packages can be imported successfully.
Run this after installing requirements.txt to ensure everything works.
"""


def test_imports():
    """Test all required package imports.
    Manually update if new packages are added.
    """
    try:
        import pandas as pd
        import numpy as np
        import scipy
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly
        import haversine
        import pywt
        import filterpy
        import dash
        import dash_leaflet
        import tqdm
        import yaml
        import dotenv

        print("\n🎉 All packages imported successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
