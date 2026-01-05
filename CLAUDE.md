# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MtbViz is a mountain bike telemetry processing pipeline and visualization tool. It processes GPS, accelerometer, and gyroscope data from the Sensor Logger smartphone app to generate ride metrics (jump detection, g-force analysis, intensity scores) and interactive visualizations.

## Commands

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Processing Data
```bash
# Process a single ride from file
process --input data/raw/ride.zip

# Process from PostgreSQL database
process --from-db --ride-id <uuid>

# Batch process all rides in a directory
process --input data/raw --batch

# Custom settings
process --input ride.zip --downsample-freq 10 --jump-threshold 2.0 --verbose
```

### Visualization
```bash
# Single ride
viz --input data/processed/ride.parquet

# Compare multiple rides
viz --input ride1.parquet ride2.parquet --names "Ride 1" "Ride 2"

# With jump cluster overlay (requires database)
viz --input ride.parquet --jumps

# Custom port
viz --input ride.parquet --port 8080
```

### Testing
```bash
# Verify requirements
python tests/test_requirements.py

# Run pipeline tests
python tests/test_pipeline.py

# Test with sample data
python tests/test_pipeline.py path/to/sample.zip
```

## Architecture

### Pipeline Components (`mtb/src/pipeline.py`)

The data processing follows a three-stage pipeline pattern:

1. **DataLoader** - Loads sensor data from ZIP files, directories, or PostgreSQL. Handles CSV parsing, timestamp conversion, resampling to target frequency, and optional data binning.

2. **FeatureBuilder** - Transforms device-frame coordinates to world frame using quaternion rotations (`scipy.spatial.transform.Rotation`). Calculates magnitude columns, identifies jumps via z-acceleration threshold detection, and adds optional lead/lag features.

3. **MetricsCalculator** - Computes summary metrics: Haversine distance, segment-based elevation gain (using `scipy.signal.find_peaks`), moving time, intensity scores, and jump event extraction.

4. **Pipeline** - Orchestrates the workflow, manages output files (parquet for data, JSON for metrics).

### Configuration

`ProcessingConfig` dataclass centralizes all parameters. Key settings:
- `downsample_freq`: Target resampling frequency (default 5Hz)
- `jump_threshold`: Z-acceleration threshold for jump detection (default 1.5 m/s²)
- `jump_min_consecutive`: Minimum points for valid jump (default 2)
- `distance_smooth_window`: GPS smoothing window (default 21)

### Visualization App (`mtb/app/app_vis.py`)

Dash web app with:
- Plotly time series plots with dynamic LTTB-inspired resampling based on zoom level
- Dash-Leaflet map with GPS track overlay and Google Maps tiles
- Synchronized hover between plot and map
- Multi-ride comparison with side-by-side metrics table
- Jump cluster visualization from database (ellipses showing takeoff/landing locations)

### Database Models (`mtb/db/models.py`)

SQLAlchemy ORM models for PostgreSQL:
- `Ride`: Core ride metadata with relationships to sensor timeseries tables
- `Jump`/`JumpGlobal`: Individual and clustered jump events
- Timeseries tables: `Location`, `AccelDevice`, `GyroDevice`, `Orientation`, `Barometer`

## Key Coordinate Transformations

Sensor data in device frame is transformed to world (earth) frame:
- Uses quaternion orientation data (qx, qy, qz, qw)
- Device columns: `ax_uc_device`, `gx_device` etc.
- Earth frame columns: `ax_uc_e`, `gx_e` etc.
- Horizontal acceleration: `ah_uc_e` (for cornering intensity)

## Data Flow

```
Sensor Logger ZIP/DB → DataLoader → FeatureBuilder → MetricsCalculator
                                         ↓
                              Processed .parquet + .json metrics
                                         ↓
                                   viz (Dash app)
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string for database mode
- `GOOGLE_MAPS_API_KEY`: Required for map tiles in visualization app
