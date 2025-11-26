# MtbViz: Mountain Bike Data Tracker and Visualizer

A data processing pipeline and visualization tool for mountain bike telemetry. Processes GPS, accelerometer, and gyroscope data from smartphone sensors to generate ride metrics and interactive visualizations.

![Dash web app](docs/assets/app_viz.png)

## Features

### Computed Metrics
- **Jump Detection**: Airtime, estimated distance, and GPS locations
- **G-Force Analysis**: Impact and cornering forces
- **Ride Intensity**: Overall and cornering intensity scores
- **Standard Metrics**: Distance, elevation gain, duration, average speed

### Visualization
- Interactive time series plots with multi-ride comparison
- GPS track overlay on satellite/terrain maps
- Real-time hover synchronization between plot and map
- Dynamic downsampling for smooth performance with large datasets
- Jump cluster visualization overlay (with database integration)

### Data Pipeline
- Processes raw sensor data from Sensor Logger app (CSV/ZIP)
- Database integration for ride storage and retrieval
- Coordinate frame transformations (device to world frame)
- Configurable resampling and feature engineering

## Quick Start

### Prerequisites
- Python 3.8+
- Smartphone with [Sensor Logger](https://apps.apple.com/us/app/sensor-logger/id1531582925) app

### Installation

```bash
git clone https://github.com/es65/MtbViz
cd MtbViz

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
python tests/test_requirements.py
```

### Sensor Logger Setup

1. Download [Sensor Logger](https://apps.apple.com/us/app/sensor-logger/id1531582925) (iOS) or [Android version](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp)
2. Configure permissions: **Allow Location Access ALWAYS** and **Enable Precise Location**
3. Set sampling frequency: Logger > Settings > 20 Hz for accelerometer/gyroscope
4. Enable sensors: Accelerometer, Gravity, Gyroscope, Orientation, Compass, Barometer, Location
5. After recording, export data as ZIP to `data/raw/`

## Usage

### Process and Visualize a Ride

```bash
# Process raw sensor data
process --input data/raw/ride.zip

# Launch interactive visualization
viz --input data/processed/ride.parquet
```

### Command Line Interface

**Processing Options:**

```bash
# Single ride from file
process --input data/raw/ride.zip

# From PostgreSQL database
process --from-db --ride-id <uuid>

# Batch process directory
process --input data/raw --batch

# Custom settings
process --input ride.zip --downsample-freq 10 --jump-threshold 2.0 --verbose
```

**Visualization Options:**

```bash
# Single ride
viz --input data/processed/ride.parquet

# Compare multiple rides
viz --input ride1.parquet ride2.parquet --names "Ride 1" "Ride 2"

# With jump overlay (requires database)
viz --input ride.parquet --jumps

# Custom port
viz --input ride.parquet --port 8080
```

### Python API

```python
from mtb.src.pipeline import Pipeline, ProcessingConfig

# Configure and run pipeline
config = ProcessingConfig(
    downsample_freq=10,
    jump_threshold=2.0,
    verbose=True
)
pipeline = Pipeline(config)

# Process from file
result = pipeline.process_ride(input_path="data/raw/ride.zip")

# Or from database
result = pipeline.process_ride(ride_id="12345678-1234-1234-1234-123456789abc")

# Access results
print(f"Distance: {result['summary_metrics']['distance_haversine']:.2f} miles")
print(f"Elevation: {result['summary_metrics']['elevation_gain']:.0f} feet")
print(f"Jumps: {len(result['jump_events_df']) if result['jump_events_df'] is not None else 0}")
```

## Architecture

```
mtb/
├── app/
│   └── app_vis.py      # Dash visualization app
├── db/
│   ├── models.py       # SQLAlchemy ORM models
│   └── session.py      # Database connection
├── src/
│   ├── cli.py          # Command line interface
│   ├── config.py       # Configuration constants
│   ├── pipeline.py     # Data processing pipeline
│   ├── plot_utils.py   # Plotting utilities
│   └── util.py         # DataFrame utilities
└── requirements.txt
```

### Pipeline Components

| Component | Description |
|-----------|-------------|
| `DataLoader` | Loads sensor data from ZIP files, directories, or PostgreSQL |
| `FeatureBuilder` | Transforms coordinates, calculates magnitudes, identifies jumps |
| `MetricsCalculator` | Computes ride statistics and extracts events |
| `Pipeline` | Orchestrates the complete workflow |

### Visualization Features

The Dash app includes:

- **Dataset Selector**: Switch between acceleration, gyroscope, orientation views
- **Map Layers**: Standard, satellite, and terrain views
- **Dynamic Resampling**: Automatically adjusts point density based on zoom level to maintain performance (LTTB-inspired algorithm)
- **Multi-Ride Comparison**: Overlay multiple rides with synchronized hover
- **Summary Metrics Table**: Side-by-side comparison of ride statistics

## Database Integration

For persistent storage and multi-device sync, configure PostgreSQL:

```bash
# Set in .env file
DATABASE_URL=postgresql://localhost/mtbviz
```

The database schema supports:
- User and bike management
- Ride metadata and metrics
- Time series sensor data (accelerometer, gyroscope, GPS, orientation)
- Jump event storage and clustering

## Configuration

Key processing parameters (adjustable via CLI or `ProcessingConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `downsample_freq` | 5 Hz | Target resampling frequency |
| `jump_threshold` | 1.5 m/s² | Z-acceleration threshold for jump detection |
| `jump_min_consecutive` | 2 | Minimum consecutive points for valid jump |
| `distance_smooth_window` | 21 | GPS smoothing window for distance calculation |
| `elevation_smooth_window` | 25 | Altitude smoothing window |

## Roadmap

Planned features:
- Boost intensity scoring
- Whip angle and duration detection
- Left/right cornering analysis
- Crash detection
- iOS companion app

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions welcome. Please open an issue to discuss proposed changes before submitting a PR.
