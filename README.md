# Mountain Bike Data Tracker and Visualizer

Cool metrics and visualization tools for serious mountain bike riders. Works with a smartphone and the Sensor Logger app to collect GPS, accelerometer, and gyroscope data.

## Background and Goals

The basic idea is to provide metrics for serious mountain bike riders that are both motivating and genuinely useful for the purpose of coaching/improving your riding. These include:

1. Jump airtime and distance
1. Impact and cornering G-forces
1. Overall avg ride intensity
1. Avg cornering intensity
1. Most common geo metrics found on Strava

Other metrics that I haven't implemented yet:
1. Boost intensity
1. Whip angle and duration
1. Resolving left and right cornering
1. Crash detection

I'm working on turning this into an iOS app, but wanted to open source some of the code, especially the visualizer, in the meantine.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pyenv (recommended for version management)
- smartphone

### Python Setup

```bash
# Clone repository
git clone https://github.com/es65/MtbViz
cd MtbViz

# Install (if needed) and Set Python version 3.8-3.12:
pyenv install 3.12.0
pyenv local 3.12.0

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
python -m pytest
```

### Install and Set Up Sensor Logger

Sensor Logger collects the accelerometer, gyroscope, and GPS data that feeds this module.

1. Download the Sensor Logger app to your phone
   - iPhone: https://apps.apple.com/us/app/sensor-logger/id1531582925
   - Android: https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=en_US
2. Set app security to **_Allow Location Access ALWAYS_** and **_Enable Precise Location_**. This is necessary so the app can continue to record in the background.
3. Set sampling frequency: Logger > Gear Icon > Sampling Frequencies: change accelerometer etc to 20 Hz. Leave others at default values.
4. From the Logger screen, enable the following: Accelerometer, Gravity, Gyroscope, Orientation, Compass, Barometer, and Location.
5. Before a ride: from the Logger screen, click "Start Recording".
6. During a ride: you can start Strava and use other apps without trouble--Sensor Logger continues to run and record data in the background. Just don't close the app. After dozens of rides using this app, I have not noticed any serious impact on iPhone battery life.
7. After a ride: click "End Recording". Date files are saved locally on your phone and are shown in the Recordings tab. Whenever convenient (i.e. when you have good signal or WiFi), click on the data file for your ride and click "Export". Keep the default settings and click "Export Recording". Save to `MtbViz/mtb/data/raw`.

### Usage (work in progress)