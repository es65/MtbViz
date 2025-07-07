from dash import Dash, dcc, html, Output, Input
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import plotly.graph_objects as go
import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv


def create_plot(columns):
    fig = go.Figure()

    for col in columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col, hoverinfo="x+y+name"
            )
        )

    fig.update_layout(hovermode="x unified")
    return fig


def load_summary_metrics(input_file_path):
    """Load and format summary metrics from JSON file."""
    try:
        # Convert input file path to summary metrics file path
        input_path = Path(input_file_path)
        summary_path = input_path.parent / f"{input_path.stem}.json"

        if not summary_path.exists():
            return None

        with open(summary_path, "r") as f:
            metrics = json.load(f)

        # Format metrics for display
        formatted_metrics = {}

        # Duration metrics
        if "duration_hr" in metrics:
            formatted_metrics["Total Duration"] = f"{metrics['duration_hr']:.2f} hours"
        if "duration_moving_hr" in metrics:
            formatted_metrics["Moving Duration"] = (
                f"{metrics['duration_moving_hr']:.2f} hours"
            )

        # Distance and speed metrics
        if "distance_haversine" in metrics:
            formatted_metrics["Total Distance"] = (
                f"{metrics['distance_haversine']:.2f} miles"
            )
        if "avg_speed_mph" in metrics:
            formatted_metrics["Average Speed"] = f"{metrics['avg_speed_mph']:.1f} mph"

        # Elevation metrics
        if "elevation_gain" in metrics:
            formatted_metrics["Elevation Gain"] = (
                f"{metrics['elevation_gain']:.0f} feet"
            )

        # Intensity metrics
        if "avg_overall_intensity" in metrics:
            formatted_metrics["Overall Intensity"] = (
                f"{metrics['avg_overall_intensity']:.2f} m/s²"
            )
        if "avg_cornering_intensity" in metrics:
            formatted_metrics["Cornering Intensity"] = (
                f"{metrics['avg_cornering_intensity']:.2f} m/s²"
            )

        # Jump metrics
        if "longest_jump_ft" in metrics:
            formatted_metrics["Longest Jump"] = f"{metrics['longest_jump_ft']:.1f} feet"
        if "total_jump_airtime_s" in metrics:
            formatted_metrics["Total Jump Airtime"] = (
                f"{metrics['total_jump_airtime_s']:.1f} seconds"
            )
        if "total_jump_distance_s" in metrics:
            formatted_metrics["Total Jump Distance"] = (
                f"{metrics['total_jump_distance_s']:.1f} feet"
            )

        return formatted_metrics

    except Exception as e:
        print(f"Error loading summary metrics: {e}")
        return None


load_dotenv()  # Load variables from .env file
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="MtbViz - Interactive mountain biking data visualization"
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    default="./data/processed/example_joaquinmiller_241117.parquet",
    help="Path to the input .parquet file (default: ./data/processed/example_joaquinmiller_241117.parquet)",
)
args = parser.parse_args()

df = pd.read_parquet(args.input)
df = df.dropna(subset=["latitude", "longitude"])

# Load summary metrics
summary_metrics = load_summary_metrics(args.input)

# Edit line below to set index as elapsed_seconds (uncomment) or keep as datetime (comment out):
df.set_index("elapsed_seconds", drop=True, inplace=True)

if isinstance(df.index, pd.DatetimeIndex):
    timezone = df.index.tz.zone

route_positions = list(zip(df["latitude"], df["longitude"]))

# Lat lon bounds with padding for initial map view
lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

lat_padding = 0.05 * (lat_max - lat_min)
lon_padding = 0.05 * (lon_max - lon_min)

bounds = [
    [lat_min - lat_padding, lon_min - lon_padding],  # Southwest corner
    [lat_max + lat_padding, lon_max + lon_padding],  # Northeast corner
]

data_sets = {
    "accel, device": ["ax_device", "ay_device", "az_device", "ar"],
    "accel, earth": ["ax_e", "ay_e", "az_e", "ar"],
    "accel, uncalib, device": [
        "ax_uc_device",
        "ay_uc_device",
        "az_uc_device",
        "ar_uc",
    ],
    "accel, uncalib, earth": ["ax_uc_e", "ay_uc_e", "az_uc_e", "ar_uc"],  # default
    "gyro, device": ["gx_device", "gy_device", "gz_device", "gr"],
    "gyro, earth": ["gx_e", "gy_e", "gz_e", "gr"],
    "gravity, device": ["gFx_device", "gFy_device", "gFz_device", "gFr"],
    "gravity, earth": ["gFx_e", "gFy_e", "gFz_e", "gFr"],
    "orientation": ["qx", "qy", "qz", "yaw", "roll", "pitch"],
    "speed": ["speed"],
    "altitude": ["altitude", "altitudeAboveMeanSeaLevel"],
}

app = Dash(__name__)


app.layout = html.Div(
    [
        # html.H1("Interactive Plotly Graph and Leaflet Map"),
        dcc.Graph(
            id="plotly-graph", figure=create_plot(data_sets["accel, uncalib, earth"])
        ),
        html.Div(
            [
                html.Label("Select Dataset:"),
                dcc.Dropdown(
                    id="dataset-selector",
                    options=[
                        {"label": key.capitalize(), "value": key}
                        for key in data_sets.keys()
                    ],
                    value="accel, uncalib, earth",  # Default dataset
                    clearable=False,
                    style={"width": "200px"},
                ),
            ],
            style={
                "position": "absolute",
                "top": "10px",
                "right": "700px",
                "zIndex": "1000",
            },
        ),
        html.Div(
            [
                html.Label("Select Map Layer:"),
                dcc.Dropdown(
                    id="layer-selector",
                    options=[
                        {"label": "Standard", "value": "m"},
                        {"label": "Satellite", "value": "y"},
                        {"label": "Terrain", "value": "p"},
                    ],
                    value="m",  # Default to standard layer
                    clearable=False,
                    style={"width": "200px"},
                ),
            ],
            style={
                "position": "absolute",
                "top": "10px",
                "right": "470px",
                "zIndex": "1000",
            },
        ),
        html.Div(
            [
                html.Label("Show Route:"),
                dcc.Checklist(
                    id="route-toggle",
                    options=[{"label": "Show", "value": "show"}],
                    value=["show"],  # Default to showing the route
                    style={"display": "inline-block"},  # , "margin-left": "10px"}
                ),
            ],
            style={
                "position": "absolute",
                "top": "10px",
                "right": "280px",
                "zIndex": "1000",
            },
        ),
        dl.Map(
            id="map",
            bounds=bounds,
            style={"width": "95%", "height": "400px", "margin": "0 auto"},
            children=[
                # dl.TileLayer(),  # Base map
                dl.TileLayer(
                    id="base-layer",
                    url=f"https://mt1.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}",
                    # satellite: lyrs=s or y; standard roadmap: lyrs=m; terrain: lyrs=p; hybrid: lyrs=hybrid or
                    attribution="Google Maps",
                    maxZoom=22,
                ),
                dl.Polyline(
                    id="route-line", positions=route_positions, color="blue"
                ),  # Route
                dl.Marker(
                    id="marker",
                    position=[df["latitude"].iloc[0], df["longitude"].iloc[0]],
                ),  # Marker to update
                dl.ScaleControl(position="bottomleft"),
            ],
        ),
        # Summary Metrics Section
        html.Div(
            [
                html.H2(
                    "Summary Metrics",
                    style={
                        "textAlign": "center",
                        "marginTop": "30px",
                        "marginBottom": "20px",
                    },
                ),
                html.Div(
                    id="summary-metrics-content",
                    style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"},
                ),
            ],
            style={"marginTop": "30px", "marginBottom": "30px"},
        ),
    ]
)


# Callback to update the graph based on dataset selection
@app.callback(Output("plotly-graph", "figure"), Input("dataset-selector", "value"))
def update_graph(selected_dataset):
    columns = data_sets[selected_dataset]
    return create_plot(columns)


# Callback to update the marker position
@app.callback(Output("marker", "position"), Input("plotly-graph", "hoverData"))
def update_marker(hover_data):
    if hover_data is None:
        # Default location if no hover data
        lat, lon = df["latitude"].iloc[0], df["longitude"].iloc[0]
    else:
        # Get the datetime of the hovered point
        hover_datetime = hover_data["points"][0]["x"]
        if isinstance(df.index, pd.DatetimeIndex):
            hover_datetime = pd.to_datetime(hover_datetime).tz_localize(timezone)
            print(hover_datetime, "\n")

        # Match the datetime in the DataFrame
        row = df.loc[hover_datetime]
        lat, lon = row["latitude"], row["longitude"]

    # Return the updated marker position
    return [lat, lon]


# Callback to update the map layer based on dropdown selection
@app.callback(Output("base-layer", "url"), Input("layer-selector", "value"))
def update_map_layer(selected_layer):
    # Update the TileLayer's URL based on the selected layer
    return f"https://mt1.google.com/vt/lyrs={selected_layer}&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}"


# Callback to toggle the visibility of the route line
@app.callback(Output("route-line", "positions"), Input("route-toggle", "value"))
def toggle_route_line(toggle_value):
    # Return the Polyline only if "show" is in the checklist's value
    if "show" in toggle_value:
        return route_positions
    return []


# Callback to populate summary metrics
@app.callback(
    Output("summary-metrics-content", "children"), Input("dataset-selector", "value")
)
def update_summary_metrics(selected_dataset):
    if summary_metrics is None:
        return html.Div(
            "No summary metrics available.",
            style={
                "textAlign": "center",
                "color": "#666",
                "fontStyle": "italic",
                "fontSize": "16px",
            },
        )

    table_rows = []
    for metric_name, metric_value in summary_metrics.items():
        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        metric_name,
                        style={
                            "padding": "12px 20px",
                            "fontWeight": "bold",
                            "backgroundColor": "#f8f9fa",
                            "borderBottom": "1px solid #dee2e6",
                            "textAlign": "left",
                        },
                    ),
                    html.Td(
                        metric_value,
                        style={
                            "padding": "12px 20px",
                            "backgroundColor": "white",
                            "borderBottom": "1px solid #dee2e6",
                            "textAlign": "right",
                            "fontFamily": "monospace",
                            "fontSize": "14px",
                        },
                    ),
                ]
            )
        )

    return html.Table(
        table_rows,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "border": "1px solid #dee2e6",
            "borderRadius": "8px",
            "overflow": "hidden",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )


def main():
    """Main function to run the Dash app"""
    app.run_server(debug=True)


# Run the Dash app
if __name__ == "__main__":
    main()
