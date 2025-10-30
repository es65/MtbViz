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


def resample_dataframe(data, x_range=None, max_points=2000):
    """
    Intelligently resample dataframe based on visible x-axis range.

    Args:
        data: DataFrame to resample
        x_range: Tuple of (min, max) for x-axis range, or None for full range
        max_points: Maximum number of points to show

    Returns:
        Tuple of (resampled DataFrame, original filtered count)
    """
    try:
        if x_range is not None:
            # Filter to visible range
            x_min, x_max = x_range

            # Convert range values to match index type if needed
            if isinstance(data.index, pd.DatetimeIndex):
                x_min = pd.to_datetime(x_min)
                x_max = pd.to_datetime(x_max)

            mask = (data.index >= x_min) & (data.index <= x_max)
            filtered_data = data[mask]
        else:
            filtered_data = data

        # Store original filtered count before resampling
        original_count = len(filtered_data)

        # If already under max_points, return as-is
        if len(filtered_data) <= max_points:
            return filtered_data, original_count

        # Handle edge case of very small data
        if len(filtered_data) < 2:
            return filtered_data, original_count

        # Calculate stride for downsampling
        stride = max(1, len(filtered_data) // max_points)

        # Use LTTB-like approach: keep first, last, and evenly distributed points
        indices = []

        # Always include first point
        indices.append(0)

        # Sample points with stride
        for i in range(stride, len(filtered_data) - stride, stride):
            indices.append(i)

        # Always include last point
        if len(filtered_data) - 1 not in indices:
            indices.append(len(filtered_data) - 1)

        return filtered_data.iloc[indices], original_count

    except Exception as e:
        print(f"Error in resample_dataframe: {e}")
        # If resampling fails, return original data
        return data, len(data)


def create_plot(columns, data_to_plot=None, show_markers=False, view_total_points=None):
    """
    Create a plotly figure with the specified columns.

    Args:
        columns: List of column names to plot
        data_to_plot: DataFrame to use for plotting (if None, uses global df)
        show_markers: Whether to show markers on the lines
        view_total_points: Total points in current view before resampling (if None, uses len(df))
    """
    fig = go.Figure()

    # Use provided data or fall back to global df
    plot_data = data_to_plot if data_to_plot is not None else df

    # Filter to only columns that exist in the dataframe
    available_columns = [col for col in columns if col in df.columns]

    # Set mode based on marker preference
    mode = "lines+markers" if show_markers else "lines"
    marker_dict = dict(size=4) if show_markers else None

    for col in available_columns:
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col],
                mode=mode,
                name=col,
                hoverinfo="x+y+name",
                marker=marker_dict,
            )
        )

    fig.update_layout(
        hovermode="x unified",
        uirevision="constant",  # Preserve UI state (zoom/pan) across updates
    )

    # Add annotations for data status
    annotations = []
    annotation_y_pos = 1.05

    # Add a note if some columns were missing
    if len(available_columns) < len(columns):
        missing_cols = [col for col in columns if col not in df.columns]
        annotations.append(
            dict(
                text=f"Note: {len(missing_cols)} column(s) not available in data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=annotation_y_pos,
                showarrow=False,
                font=dict(size=10, color="orange"),
            )
        )
        annotation_y_pos += 0.05

    # Add resampling info even if all data shown:
    total_points_in_view = (
        view_total_points if view_total_points is not None else len(df)
    )
    shown_points = len(plot_data)

    sampling_ratio = shown_points / total_points_in_view * 100
    annotations.append(
        dict(
            text=f"Showing {shown_points:,} of {total_points_in_view:,} points in view ({sampling_ratio:.1f}%) - Zoom in for more detail",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=9, color="gray"),
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
    )

    if annotations:
        fig.update_layout(annotations=annotations)

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
parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=8050,
    help="Port to run the Dash app on (default: 8050)",
)
args = parser.parse_args()

# Extract ride name from input file path
input_path = Path(args.input)
ride_name = input_path.stem  # Gets filename without extension

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
    # "accel, device": ["ax_device", "ay_device", "az_device", "ar"],
    # "accel, earth": ["ax_e", "ay_e", "az_e", "ar"],
    "accel, uncalib, device": [
        "ax_uc_device",
        "ay_uc_device",
        "az_uc_device",
        "ar",
    ],
    "accel, uncalib, earth": [
        "ax_uc_e",
        "ay_uc_e",
        "az_uc_e",
        "ar",
        "ah_uc_e",
    ],  # default
    "gyro, device": ["gx_device", "gy_device", "gz_device", "gr"],
    "gyro, earth": ["gx_e", "gy_e", "gz_e", "gr"],
    # "gravity, earth": ["gFx_e", "gFy_e", "gFz_e", "gFr"],
    "orientation": ["qx", "qy", "qz", "yaw", "roll", "pitch"],
    "speed": ["speed"],
    "altitude": ["altitude"],
}

# Filter datasets to only include those with at least one available column
available_data_sets = {
    key: cols
    for key, cols in data_sets.items()
    if any(col in df.columns for col in cols)
}

# Select default dataset (prefer the original default if available, otherwise pick the first available)
default_dataset = (
    "accel, uncalib, earth"
    if "accel, uncalib, earth" in available_data_sets
    else list(available_data_sets.keys())[0] if available_data_sets else None
)

if not available_data_sets:
    print("Warning: No data sets have available columns in the dataframe")
    available_data_sets = {"speed": ["speed"]}  # Fallback
    default_dataset = "speed"

app = Dash(__name__)


app.layout = html.Div(
    [
        # html.H1("Interactive Plotly Graph and Leaflet Map"),
        dcc.Graph(
            id="plotly-graph", figure=create_plot(available_data_sets[default_dataset])
        ),
        # Ride title at top left
        html.Div(
            html.H2(
                ride_name,
                style={
                    "margin": "0",
                    "fontSize": "20px",
                    "fontWeight": "600",
                    "color": "#2c3e50",
                },
            ),
            style={
                "position": "absolute",
                "top": "10px",
                "left": "20px",
                "zIndex": "1000",
            },
        ),
        html.Div(
            [
                html.Label("Select Dataset:"),
                dcc.Dropdown(
                    id="dataset-selector",
                    options=[
                        {"label": key.capitalize(), "value": key}
                        for key in available_data_sets.keys()
                    ],
                    value=default_dataset,  # Default dataset
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
                    options=[{"label": "", "value": "show"}],
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
        html.Div(
            [
                html.Label("Show Markers:"),
                dcc.Checklist(
                    id="marker-toggle",
                    options=[{"label": "", "value": "show"}],
                    value=[],  # Default to not showing markers
                    style={"display": "inline-block"},
                ),
            ],
            style={
                "position": "absolute",
                "top": "10px",
                "right": "80px",
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


# Callback to update the graph based on dataset selection, marker toggle, and zoom level
@app.callback(
    Output("plotly-graph", "figure"),
    Input("dataset-selector", "value"),
    Input("marker-toggle", "value"),
    Input("plotly-graph", "relayoutData"),
)
def update_graph(selected_dataset, marker_toggle, relayout_data):
    """
    Update graph with resampling based on zoom level.

    When zoomed out, shows fewer points for performance.
    When zoomed in, shows more detail up to full resolution.
    """
    columns = available_data_sets[selected_dataset]
    show_markers = "show" in marker_toggle

    # Extract x-axis range from relayout data
    x_range = None
    if relayout_data is not None:
        # Handle different relayout event types
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            x_range = (relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"])
        elif "xaxis.range" in relayout_data:
            x_range = tuple(relayout_data["xaxis.range"])

    # Determine max_points based on zoom level and markers
    # When markers are enabled, use fewer points for better performance
    # When zoomed in significantly, allow more points
    if x_range is not None:
        visible_ratio = (x_range[1] - x_range[0]) / (df.index.max() - df.index.min())
        # More zoom = more points allowed
        if visible_ratio < 0.01:  # Very zoomed in (< 1% of data visible)
            max_points = 5000
        elif visible_ratio < 0.1:  # Moderately zoomed in (< 10%)
            max_points = 3000
        else:
            max_points = 2000
    else:
        # Default: show full view with moderate point count
        max_points = 1500

    # Further reduce points if markers are enabled
    if show_markers:
        max_points = min(max_points, 2000)

    # Resample the dataframe
    resampled_df, view_total = resample_dataframe(
        df, x_range=x_range, max_points=max_points
    )

    return create_plot(
        columns,
        data_to_plot=resampled_df,
        show_markers=show_markers,
        view_total_points=view_total,
    )


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
    app.run_server(host="0.0.0.0", port=args.port, debug=True)


# Run the Dash app
if __name__ == "__main__":
    main()
