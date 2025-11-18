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


data_sets = {
    # "accel, device": ["ax_device", "ay_device", "az_device", "ar"],
    # "accel, earth": ["ax_e", "ay_e", "az_e", "ar"],
    "viz vs mtb2": [
        "az_uc_e",
        "az_uc_e_filt",
        "az_uc_e_az_uc_e_wavelet_db4_l3_t1.2_butter_5Hz",
        "jump",
        "jump_wavebutt_hyst",
    ],
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
    "az_uc_e, filt": [
        "az_uc_e",
        "az_uc_e_az_uc_e_wavelet_db4_l3_t1.2_butter_5Hz",
        "ah_uc_e",
        "jump_wavebutt_hyst",
    ],  # default
    "gyro, device": ["gx_device", "gy_device", "gz_device", "gr"],
    "gyro, earth": ["gx_e", "gy_e", "gz_e", "gr"],
    # "gravity, earth": ["gFx_e", "gFy_e", "gFz_e", "gFr"],
    "orientation": ["qx", "qy", "qz", "qw", "yaw", "roll", "pitch"],
    "speed": ["speed"],
    "altitude": ["altitude"],
}


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


def create_plot(
    columns, rides_to_plot=None, show_markers=False, view_total_points_list=None
):
    """
    Create a plotly figure with the specified columns for multiple rides.

    Args:
        columns: List of column names to plot
        rides_to_plot: List of ride dicts with 'name' and 'df' keys (if None, uses global rides)
        show_markers: Whether to show markers on the lines
        view_total_points_list: List of total points per ride before resampling (if None, uses len(df) for each)
    """
    fig = go.Figure()

    # Use provided rides or fall back to global rides
    plot_rides = rides_to_plot if rides_to_plot is not None else rides

    # Determine if we have multiple rides for legend formatting
    multiple_rides = len(plot_rides) > 1

    # Color palette for different rides
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    total_shown_points = 0
    total_view_points = 0
    all_missing_cols = []

    for ride_idx, ride in enumerate(plot_rides):
        ride_name = ride["name"]
        ride_df = ride["df"]

        # Filter to only columns that exist in this ride's dataframe
        available_columns = [col for col in columns if col in ride_df.columns]
        missing_cols = [col for col in columns if col not in ride_df.columns]

        if missing_cols:
            all_missing_cols.extend([(ride_name, col) for col in missing_cols])

        # Set mode based on marker preference
        mode = "lines+markers" if show_markers else "lines"

        # Get base color for this ride
        base_color = colors[ride_idx % len(colors)]

        for col in available_columns:
            # Format legend name
            if multiple_rides:
                legend_name = f"{ride_name} - {col}"
            else:
                legend_name = col

            fig.add_trace(
                go.Scatter(
                    x=ride_df.index,
                    y=ride_df[col],
                    mode=mode,
                    name=legend_name,
                    hoverinfo="x+y+name",
                    marker=dict(size=4, color=base_color) if show_markers else None,
                    line=dict(color=base_color),
                )
            )

        # Calculate points for this ride
        if view_total_points_list is not None and ride_idx < len(
            view_total_points_list
        ):
            total_view_points += view_total_points_list[ride_idx]
        else:
            total_view_points += len(ride_df)
        total_shown_points += len(ride_df)

    fig.update_layout(
        hovermode="x unified",
        uirevision="constant",  # Preserve UI state (zoom/pan) across updates
    )

    # Add annotations for data status
    annotations = []
    annotation_y_pos = 1.05

    # Add a note if some columns were missing
    if all_missing_cols:
        unique_missing = len(set(all_missing_cols))
        annotations.append(
            dict(
                text=f"Note: {unique_missing} column(s) not available in some rides",
                xref="paper",
                yref="paper",
                x=0.5,
                y=annotation_y_pos,
                showarrow=False,
                font=dict(size=10, color="orange"),
            )
        )
        annotation_y_pos += 0.05

    # Add resampling info
    sampling_ratio = (
        total_shown_points / total_view_points * 100 if total_view_points > 0 else 100
    )
    annotations.append(
        dict(
            text=f"Showing {total_shown_points:,} of {total_view_points:,} points in view ({sampling_ratio:.1f}%) - Zoom in for more detail",
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
    nargs="+",
    default=["./data/processed/example_joaquinmiller_241117.parquet"],
    help="Path(s) to the input .parquet file(s) (default: ./data/processed/example_joaquinmiller_241117.parquet)",
)
parser.add_argument(
    "--names",
    "-n",
    type=str,
    nargs="*",
    default=None,
    help="Optional short names for the input files (must match number of inputs)",
)
parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=8050,
    help="Port to run the Dash app on (default: 8050)",
)
args = parser.parse_args()

# Validate names argument if provided
if args.names is not None and len(args.names) != len(args.input):
    raise ValueError(
        f"Number of names ({len(args.names)}) must match number of inputs ({len(args.input)})"
    )

# Load all rides with their names
rides = []
for idx, input_file in enumerate(args.input):
    input_path = Path(input_file)

    # Determine the name for this ride
    if args.names is not None:
        ride_name = args.names[idx]
    else:
        # Use last 10 characters of filename (without extension)
        ride_name = (
            input_path.stem[-10:] if len(input_path.stem) > 10 else input_path.stem
        )

    # Load dataframe
    df = pd.read_parquet(input_file)
    df = df.dropna(subset=["latitude", "longitude"])

    # Set index as elapsed_seconds
    df.set_index("elapsed_seconds", drop=True, inplace=True)

    # Load summary metrics
    metrics = load_summary_metrics(input_file)

    # Store ride info
    rides.append(
        {
            "name": ride_name,
            "df": df,
            "metrics": metrics,
            "input_path": input_path,
        }
    )

# For backward compatibility, also keep a reference to the first dataframe as 'df'
df = rides[0]["df"]

if isinstance(df.index, pd.DatetimeIndex):
    timezone = df.index.tz.zone

# Calculate route positions and bounds for all rides
all_route_positions = []
all_lats = []
all_lons = []

for ride in rides:
    ride_df = ride["df"]
    positions = list(zip(ride_df["latitude"], ride_df["longitude"]))
    all_route_positions.append(positions)
    all_lats.extend(ride_df["latitude"].tolist())
    all_lons.extend(ride_df["longitude"].tolist())

# Lat lon bounds with padding for initial map view (encompassing all rides)
lat_min, lat_max = min(all_lats), max(all_lats)
lon_min, lon_max = min(all_lons), max(all_lons)

lat_padding = 0.05 * (lat_max - lat_min)
lon_padding = 0.05 * (lon_max - lon_min)

bounds = [
    [lat_min - lat_padding, lon_min - lon_padding],  # Southwest corner
    [lat_max + lat_padding, lon_max + lon_padding],  # Northeast corner
]


# Filter datasets to only include those with at least one available column in any ride
available_data_sets = {
    key: cols
    for key, cols in data_sets.items()
    if any(col in ride["df"].columns for ride in rides for col in cols)
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
            id="plotly-graph",
            figure=create_plot(available_data_sets[default_dataset]),
            style={"height": "600px"},
        ),
        # Ride title at top left (comma-separated list for multiple rides)
        html.Div(
            html.H2(
                ", ".join([ride["name"] for ride in rides]),
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
                "right": "930px",
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
                "right": "700px",
                "zIndex": "1000",
            },
        ),
        html.Div(
            [
                html.Label("Graph Height:"),
                dcc.Dropdown(
                    id="graph-height-selector",
                    options=[
                        {"label": "500px", "value": 500},
                        {"label": "600px", "value": 600},
                        {"label": "700px", "value": 700},
                    ],
                    value=600,  # Default to 600px
                    clearable=False,
                    style={"width": "120px"},
                ),
            ],
            style={
                "position": "absolute",
                "top": "10px",
                "right": "550px",
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
        html.Div(
            id="map-container",
            style={"width": "95%", "height": "400px", "margin": "0 auto"},
            children=[
                dl.Map(
                    id="map",
                    bounds=bounds,
                    style={"width": "100%", "height": "100%"},
                    children=[
                        # dl.TileLayer(),  # Base map
                        dl.TileLayer(
                            id="base-layer",
                            url=f"https://mt1.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}",
                            # satellite: lyrs=s or y; standard roadmap: lyrs=m; terrain: lyrs=p; hybrid: lyrs=hybrid or
                            attribution="Google Maps",
                            maxZoom=22,
                        ),
                        # Create a Polyline for each ride with different colors
                        *[
                            dl.Polyline(
                                id=f"route-line-{idx}",
                                positions=all_route_positions[idx],
                                color=["blue", "red", "green", "purple", "orange"][
                                    idx % 5
                                ],
                            )
                            for idx in range(len(rides))
                        ],
                        dl.Marker(
                            id="marker",
                            position=[df["latitude"].iloc[0], df["longitude"].iloc[0]],
                        ),  # Marker to update
                        dl.ScaleControl(position="bottomleft"),
                    ],
                ),
            ],
        ),
        # Summary Metrics Section
        html.Div(
            [
                html.Div(
                    id="summary-metrics-content",
                    style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"},
                ),
            ],
            style={"marginTop": "30px", "marginBottom": "30px"},
        ),
    ]
)


# Callback to update graph height
@app.callback(Output("plotly-graph", "style"), Input("graph-height-selector", "value"))
def update_graph_height(height):
    return {"height": f"{height}px"}


# Callback to update the graph based on dataset selection, marker toggle, and zoom level
@app.callback(
    Output("plotly-graph", "figure"),
    Input("dataset-selector", "value"),
    Input("marker-toggle", "value"),
    Input("plotly-graph", "relayoutData"),
)
def update_graph(selected_dataset, marker_toggle, relayout_data):
    """
    Update graph with resampling based on zoom level for all rides.

    When zoomed out, shows fewer points for performance.
    When zoomed in, shows more detail up to full resolution.
    """
    from dash import no_update

    columns = available_data_sets[selected_dataset]
    show_markers = "show" in marker_toggle

    # Extract x-axis range from relayout data
    x_range = None
    if relayout_data is not None:
        # Check if this is only a y-axis change (or other layout change that doesn't affect x-axis)
        # If so, don't recalculate to preserve current x-axis zoom and resampling
        has_xaxis_change = any(key.startswith("xaxis") for key in relayout_data.keys())
        has_autosize = "autosize" in relayout_data

        # If it's only y-axis or other changes (not x-axis), skip update
        if not has_xaxis_change and not has_autosize:
            return no_update

        # Handle different relayout event types
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            x_range = (relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"])
        elif "xaxis.range" in relayout_data:
            x_range = tuple(relayout_data["xaxis.range"])

    # Determine max_points based on zoom level and markers
    # When markers are enabled, use fewer points for better performance
    # When zoomed in significantly, allow more points
    if x_range is not None:
        # Calculate visible ratio based on first ride (they should be similar)
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
        max_points = 2000

    # Further reduce points if markers are enabled
    if show_markers:
        max_points = min(max_points, 2000)

    # Resample each ride's dataframe
    resampled_rides = []
    view_totals = []

    for ride in rides:
        ride_df = ride["df"]
        resampled_df, view_total = resample_dataframe(
            ride_df, x_range=x_range, max_points=max_points
        )

        resampled_rides.append(
            {
                "name": ride["name"],
                "df": resampled_df,
                "metrics": ride["metrics"],
            }
        )
        view_totals.append(view_total)

    return create_plot(
        columns,
        rides_to_plot=resampled_rides,
        show_markers=show_markers,
        view_total_points_list=view_totals,
    )


# Callback to update the marker position
@app.callback(Output("marker", "position"), Input("plotly-graph", "hoverData"))
def update_marker(hover_data):
    if hover_data is None:
        # Default location if no hover data
        lat, lon = df["latitude"].iloc[0], df["longitude"].iloc[0]
    else:
        # Get the hovered point information
        point = hover_data["points"][0]
        hover_datetime = point["x"]
        trace_name = point.get("data", {}).get("name", "")

        # Determine which ride this belongs to
        target_ride = None
        if len(rides) > 1:
            # For multiple rides, trace name is "name - col_name"
            # Extract the ride name
            if " - " in trace_name:
                ride_name = trace_name.split(" - ")[0]
                target_ride = next((r for r in rides if r["name"] == ride_name), None)

        # If we couldn't determine the ride or only one ride, use first ride
        if target_ride is None:
            target_ride = rides[0]

        target_df = target_ride["df"]

        # Handle datetime conversion if needed
        if isinstance(target_df.index, pd.DatetimeIndex):
            hover_datetime = pd.to_datetime(hover_datetime).tz_localize(timezone)
            print(hover_datetime, "\n")

        # Match the datetime in the DataFrame
        try:
            row = target_df.loc[hover_datetime]
            lat, lon = row["latitude"], row["longitude"]
        except KeyError:
            # If exact match not found, use closest
            idx = target_df.index.get_indexer([hover_datetime], method="nearest")[0]
            lat, lon = target_df["latitude"].iloc[idx], target_df["longitude"].iloc[idx]

    # Return the updated marker position
    return [lat, lon]


# Callback to update the map layer based on dropdown selection
@app.callback(Output("base-layer", "url"), Input("layer-selector", "value"))
def update_map_layer(selected_layer):
    # Update the TileLayer's URL based on the selected layer
    return f"https://mt1.google.com/vt/lyrs={selected_layer}&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}"


# Callbacks to toggle the visibility of all route lines
for idx in range(len(rides)):

    @app.callback(
        Output(f"route-line-{idx}", "positions"), Input("route-toggle", "value")
    )
    def toggle_route_line(toggle_value, route_idx=idx):
        # Return the Polyline only if "show" is in the checklist's value
        if "show" in toggle_value:
            return all_route_positions[route_idx]
        return []


# Callback to populate summary metrics
@app.callback(
    Output("summary-metrics-content", "children"), Input("dataset-selector", "value")
)
def update_summary_metrics(selected_dataset):
    # Check if any rides have metrics
    has_any_metrics = any(ride["metrics"] is not None for ride in rides)

    if not has_any_metrics:
        return html.Div(
            "No summary metrics available.",
            style={
                "textAlign": "center",
                "color": "#666",
                "fontStyle": "italic",
                "fontSize": "16px",
            },
        )

    # Collect all unique metric names across all rides
    all_metric_names = set()
    for ride in rides:
        if ride["metrics"] is not None:
            all_metric_names.update(ride["metrics"].keys())

    # Sort metric names for consistent ordering
    all_metric_names = sorted(all_metric_names)

    # Create header row with ride names
    header_cells = [
        html.Th(
            "Metric",
            style={
                "padding": "12px 20px",
                "backgroundColor": "#2c3e50",
                "color": "white",
                "borderBottom": "2px solid #dee2e6",
                "textAlign": "left",
                "fontWeight": "bold",
            },
        )
    ]

    for ride in rides:
        header_cells.append(
            html.Th(
                ride["name"],
                style={
                    "padding": "12px 20px",
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "borderBottom": "2px solid #dee2e6",
                    "textAlign": "right",
                    "fontWeight": "bold",
                },
            )
        )

    # Create data rows
    table_rows = [html.Tr(header_cells)]

    for metric_name in all_metric_names:
        row_cells = [
            html.Td(
                metric_name,
                style={
                    "padding": "12px 20px",
                    "fontWeight": "bold",
                    "backgroundColor": "#f8f9fa",
                    "borderBottom": "1px solid #dee2e6",
                    "textAlign": "left",
                },
            )
        ]

        for ride in rides:
            if ride["metrics"] is not None and metric_name in ride["metrics"]:
                value = ride["metrics"][metric_name]
            else:
                value = "-"

            row_cells.append(
                html.Td(
                    value,
                    style={
                        "padding": "12px 20px",
                        "backgroundColor": "white",
                        "borderBottom": "1px solid #dee2e6",
                        "textAlign": "right",
                        "fontFamily": "monospace",
                        "fontSize": "14px",
                    },
                )
            )

        table_rows.append(html.Tr(row_cells))

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
