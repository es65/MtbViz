from dash import Dash, dcc, html, Output, Input
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# import pytz # timezone converter - unneeded / unused

# Good examples:
# Briones, many jumps: './data/processed/2025-03-29_23-55-46_5Hz.parquet'

df = pd.read_parquet("./data/processed_250530/2025-03-29_23-55-46_5Hz.parquet")

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


# Create the Plotly graph
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


# Could not get the code below to work for custom hover box - %{name} would appear as literal instead of column name
"""
def create_plot(columns):
    fig = go.Figure()

    for i, col in enumerate(columns):
        
        
        # For the first trace, include elapsed_seconds in hovertemplate
        if i == 0:
            hovertemplate = (
                "Elapsed: %{customdata:.1f} s<br>"  # One decimal place
                "%{name}: %{y}<extra></extra>"
            )
        else:
            # For subsequent traces, omit elapsed_seconds
            hovertemplate = (
                "%{name}: %{y}<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                # Pass elapsed_seconds (rounded if desired) as customdata
                customdata=df["elapsed_seconds"].round(1),
                hoverinfo="skip",
                hovertemplate=hovertemplate
            )
        )

    # Unify the hover so that x (time) is shown only once at the top
    fig.update_layout(hovermode="x unified") # "x unified" is default
    return fig

"""


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


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
