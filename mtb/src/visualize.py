import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "browser"  # notebook

##################################

# User inputs:

full_path = "path to parquet file"

##################################

df = pd.read_parquet(full_path)
cols = df.columns
df.info()
x = "elapsed_seconds"
# Note: GPS data is about 1 Hz


ys = [
    "gx_device",
    "gx_uc_device",
    "gy_device",
    "gy_uc_device",
    "gz_uc_device",
    "gz_device",
    "gr",
]
df_plot = df.dropna(subset=ys)
fig = px.line(df_plot, x=x, y=ys, title="Gyroscope")
fig.show()


ys = ["roll", "pitch", "yaw"]
df_plot = df.dropna(subset=ys)
fig = px.line(df_plot, x=x, y=ys, title="Orientation")
fig.show()


ys = ["ar", "gr", "gFr", "ar_uc"]
df_plot = df.dropna(subset=ys)
fig = px.line(df_plot, x=x, y=ys, title="Magnitudes")
fig.show()


ys = ["gFx_device", "gFy_device", "gFz_device", "gFr"]
if all([col in df.columns for col in ys]):
    df_plot = df.dropna(subset=ys)
    fig = px.line(df_plot, x=x, y=ys, title="Gravity")
    fig.show()
else:
    print("One or more columns not found in DataFrame:", ys)


ys = ["altitude", "altitudeAboveMeanSeaLevel", "relativeAltitude"]
alt_init = df["altitudeAboveMeanSeaLevel"].dropna().iloc[:3].mean()
if all([col in df.columns for col in ys]):
    df_plot = df.dropna(subset=ys)
    df_plot["alt_by_p"] = df["relativeAltitude"] + alt_init
    ys.append("alt_by_p")
    fig = px.line(df_plot, x=x, y=ys, title="Altitude")
    fig.show()
else:
    print("One or more columns not found in DataFrame:", ys)


ys = ["speedAccuracy", "bearingAccuracy", "horizontalAccuracy", "verticalAccuracy"]
if all([col in df.columns for col in ys]):
    df_plot = df.dropna(subset=ys)
    fig = px.line(df_plot, x=x, y=ys, title="Accuracy")
    fig.show()
else:
    print("One or more columns not found in DataFrame:", ys)


ys = ["bearing", "magneticBearing"]
if all([col in df.columns for col in ys]):
    df_plot = df.dropna(subset=ys)
    fig = px.line(df_plot, x=x, y=ys, title="Bearing")
    fig.show()
else:
    print("One or more columns not found in DataFrame:", ys)

# Ensure 'speed' and 'speedAccuracy' columns exist in the DataFrame
if "speed" in df.columns and "speedAccuracy" in df.columns:
    # Prepare data
    x = df["elapsed_seconds"]
    y = df["speed"]
    error = df["speedAccuracy"]

    # Create the plot
    fig = go.Figure()

    # Add shaded error boundary
    fig.add_trace(
        go.Scatter(
            x=list(x)
            + list(x[::-1]),  # x values for the boundary (forward and reverse)
            y=list(y + error) + list((y - error)[::-1]),  # Upper and lower bounds
            fill="toself",
            fillcolor="rgba(0,100,200,0.3)",  # Semi-transparent fill color
            line=dict(color="rgba(255,255,255,0)"),  # No line for the boundary
            hoverinfo="skip",  # Skip hover info for the boundary
            showlegend=False,  # Do not show in legend
        )
    )

    # Add the main line plot
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue"), name="Speed")
    )

    # Customize layout
    fig.update_layout(
        title="Speed with Error Boundary",
        xaxis_title="Elapsed Seconds",
        yaxis_title="Speed (m/s)",
        template="plotly_white",
    )

    # Show the plot
    fig.show()
else:
    print("Columns 'speed' and/or 'speedAccuracy' not found in the DataFrame.")


# Ensure 'speed' and 'speedAccuracy' columns exist in the DataFrame
if "bearing" in df.columns and "bearingAccuracy" in df.columns:
    # Prepare data
    x = df["elapsed_seconds"]
    y = df["bearing"]
    error = df["bearingAccuracy"]

    # Create the plot
    fig = go.Figure()

    # Add shaded error boundary
    fig.add_trace(
        go.Scatter(
            x=list(x)
            + list(x[::-1]),  # x values for the boundary (forward and reverse)
            y=list(y + error) + list((y - error)[::-1]),  # Upper and lower bounds
            fill="toself",
            fillcolor="rgba(0,100,200,0.3)",  # Semi-transparent fill color
            line=dict(color="rgba(255,255,255,0)"),  # No line for the boundary
            hoverinfo="skip",  # Skip hover info for the boundary
            showlegend=False,  # Do not show in legend
        )
    )

    # Add the main line plot
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue"), name="Speed")
    )

    # Customize layout
    fig.update_layout(
        title="Bearing with Error Boundary",
        xaxis_title="Elapsed Seconds",
        yaxis_title="Bearing (deg)",
        template="plotly_white",
    )

    # Show the plot
    fig.show()
else:
    print("One or more cols not found in the DataFrame.")


"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import folium
from cycler import cycler

colors = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]

mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "white"
mpl.rcParams["axes.prop_cycle"] = colors
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.titlesize"] = 25
mpl.rcParams["figure.dpi"] = 100


map_df = df_200ms.iloc[:,1:11].dropna()
coords = [(row.latitude, row.longitude) for _, row in map_df.iterrows()]

my_map = folium.Map(location=[map_df.latitude.mean(), map_df.longitude.mean()], 
                    zoom_start=16,)
folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)

my_map
"""
