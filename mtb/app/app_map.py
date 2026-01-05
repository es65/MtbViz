"""
Full-screen Google Map visualization app for mountain bike rides.

Pulls ride and jump data from PostgreSQL database or parquet files.
"""

from dash import Dash, dcc, html, Output, Input, State, callback_context, ALL
import dash_leaflet as dl
import argparse
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from uuid import UUID

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Import database session
try:
    from mtb.db.session import SessionLocal

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Error: Database not available. This app requires database connection.")


def create_ellipse_points(center_lat, center_lng, lat_std, lng_std, num_points=50):
    """
    Create points for an ellipse shape.

    Args:
        center_lat: Center latitude
        center_lng: Center longitude
        lat_std: Standard deviation in latitude (determines height)
        lng_std: Standard deviation in longitude (determines width)
        num_points: Number of points to create for the ellipse

    Returns:
        List of [lat, lng] coordinates forming an ellipse
    """
    a = lat_std  # Height radius
    b = lng_std  # Width radius

    angles = np.linspace(0, 2 * np.pi, num_points)
    points = []

    for angle in angles:
        lat = center_lat + a * np.sin(angle)
        lng = center_lng + b * np.cos(angle)
        points.append([lat, lng])

    # Close the ellipse
    points.append(points[0])
    return points


def query_rides_by_ids(ride_ids):
    """
    Query rides table for specific ride IDs.

    Args:
        ride_ids: List of ride UUIDs

    Returns:
        List of ride dictionaries with id, title, and start_time
    """
    if not DB_AVAILABLE:
        return []

    try:
        db = SessionLocal()

        # Convert ride_ids to strings for SQL
        ride_id_strs = [str(rid) for rid in ride_ids]

        query = text(
            """
            SELECT id, title, start_time
            FROM rides
            WHERE id = ANY(CAST(:ride_ids AS uuid[]))
        """
        )

        result = db.execute(query, {"ride_ids": ride_id_strs})

        rides = []
        for row in result:
            rides.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "start_time": row.start_time,
                }
            )

        db.close()
        return rides

    except Exception as e:
        print(f"Error querying rides: {e}")
        return []


def query_rides_by_bbox(lat_min, lat_max, lon_min, lon_max):
    """
    Query rides that have any location points within the bounding box.

    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds

    Returns:
        List of ride dictionaries
    """
    if not DB_AVAILABLE:
        return []

    try:
        db = SessionLocal()

        query = text(
            """
            SELECT DISTINCT r.id, r.title, r.start_time
            FROM rides r
            JOIN location l ON r.id = l.ride_id
            WHERE l.latitude BETWEEN :lat_min AND :lat_max
              AND l.longitude BETWEEN :lon_min AND :lon_max
        """
        )

        result = db.execute(
            query,
            {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        )

        rides = []
        for row in result:
            rides.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "start_time": row.start_time,
                }
            )

        db.close()
        return rides

    except Exception as e:
        print(f"Error querying rides by bbox: {e}")
        return []


def query_locations_for_ride(ride_id):
    """
    Query location data for a specific ride.

    Args:
        ride_id: UUID of the ride

    Returns:
        List of (latitude, longitude) tuples ordered by timestamp
    """
    if not DB_AVAILABLE:
        return []

    try:
        db = SessionLocal()

        query = text(
            """
            SELECT latitude, longitude
            FROM location
            WHERE ride_id = CAST(:ride_id AS uuid)
            ORDER BY ts
        """
        )

        result = db.execute(query, {"ride_id": str(ride_id)})

        positions = []
        for row in result:
            if row.latitude is not None and row.longitude is not None:
                positions.append([row.latitude, row.longitude])

        db.close()
        return positions

    except Exception as e:
        print(f"Error querying locations for ride {ride_id}: {e}")
        return []


def get_bbox_for_rides(ride_ids):
    """
    Calculate bounding box for a set of rides.

    Args:
        ride_ids: List of ride UUIDs

    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max) or None if no data
    """
    if not DB_AVAILABLE:
        return None

    try:
        db = SessionLocal()

        ride_id_strs = [str(rid) for rid in ride_ids]

        query = text(
            """
            SELECT
                MIN(latitude) as lat_min,
                MAX(latitude) as lat_max,
                MIN(longitude) as lon_min,
                MAX(longitude) as lon_max
            FROM location
            WHERE ride_id = ANY(CAST(:ride_ids AS uuid[]))
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
        """
        )

        result = db.execute(query, {"ride_ids": ride_id_strs})
        row = result.fetchone()

        db.close()

        if row and row.lat_min is not None:
            return (row.lat_min, row.lat_max, row.lon_min, row.lon_max)
        return None

    except Exception as e:
        print(f"Error getting bbox for rides: {e}")
        return None


def query_jumps_global(lat_min, lat_max, lon_min, lon_max):
    """
    Query jumps_global table for jumps within the given bounds.

    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds

    Returns:
        List of jump dictionaries with takeoff and landing statistics
    """
    if not DB_AVAILABLE:
        return []

    try:
        db = SessionLocal()

        query = text(
            """
            SELECT
                lat_takeoff_avg,
                lng_takeoff_avg,
                lat_takeoff_std,
                lng_takeoff_std,
                lat_landing_avg,
                lng_landing_avg,
                lat_landing_std,
                lng_landing_std,
                distance_avg,
                airtime_avg,
                jump_count
            FROM jumps_global
            WHERE (
                (lat_takeoff_avg BETWEEN :lat_min AND :lat_max
                 AND lng_takeoff_avg BETWEEN :lon_min AND :lon_max)
                OR
                (lat_landing_avg BETWEEN :lat_min AND :lat_max
                 AND lng_landing_avg BETWEEN :lon_min AND :lon_max)
            )
        """
        )

        result = db.execute(
            query,
            {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        )

        jumps = []
        for row in result:
            jumps.append(
                {
                    "lat_takeoff_avg": row.lat_takeoff_avg,
                    "lng_takeoff_avg": row.lng_takeoff_avg,
                    "lat_takeoff_std": row.lat_takeoff_std,
                    "lng_takeoff_std": row.lng_takeoff_std,
                    "lat_landing_avg": row.lat_landing_avg,
                    "lng_landing_avg": row.lng_landing_avg,
                    "lat_landing_std": row.lat_landing_std,
                    "lng_landing_std": row.lng_landing_std,
                    "distance_avg": row.distance_avg,
                    "airtime_avg": row.airtime_avg,
                    "jump_count": row.jump_count,
                }
            )

        db.close()
        return jumps

    except Exception as e:
        print(f"Error querying jumps_global: {e}")
        return []


def load_parquet_locations(file_path, lat_col="lat", lon_col="lon"):
    """
    Load location data from a parquet file.

    Args:
        file_path: Path to the parquet file
        lat_col: Name of the latitude column (default: "lat")
        lon_col: Name of the longitude column (default: "lon")

    Returns:
        List of [latitude, longitude] coordinates
    """
    try:
        df = pd.read_parquet(file_path)

        if lat_col not in df.columns:
            print(f"Error: Column '{lat_col}' not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return []

        if lon_col not in df.columns:
            print(f"Error: Column '{lon_col}' not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return []

        # Filter out null values and convert to list of [lat, lon]
        valid_mask = df[lat_col].notna() & df[lon_col].notna()
        positions = df.loc[valid_mask, [lat_col, lon_col]].values.tolist()

        return positions

    except Exception as e:
        print(f"Error loading parquet file {file_path}: {e}")
        return []


def get_ride_display_name(ride):
    """Generate a display name for a ride."""
    if ride["title"]:
        return ride["title"]
    elif ride["start_time"]:
        return ride["start_time"].strftime("%Y-%m-%d %H:%M")
    else:
        return str(ride["id"])[:8]


# Color palette for rides
RIDE_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MtbViz Map - Full-screen map visualization of mountain bike rides"
    )
    parser.add_argument(
        "--ride-id",
        "-r",
        type=str,
        nargs="+",
        default=None,
        help="One or more ride IDs (UUIDs) to display",
    )
    parser.add_argument(
        "--bbox",
        "-b",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=None,
        help="Bounding box to select rides (lat_min lat_max lon_min lon_max)",
    )
    parser.add_argument(
        "--incl-similar",
        "-s",
        action="store_true",
        help="Include all rides with points in the same bbox as the selected rides",
    )
    parser.add_argument(
        "--parquet",
        "-f",
        type=str,
        nargs="+",
        default=None,
        help="One or more parquet files containing location data",
    )
    parser.add_argument(
        "--lat-col",
        type=str,
        default="lat",
        help="Name of the latitude column in parquet files (default: lat)",
    )
    parser.add_argument(
        "--lon-col",
        type=str,
        default="lon",
        help="Name of the longitude column in parquet files (default: lon)",
    )
    parser.add_argument(
        "--names",
        "-n",
        type=str,
        nargs="+",
        default=None,
        help="Display names for parquet files (in same order as --parquet)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8051,
        help="Port to run the Dash app on (default: 8051)",
    )
    return parser.parse_args()


def main():
    """Main function to run the Dash app."""
    args = parse_args()

    # Check if we have any data source
    has_db_source = args.ride_id or args.bbox
    has_parquet_source = args.parquet

    if not has_db_source and not has_parquet_source:
        print("Error: Must specify --ride-id, --bbox, or --parquet")
        return

    if has_db_source and not DB_AVAILABLE:
        print("Error: Database connection required for --ride-id or --bbox.")
        print("Set DATABASE_URL environment variable or use --parquet instead.")
        return

    # Collect rides from database
    rides = []

    if args.ride_id:
        # Parse ride IDs as UUIDs
        ride_ids = [UUID(rid) for rid in args.ride_id]
        rides = query_rides_by_ids(ride_ids)

        if args.incl_similar and rides:
            # Get bbox for the selected rides and find similar rides
            bbox = get_bbox_for_rides([r["id"] for r in rides])
            if bbox:
                lat_min, lat_max, lon_min, lon_max = bbox
                # Add some padding
                lat_padding = 0.01 * (lat_max - lat_min) if lat_max != lat_min else 0.001
                lon_padding = 0.01 * (lon_max - lon_min) if lon_max != lon_min else 0.001
                similar_rides = query_rides_by_bbox(
                    lat_min - lat_padding,
                    lat_max + lat_padding,
                    lon_min - lon_padding,
                    lon_max + lon_padding,
                )
                # Merge rides, avoiding duplicates
                existing_ids = {r["id"] for r in rides}
                for sr in similar_rides:
                    if sr["id"] not in existing_ids:
                        rides.append(sr)
                        existing_ids.add(sr["id"])

    elif args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        rides = query_rides_by_bbox(lat_min, lat_max, lon_min, lon_max)

    # Load location data for each ride from database
    ride_data = []
    all_lats = []
    all_lons = []
    color_idx = 0

    if rides:
        print(f"Loading {len(rides)} ride(s) from database...")

    for ride in rides:
        positions = query_locations_for_ride(ride["id"])
        if positions:
            ride_name = get_ride_display_name(ride)
            ride_data.append(
                {
                    "id": str(ride["id"]),
                    "name": ride_name,
                    "positions": positions,
                    "color": RIDE_COLORS[color_idx % len(RIDE_COLORS)],
                }
            )
            color_idx += 1
            for pos in positions:
                all_lats.append(pos[0])
                all_lons.append(pos[1])
            print(f"  - {ride_name}: {len(positions)} points")

    # Load location data from parquet files
    if args.parquet:
        print(f"Loading {len(args.parquet)} parquet file(s)...")
        for i, parquet_path in enumerate(args.parquet):
            positions = load_parquet_locations(parquet_path, args.lat_col, args.lon_col)
            if positions:
                # Use provided name or derive from filename
                if args.names and i < len(args.names):
                    ride_name = args.names[i]
                else:
                    ride_name = os.path.splitext(os.path.basename(parquet_path))[0]

                ride_data.append(
                    {
                        "id": f"parquet-{i}",
                        "name": ride_name,
                        "positions": positions,
                        "color": RIDE_COLORS[color_idx % len(RIDE_COLORS)],
                    }
                )
                color_idx += 1
                for pos in positions:
                    all_lats.append(pos[0])
                    all_lons.append(pos[1])
                print(f"  - {ride_name}: {len(positions)} points (from parquet)")

    if not ride_data:
        print("No location data found from any source")
        return

    # Calculate map bounds
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    lat_padding = 0.05 * (lat_max - lat_min) if lat_max != lat_min else 0.01
    lon_padding = 0.05 * (lon_max - lon_min) if lon_max != lon_min else 0.01

    bounds = [
        [lat_min - lat_padding, lon_min - lon_padding],
        [lat_max + lat_padding, lon_max + lon_padding],
    ]

    # Query jump clusters
    jumps_data = query_jumps_global(lat_min, lat_max, lon_min, lon_max)
    print(f"Found {len(jumps_data)} jump clusters in map bounds")

    # Create jump overlays
    jump_overlays = []
    for idx, jump in enumerate(jumps_data):
        # Takeoff ellipse (light blue)
        takeoff_points = create_ellipse_points(
            jump["lat_takeoff_avg"],
            jump["lng_takeoff_avg"],
            jump["lat_takeoff_std"],
            jump["lng_takeoff_std"],
        )
        jump_overlays.append(
            dl.Polygon(
                id={"type": "jump-takeoff", "index": idx},
                positions=takeoff_points,
                color="#4DA6FF",
                fillColor="#4DA6FF",
                fillOpacity=0.3,
                weight=2,
            )
        )

        # Landing ellipse (light red)
        landing_points = create_ellipse_points(
            jump["lat_landing_avg"],
            jump["lng_landing_avg"],
            jump["lat_landing_std"],
            jump["lng_landing_std"],
        )
        jump_overlays.append(
            dl.Polygon(
                id={"type": "jump-landing", "index": idx},
                positions=landing_points,
                color="#FF6B6B",
                fillColor="#FF6B6B",
                fillOpacity=0.3,
                weight=2,
            )
        )

    # Store data for callbacks
    jump_overlay_positions = {}
    for idx, jump in enumerate(jumps_data):
        takeoff_key = f"jump-takeoff-{idx}"
        jump_overlay_positions[takeoff_key] = create_ellipse_points(
            jump["lat_takeoff_avg"],
            jump["lng_takeoff_avg"],
            jump["lat_takeoff_std"],
            jump["lng_takeoff_std"],
        )
        landing_key = f"jump-landing-{idx}"
        jump_overlay_positions[landing_key] = create_ellipse_points(
            jump["lat_landing_avg"],
            jump["lng_landing_avg"],
            jump["lat_landing_std"],
            jump["lng_landing_std"],
        )

    # Create the Dash app
    app = Dash(__name__)

    # Build ride toggle controls
    ride_toggles = []
    for rd in ride_data:
        ride_toggles.append(
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "12px",
                            "height": "12px",
                            "backgroundColor": rd["color"],
                            "display": "inline-block",
                            "marginRight": "5px",
                            "borderRadius": "2px",
                        }
                    ),
                    dcc.Checklist(
                        id={"type": "ride-toggle", "index": rd["id"]},
                        options=[{"label": rd["name"], "value": "show"}],
                        value=["show"],
                        style={"display": "inline-block"},
                        inputStyle={"marginRight": "5px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginRight": "15px",
                },
            )
        )

    # Build polylines for each ride
    route_polylines = [
        dl.Polyline(
            id={"type": "route-line", "index": rd["id"]},
            positions=rd["positions"],
            color=rd["color"],
            weight=3,
        )
        for rd in ride_data
    ]

    app.layout = html.Div(
        [
            # Control bar at top
            html.Div(
                [
                    # Map layer selector
                    html.Div(
                        [
                            html.Label(
                                "Map Layer:",
                                style={"marginRight": "5px", "fontWeight": "bold"},
                            ),
                            dcc.Dropdown(
                                id="layer-selector",
                                options=[
                                    {"label": "Standard", "value": "m"},
                                    {"label": "Satellite", "value": "y"},
                                    {"label": "Terrain", "value": "p"},
                                ],
                                value="p",  # Default to terrain
                                clearable=False,
                                style={"width": "120px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginRight": "20px"},
                    ),
                    # Jumps toggle
                    html.Div(
                        [
                            html.Label(
                                "Show Jumps:",
                                style={"marginRight": "5px", "fontWeight": "bold"},
                            ),
                            dcc.Checklist(
                                id="jumps-toggle",
                                options=[{"label": "", "value": "show"}],
                                value=["show"] if jumps_data else [],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginRight": "20px"},
                    ),
                    # Separator
                    html.Div(
                        style={
                            "borderLeft": "1px solid #ccc",
                            "height": "30px",
                            "marginRight": "20px",
                        }
                    ),
                    # Ride toggles
                    html.Label(
                        "Rides:",
                        style={"marginRight": "10px", "fontWeight": "bold"},
                    ),
                    *ride_toggles,
                ],
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "right": "0",
                    "zIndex": "1000",
                    "backgroundColor": "white",
                    "padding": "10px 20px",
                    "display": "flex",
                    "alignItems": "center",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "flexWrap": "wrap",
                },
            ),
            # Full-screen map
            html.Div(
                [
                    dl.Map(
                        id="map",
                        bounds=bounds,
                        style={"width": "100%", "height": "100%"},
                        children=[
                            dl.TileLayer(
                                id="base-layer",
                                url=f"https://mt1.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}",
                                attribution="Google Maps",
                                maxZoom=22,
                            ),
                            # Route polylines
                            *route_polylines,
                            # Jump overlays
                            *jump_overlays,
                            dl.ScaleControl(position="bottomleft"),
                        ],
                    ),
                ],
                style={
                    "position": "fixed",
                    "top": "50px",  # Height of control bar
                    "left": "0",
                    "right": "0",
                    "bottom": "0",
                },
            ),
            # Store ride data for callbacks
            dcc.Store(id="ride-data-store", data=[{"id": rd["id"], "positions": rd["positions"]} for rd in ride_data]),
            dcc.Store(id="jump-positions-store", data=jump_overlay_positions),
        ],
        style={"margin": "0", "padding": "0", "overflow": "hidden"},
    )

    # Callback to update map layer
    @app.callback(Output("base-layer", "url"), Input("layer-selector", "value"))
    def update_map_layer(selected_layer):
        return f"https://mt1.google.com/vt/lyrs={selected_layer}&x={{x}}&y={{y}}&z={{z}}&apikey={GOOGLE_MAPS_API_KEY}"

    # Callback to toggle ride visibility
    @app.callback(
        Output({"type": "route-line", "index": ALL}, "positions"),
        Input({"type": "ride-toggle", "index": ALL}, "value"),
        State("ride-data-store", "data"),
    )
    def toggle_rides(toggle_values, ride_data_store):
        # Build a mapping from ride_id to positions
        ride_positions = {rd["id"]: rd["positions"] for rd in ride_data_store}

        # Get the IDs in the same order as the toggles
        ctx = callback_context
        outputs = []

        for i, triggered in enumerate(ctx.outputs_list):
            ride_id = triggered["id"]["index"]
            if toggle_values[i] and "show" in toggle_values[i]:
                outputs.append(ride_positions.get(ride_id, []))
            else:
                outputs.append([])

        return outputs

    # Callback to toggle jump visibility
    @app.callback(
        Output({"type": "jump-takeoff", "index": ALL}, "positions"),
        Output({"type": "jump-landing", "index": ALL}, "positions"),
        Input("jumps-toggle", "value"),
        State("jump-positions-store", "data"),
    )
    def toggle_jumps(toggle_value, jump_positions):
        show = "show" in toggle_value if toggle_value else False

        takeoff_outputs = []
        landing_outputs = []

        # Count jumps based on keys in jump_positions
        num_jumps = len([k for k in jump_positions.keys() if k.startswith("jump-takeoff-")])

        for idx in range(num_jumps):
            takeoff_key = f"jump-takeoff-{idx}"
            landing_key = f"jump-landing-{idx}"

            if show:
                takeoff_outputs.append(jump_positions.get(takeoff_key, []))
                landing_outputs.append(jump_positions.get(landing_key, []))
            else:
                takeoff_outputs.append([])
                landing_outputs.append([])

        return takeoff_outputs, landing_outputs

    # Run the app
    app.run_server(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    main()
