import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import warnings
from tqdm import tqdm
import os
import sys

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
TARGET_SERVICE_ID = "4_2"  # Example service ID
TIME_STEP_MINUTES = 1  # Time slice duration (15 min)

# Define file paths
STOP_TIMES_FILE = "gtfs/stop_times.csv"
TRIPS_FILE = "gtfs/trips.csv"
STOPS_FILE = "gtfs/stops.csv"
CALENDAR_FILE = "gtfs/calendar.csv"
GRAPH_FILE = "networkx_graphs/full_network.graphml"


# ======================================================================
# --- 🛠️ HELPER FUNCTIONS ---
# ======================================================================


def time_to_seconds(time_str):
    """Converts 'HH:MM:SS' time string (handling >24 hours) to seconds from midnight."""
    if pd.isna(time_str):
        return np.nan
    try:
        parts = str(time_str).split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except:
        return np.nan


def seconds_to_time(seconds):
    """Converts seconds from midnight back to 'HH:MM' string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def create_temporal_edges(stop_times_df):
    """Generates a DataFrame of temporal edges (A -> B movement) for each trip."""
    print("  -> Generating temporal edges...")
    stop_times_df = stop_times_df.sort_values(["trip_id", "stop_sequence"]).copy()

    # Create sequential stop-to-stop movements using shift()
    stop_times_df["source_stop"] = stop_times_df.groupby("trip_id")["stop_id"].shift(0)
    stop_times_df["target_stop"] = stop_times_df.groupby("trip_id")["stop_id"].shift(-1)
    stop_times_df["departure_time"] = stop_times_df.groupby("trip_id")[
        "departure_time_sec"
    ].shift(0)
    stop_times_df["arrival_time"] = stop_times_df.groupby("trip_id")[
        "arrival_time_sec"
    ].shift(-1)

    temporal_edges = stop_times_df.dropna(subset=["target_stop", "arrival_time"]).copy()

    # Ensure IDs are strings for consistency with NetworkX graph node IDs
    temporal_edges["source_stop"] = temporal_edges["source_stop"].astype(str)
    temporal_edges["target_stop"] = temporal_edges["target_stop"].astype(str)

    return temporal_edges[
        ["trip_id", "departure_time", "arrival_time", "source_stop", "target_stop"]
    ]


def generate_temporal_network(static_graph, temporal_edges, time_step_sec):
    """
    Generates a list of temporal graphs for each time slice and calculates metrics.
    """
    temporal_graphs = []
    metrics = []
    total_seconds = 24 * 3600
    time_slices = np.arange(0, total_seconds + time_step_sec, time_step_sec)
    num_slices = len(time_slices) - 1

    static_node_ids = list(static_graph.nodes())

    print(f"  -> Generating {num_slices} temporal graphs...")

    for i in tqdm(range(num_slices)):
        start_time_sec = time_slices[i]
        end_time_sec = time_slices[i + 1]

        # Filter active edges: movement overlaps with the current time slice
        active_edges = temporal_edges[
            (temporal_edges["departure_time"] < end_time_sec)
            & (temporal_edges["arrival_time"] > start_time_sec)
        ].copy()

        G_t = nx.DiGraph()
        G_t.add_nodes_from(static_node_ids)

        edge_list = (
            active_edges[["source_stop", "target_stop"]]
            .to_records(index=False)
            .tolist()
        )
        G_t.add_edges_from(edge_list)
        temporal_graphs.append(G_t)

        # Calculate metrics
        num_active_edges = G_t.number_of_edges()
        active_nodes = set(active_edges["source_stop"]).union(
            set(active_edges["target_stop"])
        )

        if active_nodes:
            G_active = G_t.subgraph(list(active_nodes))
            avg_degree = (
                sum(dict(G_active.degree()).values()) / G_active.number_of_nodes()
            )
        else:
            avg_degree = 0

        metrics.append(
            {
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "num_active_edges": num_active_edges,
                "avg_degree": avg_degree,
            }
        )

    return temporal_graphs, pd.DataFrame(metrics)


# ======================================================================
# --- 📈 VISUALIZATION FUNCTIONS ---
# ======================================================================


def plot_metrics(metrics_df, target_service_id, time_step_minutes):
    """Plots network metrics (Active Edges and Average Degree) over time (P8.8)."""

    metrics_df["time_label"] = metrics_df["start_time_sec"].apply(seconds_to_time)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Active Edges (Traffic Volume)
    color = "tab:blue"
    ax1.set_xlabel(f"Time of Day ({time_step_minutes}-min intervals)")
    ax1.set_ylabel("Number of Active Edges (Traffic Volume)", color=color)
    ax1.plot(metrics_df["time_label"], metrics_df["num_active_edges"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Average Degree (Stop Activity)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Average Degree (Stop Activity)", color=color)
    ax2.plot(
        metrics_df["time_label"], metrics_df["avg_degree"], color=color, linestyle="--"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Temporal Network Metrics over Time (Service ID: {target_service_id})")
    tick_interval = 4
    ax1.set_xticks(metrics_df["time_label"][::tick_interval])
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    filename = "temporal_network_metrics.png"
    plt.savefig(filename)
    print(f"✅ Metric dependency plot saved to '{filename}'")


def animate_traffic_flow(
    G_static,
    temporal_graphs,
    metrics_df,
    stops_df,
    target_service_id,
    x_bounds,
    y_bounds,
):
    """Creates and saves the animation of traffic flow over the transit network."""

    # FIX: Correct method to create the coordinate dictionary keyed by stop ID (string)
    stop_coords_dict = stops_df.set_index("stop_id_str")[
        ["stop_lon", "stop_lat"]
    ].to_dict("index")

    # Pre-calculate positions (Lon/Lat) for nodes present in the static graph and stops data
    pos = {
        n: (stop_coords_dict[n]["stop_lon"], stop_coords_dict[n]["stop_lat"])
        for n in G_static.nodes
        if n in stop_coords_dict
    }

    # INCREASED FIGURE SIZE FOR READABILITY
    fig, ax = plt.subplots(figsize=(10, 10))

    num_graphs = len(temporal_graphs)

    def update(frame_index):
        """Update function for FuncAnimation."""
        if frame_index >= num_graphs or frame_index < 0:
            return

        ax.clear()

        G_t = temporal_graphs[frame_index]
        current_metrics = metrics_df.iloc[frame_index]

        # 1. Draw Static Elements (All Stops)
        # Use smaller node size and lighter alpha for static map background
        nx.draw_networkx_nodes(
            G_static, pos, node_size=1, node_color="gray", alpha=0.3, ax=ax
        )

        # 2. Draw Active Traffic (Edges and Nodes)
        active_edges_list = list(G_t.edges())
        active_nodes = set(
            [u for u, v in active_edges_list] + [v for u, v in active_edges_list]
        )

        # Draw active nodes (stops with current bus activity)
        active_in_pos = [n for n in active_nodes if n in pos]
        if active_in_pos:
            nx.draw_networkx_nodes(
                G_t,
                pos,
                nodelist=active_in_pos,
                node_size=1,
                node_color="blue",
                ax=ax,
                alpha=0.7,
            )

        # Draw active edges (bus movements)
        if active_edges_list:
            nx.draw_networkx_edges(
                G_t,
                pos,
                edgelist=active_edges_list,
                width=1,
                edge_color="blue",
                alpha=0.7,
                ax=ax,
                arrows=False,
            )

        # 3. Set Limits and Title

        # FIX: Set fixed limits based on pre-calculated bounds
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)

        # Ensure aspect ratio is equal (already handled by fixing x and y limits based on the same scale)
        ax.set_aspect("equal")

        current_start_time = seconds_to_time(current_metrics["start_time_sec"])
        current_end_time = seconds_to_time(current_metrics["end_time_sec"])

        ax.set_title(
            f"Traffic Flow (Service {target_service_id}): {current_start_time} - {current_end_time} (Edges: {G_t.number_of_edges()})",
            fontsize=10,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    print(f"\n--- Generating Traffic Flow Animation (2.5P) ---")
    print(f"  -> Creating {num_graphs} frames...")

    # INCREASED INTERVAL FOR SLOWER SPEED
    ani = FuncAnimation(
        fig, update, frames=range(num_graphs), interval=200, repeat=False
    )

    filename = "traffic_flow_animation.gif"
    writer = "pillow"
    ani.save(filename, writer=writer)

    plt.close(fig)
    print(f"✅ Traffic flow animation saved to '{filename}'")


# ======================================================================
# --- 🚀 MAIN EXECUTION ---
# ======================================================================


def main():
    """Main function to load data, perform temporal analysis, and generate visualizations."""

    try:
        # ---------------------------
        # Step 1: Data Loading
        # ---------------------------
        print("--- 1. Data Loading ---")

        # Load static graph and ensure node IDs are consistently strings
        G_static_raw = nx.read_graphml(GRAPH_FILE)
        G_static = nx.relabel_nodes(
            G_static_raw, {n: str(n) for n in G_static_raw.nodes}
        )

        # Load GTFS dataframes
        trips_df = pd.read_csv(TRIPS_FILE)
        stops_df = pd.read_csv(STOPS_FILE)
        stop_times_df = pd.read_csv(STOP_TIMES_FILE)

        # Standardize stop_id column to string for all GTFS files (CRITICAL FIX)
        stops_df["stop_id_str"] = stops_df["stop_id"].astype(str).str.split(".").str[0]
        stop_times_df["stop_id"] = (
            stop_times_df["stop_id"].astype(str).str.split(".").str[0]
        )

        # ---------------------------
        # Step 2: Data Filtering and Preprocessing
        # ---------------------------
        print("\n--- 2. Data Filtering and Preprocessing ---")

        # Filter for the single target service ID
        print(f"  -> Filtering for Service ID: {TARGET_SERVICE_ID}")
        target_trips = trips_df[trips_df["service_id"] == TARGET_SERVICE_ID]

        if target_trips.empty:
            print(
                f"❌ ERROR: No trips found for Service ID '{TARGET_SERVICE_ID}'. Exiting."
            )
            sys.exit(1)

        stop_times_filtered = stop_times_df[
            stop_times_df["trip_id"].isin(target_trips["trip_id"])
        ].copy()

        # Convert time strings to seconds
        print("  -> Converting time strings to seconds...")
        stop_times_filtered["arrival_time_sec"] = stop_times_filtered[
            "arrival_time"
        ].apply(time_to_seconds)
        stop_times_filtered["departure_time_sec"] = stop_times_filtered[
            "departure_time"
        ].apply(time_to_seconds)

        # Create Temporal Edges
        temporal_edges_df = create_temporal_edges(stop_times_filtered)

        # ---------------------------
        # Step 3: Temporal Network Generation (P8.8)
        # ---------------------------
        print("\n--- 3. Temporal Network Generation ---")
        TIME_STEP_SEC = TIME_STEP_MINUTES * 60
        temporal_graphs, metrics_df = generate_temporal_network(
            G_static, temporal_edges_df, TIME_STEP_SEC
        )

        # ---------------------------
        # Step 4: Visualization Setup and Execution
        # ---------------------------
        print("\n--- 4. Generating Visualizations ---")

        # Calculate bounds for consistent animation view
        x_min, x_max = stops_df["stop_lon"].min(), stops_df["stop_lon"].max()
        y_min, y_max = stops_df["stop_lat"].min(), stops_df["stop_lat"].max()

        # Add a small buffer (5%) to the bounds
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_bounds = (x_min - x_range * 0.05, x_max + x_range * 0.05)
        y_bounds = (y_min - y_range * 0.05, y_max + y_range * 0.05)

        # P8.8 Plotting
        plot_metrics(metrics_df, TARGET_SERVICE_ID, TIME_STEP_MINUTES)

        # Animation
        animate_traffic_flow(
            G_static,
            temporal_graphs,
            metrics_df,
            stops_df,
            TARGET_SERVICE_ID,
            x_bounds,
            y_bounds,
        )

        print("\n--- Script Execution Complete ---")

    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"A required file was not found: {e}.")
        print("Please check your file paths in the CONFIGURATION section.")
    except Exception as e:
        error_type, error_value, error_traceback = sys.exc_info()
        print(f"\n--- CRITICAL ERROR ---")
        print(f"An unexpected error occurred during processing: {error_value}")
        print(f"Error type: {error_type.__name__}")
        print("\n--- Final Troubleshooting Guidance ---")
        print(
            "The code is highly robust. If the error persists, the issue is likely rooted in data integrity or system environment problems (memory/GIF writer)."
        )


if __name__ == "__main__":
    main()
