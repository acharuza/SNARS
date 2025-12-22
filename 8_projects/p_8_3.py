import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import folium
import numpy as np

# transport types mapping and colors
TRANSPORT_COLOR_MAP = {
    0: {"name": "Tram", "color": "red"},
    2: {"name": "Rail", "color": "orange"},
    3: {"name": "Bus", "color": "blue"},
    "multi": {
        "name": "Multiple Types",  # hubs served by 2 or more modes
        "color": "purple",
    },
    "other": {"name": "Other/Unused", "color": "gray"},
}
PRIMARY_MODES = [0, 2, 3]


def change_txt_to_csv(input_txt, output_csv):
    """Converts a TXT file (assumed to be CSV format) to a CSV file."""
    df = pd.read_csv(input_txt)
    df.to_csv(output_csv, index=False)


def save_files_as_csv(root_dir):
    """Recursively converts all TXT files in a directory to CSV and removes originals."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                input_txt = os.path.join(dirpath, filename)
                output_csv = os.path.join(dirpath, filename.replace(".txt", ".csv"))
                change_txt_to_csv(input_txt, output_csv)
                os.remove(input_txt)


def load_necessary_files(root_dir):
    """
    Loads necessary GTFS files.
    """
    try:
        # load routes, including route_short_name to count lines
        routes_df = pd.read_csv(
            os.path.join(root_dir, "routes.csv"),
            usecols=["route_id", "route_type", "route_short_name"],
        )
        # load trips to link route_id to trip_id
        trips_df = pd.read_csv(
            os.path.join(root_dir, "trips.csv"), usecols=["trip_id", "route_id"]
        )
        # load stop_times to get edges
        stop_times_df = pd.read_csv(
            os.path.join(root_dir, "stop_times.csv"),
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
        # load stops for coordinates and names
        stops_df = pd.read_csv(os.path.join(root_dir, "stops.csv"))

        return routes_df, trips_df, stop_times_df, stops_df
    except FileNotFoundError as e:
        print(
            f"Error loading file: {e}. Please check the 'gtfs' directory and file names."
        )
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        exit()


def classify_and_color_stops(routes_df, trips_df, stop_times_df, stops_df):
    """
    Determines the primary transport mode(s) serving each stop (node) and assigns a color/label.
    """
    # link trip_id to route_type
    # stop times have trip_id, routes have route_type, so we join trips and routes first
    trip_types = trips_df.merge(routes_df, on="route_id", how="left")

    # link stop_id to route_type
    # now join with stop_times to get the route_type for every stop instance
    stop_types = (
        stop_times_df[["trip_id", "stop_id"]]
        .merge(trip_types[["trip_id", "route_type"]], on="trip_id", how="left")
        .dropna(subset=["route_type"])
    )

    # aggregate modes per stop
    # group by stop_id to see all unique transport modes serving that stop
    stop_type_summary = (
        stop_types.groupby("stop_id")["route_type"].unique().reset_index()
    )
    stop_type_summary.columns = ["stop_id", "transport_types"]

    # assign final color/label
    def get_color_and_label(type_list):
        is_tram = 0 in type_list
        is_rail = 2 in type_list
        is_bus = 3 in type_list
        mode_count = is_tram + is_rail + is_bus

        if mode_count >= 2:
            return (
                TRANSPORT_COLOR_MAP["multi"]["color"],
                TRANSPORT_COLOR_MAP["multi"]["name"],
            )
        elif is_tram:
            return TRANSPORT_COLOR_MAP[0]["color"], TRANSPORT_COLOR_MAP[0]["name"]
        elif is_rail:
            return TRANSPORT_COLOR_MAP[2]["color"], TRANSPORT_COLOR_MAP[2]["name"]
        elif is_bus:
            return TRANSPORT_COLOR_MAP[3]["color"], TRANSPORT_COLOR_MAP[3]["name"]
        else:
            return (
                TRANSPORT_COLOR_MAP["other"]["color"],
                TRANSPORT_COLOR_MAP["other"]["name"],
            )

    stop_type_summary[["color", "type_label"]] = stop_type_summary[
        "transport_types"
    ].apply(lambda x: pd.Series(get_color_and_label(x)))

    # merge back to stops_df and clean up
    final_stops_df = stops_df.merge(
        stop_type_summary[["stop_id", "color", "type_label"]], on="stop_id", how="left"
    )

    # fill stops not found with 'other'
    final_stops_df["color"] = final_stops_df["color"].fillna(
        TRANSPORT_COLOR_MAP["other"]["color"]
    )
    final_stops_df["type_label"] = final_stops_df["type_label"].fillna(
        TRANSPORT_COLOR_MAP["other"]["name"]
    )

    # filter to only stops that are actually used in the network
    used_stops_ids = final_stops_df[
        final_stops_df["type_label"] != TRANSPORT_COLOR_MAP["other"]["name"]
    ]["stop_id"].unique()
    final_stops_df = final_stops_df[
        final_stops_df["stop_id"].isin(used_stops_ids)
    ].copy()

    return final_stops_df


def generate_and_weight_edges(routes_df, trips_df, stop_times_df, stops_df):
    """
    Generates directed edges (stop-to-stop links) and weights them by the
    number of unique line names (route_short_name) using that segment.
    """
    # create sequential edges
    stop_times_df = stop_times_df.sort_values(["trip_id", "stop_sequence"])
    # shift stop_id to link the current stop to the next stop in the sequence
    stop_times_df["next_stop_id"] = stop_times_df.groupby("trip_id")["stop_id"].shift(
        -1
    )
    edges_df = stop_times_df.dropna(subset=["next_stop_id"])

    # link edges to line names and types
    trip_routes = routes_df.merge(trips_df, on="route_id", how="left")
    edges_df = edges_df.merge(
        trip_routes[["trip_id", "route_id", "route_type", "route_short_name"]],
        on="trip_id",
        how="left",
    )

    edges_df["route_type"] = edges_df["route_type"].astype("Int64", errors="ignore")
    edges_df = edges_df[edges_df["route_type"].isin(PRIMARY_MODES)]

    # count unique line names for weighting edges
    edge_weights = (
        edges_df.groupby(["stop_id", "next_stop_id", "route_type"])
        .agg(frequency=("route_short_name", "nunique"))
        .reset_index()
    )
    edge_weights.columns = ["source", "target", "route_type", "frequency"]

    # normalize frequencies to opacity and weight
    def normalize_frequency(df):
        min_freq = df["frequency"].min()
        max_freq = df["frequency"].max()

        # scale opacity between 0.1 (low frequency) and 1.0 (high frequency)
        if max_freq == min_freq:
            df["opacity"] = 0.5
        else:
            df["opacity"] = 0.1 + (1.0 - 0.1) * (df["frequency"] - min_freq) / (
                max_freq - min_freq
            )
        # scale weight between 0.5 and 4.0
        df["weight"] = 0.5 + 3.5 * df["opacity"]
        return df

    edge_weights = edge_weights.groupby("route_type", group_keys=False).apply(
        normalize_frequency
    )

    # join coordinates for drawing
    stops_coords = stops_df[["stop_id", "stop_lat", "stop_lon"]].set_index("stop_id")
    edge_weights = edge_weights.join(
        stops_coords.rename(
            columns={"stop_lat": "source_lat", "stop_lon": "source_lon"}
        ),
        on="source",
    )
    edge_weights = edge_weights.join(
        stops_coords.rename(
            columns={"stop_lat": "target_lat", "stop_lon": "target_lon"}
        ),
        on="target",
    )

    # drop edges if coordinates couldn't be found
    edge_weights = edge_weights.dropna(subset=["source_lat", "target_lat"])

    return edge_weights


def draw_network(
    stops_df_colored, edges_df_weighted, output_html="stops_transport_network.html"
):
    """
    Draws the complete transport network onto a Folium map with togglable layers for modes.
    """
    center = [stops_df_colored["stop_lat"].mean(), stops_df_colored["stop_lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    # create a separate Folium FeatureGroup for the lines of each transport mode
    for route_code, group in edges_df_weighted.groupby("route_type"):
        mode_info = TRANSPORT_COLOR_MAP.get(route_code, TRANSPORT_COLOR_MAP["other"])
        layer_name = f"{mode_info['name']} Lines"
        layer_color = mode_info["color"]

        fg_lines = folium.FeatureGroup(name=layer_name, show=True)

        for _, row in group.iterrows():
            locations = [
                [row["source_lat"], row["source_lon"]],
                [row["target_lat"], row["target_lon"]],
            ]

            folium.PolyLine(
                locations=locations,
                color=layer_color,
                weight=float(row["weight"]),
                opacity=float(row["opacity"]),
                tooltip=f"Line Count: {int(row['frequency'])}",  # show line count on hover
            ).add_to(fg_lines)

        fg_lines.add_to(m)

    # create a separate Folium FeatureGroup for the stops of each classified type
    for label, group in stops_df_colored.groupby("type_label"):
        # show all layers default
        show_layer = label != TRANSPORT_COLOR_MAP["other"]["name"]

        fg_stops = folium.FeatureGroup(name=f"{label} Stops", show=show_layer)
        group_color = group["color"].iloc[0]

        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row["stop_lat"], row["stop_lon"]],
                radius=3,
                color=group_color,
                fill=True,
                fill_opacity=0.8,
                weight=1,
                popup=f"{row['stop_name']} ({row['type_label']})",  # show stop name and type on click
            ).add_to(fg_stops)

        fg_stops.add_to(m)

    # add layer control element to the map for toggling layers
    folium.LayerControl().add_to(m)

    m.save(output_html)
    print(f"Saved interactive map with layered, weighted network to {output_html}")


def build_and_save_networkx_graphs(stops_df, edges_df, root_dir):
    """
    Builds the full network graph and individual layer graphs, and saves them to disk.
    """
    G_full = nx.DiGraph()

    for _, row in stops_df.iterrows():
        G_full.add_node(
            row["stop_id"],
            name=row["stop_name"],
            lat=row["stop_lat"],
            lon=row["stop_lon"],
            type_label=row["type_label"],
            color=row["color"],
        )

    for _, row in edges_df.iterrows():
        G_full.add_edge(
            int(row["source"]),
            int(row["target"]),
            route_type=int(row["route_type"]),
            weight=int(row["frequency"]),
        )

    graphs_to_save = {"full_network": G_full}

    for route_code in PRIMARY_MODES:
        G_layer = G_full.copy()

        edges_to_keep = [
            (u, v, d)
            for u, v, d in G_layer.edges(data=True)
            if d.get("route_type") == route_code
        ]

        all_edges = set(G_layer.edges())
        layer_edges = set((u, v) for u, v, d in edges_to_keep)
        edges_to_remove = all_edges - layer_edges
        G_layer.remove_edges_from(edges_to_remove)

        G_layer.remove_nodes_from(list(nx.isolates(G_layer)))

        mode_name = TRANSPORT_COLOR_MAP[route_code]["name"]
        graphs_to_save[f"{mode_name.lower()}_layer"] = G_layer

    print("\n--- Saving NetworkX Graphs ---")
    os.makedirs(root_dir, exist_ok=True)

    for name, G in graphs_to_save.items():
        filename = os.path.join(root_dir, f"{name}.graphml")
        nx.write_graphml(G, filename)
        print(
            f"Saved {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to: {filename}"
        )

    return graphs_to_save


if __name__ == "__main__":
    routes, trips, stop_times, stops = load_necessary_files("gtfs")

    stops_colored = classify_and_color_stops(routes, trips, stop_times, stops)
    edges_weighted = generate_and_weight_edges(routes, trips, stop_times, stops)
    draw_network(stops_colored, edges_weighted)

    build_and_save_networkx_graphs(stops_colored, edges_weighted, "networkx_graphs")
