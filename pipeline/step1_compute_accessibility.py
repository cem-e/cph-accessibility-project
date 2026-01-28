import json
import math
import logging
import os
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import osmnx
import pandas as pd
import pandana as pdna
import rasterio
from pyproj import Transformer

logging.getLogger("pandana").setLevel(logging.ERROR)

# ===================== CONFIGURATION =====================
# All paths are relative to this file so the script works from any working directory.
script_dir = Path(__file__).parent

# Areas to download street networks for.
place_queries = [
    "Copenhagen Municipality, Denmark",
    "Frederiksberg Municipality, Denmark",
]

# Walking assumptions and search limits
walk_speed_meters_per_minute = 80  # distance / 80 m per min ≈ 5 km/h
max_search_minutes = 9999  # keep searches effectively unlimited
max_search_distance_m = max_search_minutes * walk_speed_meters_per_minute

# Population lookup settings
amenity_population_threshold = 22.0  # minimum residents before an amenity is considered “served”
population_raster_path = script_dir / "population_denmark_2024_100m.tif"
node_population_threshold = amenity_population_threshold

# Amenity categories used throughout the pipeline and the accessibility map
pois = {
    "cafes_restaurants": ["cafe", "restaurant", "fast_food", "bar", "pub"],
    "supermarkets": ["supermarket", "convenience"],
    "education": ["school", "college", "university", "kindergarten"],
    "health": ["pharmacy", "clinic", "doctors", "hospital", "dentist"],
    "parks": ["park", "garden"],
    "playgrounds": ["playground"],
    "pt_stops": ["stop_position", "platform"],
    "culture": ["museum", "cinema", "theatre", "arts_centre", "gallery"],
}
excluded_poi_names = {
    "søfortet trekroner nordre fløj",
    "søfortet trekroner sønder fløj",
    "flakfortet minimarked",
    "restaurant flakfortet på øen i øresund",
}

# Helper functions that clean and standardise the raw OSM POIs.
def find_pois_matching_tags(
    raw_pois: pd.DataFrame, poi_tags, osm_tag_columns
) -> pd.Series:
    """Return a True/False Series for rows that match any of the given OSM tags."""
    matches_any_tag = pd.Series(False, index=raw_pois.index)
    for col in osm_tag_columns:
        if col not in raw_pois.columns:
            continue
        col_matches = raw_pois[col].isin(poi_tags)
        matches_any_tag = matches_any_tag | col_matches
    return matches_any_tag


def normalize_poi_geometry(filtered_pois: pd.DataFrame) -> Iterable[Dict]:
    """Convert geometries to representative point records with x/y coordinates."""
    normalized_pois = []
    for index, row in filtered_pois.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        point = None
        # Shapely attribute that returns the geometry type as a string
        geom_type = geom.geom_type
        if geom_type == "Point":
            point = geom
        elif geom_type in ["Polygon", "MultiPolygon"]:
            point = geom.representative_point()
        elif geom_type in ["LineString", "MultiLineString"]:
            point = geom.interpolate(0.5, normalized=True)

        if point is None:
            continue

        record = row.drop(labels=["geometry"]).to_dict()
        record["x"] = float(point.x)
        record["y"] = float(point.y)
        normalized_pois.append(record)

    return normalized_pois


def ensure_poi_tag_columns(processed_pois: pd.DataFrame, osm_tag_columns) -> None:
    """Ensure every expected tag column exists in the POI table."""
    for col in osm_tag_columns:
        if col not in processed_pois.columns:
            processed_pois[col] = pd.NA


def resolve_poi_display_names(processed_pois: pd.DataFrame, osm_tag_columns) -> Iterable[str]:
    """Generate display names for each POI."""
    resolved = []
    for index, row in processed_pois.iterrows():
        chosen = None
        for col in ["name"] + list(osm_tag_columns):
            value = row.get(col)
            has_value = pd.notna(value)
            has_content = str(value).strip() != ""
            if has_value and has_content:
                chosen = str(value)
                break
        if chosen is None:
            chosen = "(unnamed)"
        resolved.append(chosen)
    return resolved


def filter_normalize_label_pois_for_category(raw_pois, poi_tags):
    """Pick amenities that match the given tags, normalise geometry, and assign display names."""
    osm_tag_columns = ["amenity", "leisure", "shop", "public_transport", "tourism"]
    existing_tag_columns = [col for col in osm_tag_columns if col in raw_pois.columns]

    match_mask = find_pois_matching_tags(raw_pois, poi_tags, existing_tag_columns)
    filtered_pois = raw_pois[match_mask]

    normalized_pois = normalize_poi_geometry(filtered_pois)

    processed_pois = pd.DataFrame(normalized_pois)
    ensure_poi_tag_columns(processed_pois, osm_tag_columns)
    processed_pois["display_name"] = resolve_poi_display_names(
        processed_pois, osm_tag_columns
    )
    if not processed_pois.empty:
        name_norm = processed_pois.get("name", pd.Series([], dtype=str)).fillna("")
        display_norm = processed_pois["display_name"].fillna("")
        name_norm = name_norm.astype(str).str.strip().str.lower()
        display_norm = display_norm.astype(str).str.strip().str.lower()
        exclude_mask = name_norm.isin(excluded_poi_names) | display_norm.isin(
            excluded_poi_names
        )
        processed_pois = processed_pois.loc[~exclude_mask].reset_index(drop=True)

    processed_pois.reset_index(drop=True, inplace=True)
    return processed_pois

def normalise_for_json(value):
    """Convert pandas/numpy scalars to plain Python types for JSON output."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

# ===================== POPULATION PROCESSING =====================
def make_population_lookup(dataset):
    """
    Return a cached lookup that fetches the raw raster cell value at the given lon/lat. So basically, I made it so the cell value is treated
    as the local population for that point.
    """
    dataset_crs = dataset.crs
    # Convert lon/lat into the raster's coordinate system so I can read the population value
    to_dataset_crs = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
    nodata = dataset.nodata

    @lru_cache(maxsize=8192)
    def lookup(lon: float, lat: float) -> float:
        lon_f = float(lon)
        lat_f = float(lat)
        x_ds, y_ds = to_dataset_crs.transform(lon_f, lat_f)
        # sample returns an iterator of arrays - take the first value
        val = list(dataset.sample([(x_ds, y_ds)]))[0][0]
        if nodata is not None and val == nodata:
            return math.nan
        return float(val)

    return lookup

def annotate_amenity_population_and_isolation(
    df,
    lookup,
    population_threshold,
):
    """Attach population counts and isolation flags to the amenity table."""
    # I'm making sure to work on a copy so I do not mutate the caller's DataFrame.
    df = df.copy()

    # Look up the raster population for each row's coordinates.
    population_values = []
    for index, row in df.iterrows():
        lon = row["x"]
        lat = row["y"]
        population_value = lookup(lon, lat)
        population_values.append(population_value)

    # Store the population values in a Series with the same index as the input.
    pop_series = pd.Series(population_values, index=df.index, dtype="float64")
    df["amenity_pop_at_location"] = pop_series
    # Keep the threshold on every row for downstream reference.
    df["population_isolation_threshold"] = population_threshold

    # Mark amenities below the population threshold as isolated.
    df["is_isolated"] = pop_series < population_threshold

    return df

def annotate_node_population(
    df,
    lookup,
    threshold,
):
    """Attach population counts + a node-level isolation flag based on local residents."""
    df = df.copy()
    if df.empty:
        df["node_pop_at_location"] = pd.Series(dtype="float64")
        df["node_population_threshold"] = pd.Series(dtype="float64")
        df["is_isolated_population"] = pd.Series(dtype="boolean")
        return df

    values = []
    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]
        values.append(lookup(x, y))

    pop_series = pd.Series(values, index=df.index, dtype="float64")
    df["node_pop_at_location"] = pop_series
    df["node_population_threshold"] = threshold

    df["is_isolated_population"] = pop_series < threshold
    return df

def build_feature_collection(
    tables: Dict[str, pd.DataFrame],
    threshold: float,
) -> Dict:
    """Convert amenity tables into a GeoJSON FeatureCollection for the web map."""
    feature_store: Dict[Tuple, Dict] = {}

    for category, table in tables.items():
        if table is None or table.empty:
            continue
        enriched = table.copy()
        enriched["source_category"] = category
        for record in enriched.to_dict(orient="records"):
            lon = record.get("x")
            lat = record.get("y")
            if lon is None or lat is None:
                continue


            key = (
                record.get("@id")
                or record.get("osmid")
                or (round(float(lon), 7), round(float(lat), 7))
            )

            properties = {
                k: normalise_for_json(v)
                for k, v in record.items()
                if k not in {"x", "y"}
            }
            if "population_isolation_threshold" not in properties:
                properties["population_isolation_threshold"] = threshold

            existing = feature_store.get(key)
            if existing is None:
                properties["source_categories"] = [category]
                feature_store[key] = {
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)],
                    },
                    "properties": properties,
                }
                continue

            # Duplicate key found; keep the first occurrence and skip adding categories/properties.
            continue
    features = []
    for feature in feature_store.values():
        props = feature["properties"]
        cats = props.get("source_categories")
        if isinstance(cats, list):
            props["source_categories"] = sorted(set(cats))
        features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": props,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "population_isolation_threshold": threshold,
        },
    }

def compute_access_for_category(
    category,
    processed_pois,
    iso_suffix,
):
    """
    Register POIs with Pandana and attach travel metrics to the node table.
    This helper is reused for both the individual amenity categories and the all-in-one blend.
    """
    # Column names for the base (non-iso) values.
    base_min_col = f"min_to_{category}"
    base_name_col = f"nearest_poi_{category}_name"
    base_lat_col = f"nearest_poi_{category}_lat"
    base_lng_col = f"nearest_poi_{category}_lng"
    base_dist_col = f"nearest_poi_{category}_dist_m"

    # Isolation mode: reuse the baseline values directly.
    if iso_suffix == "_iso":
        if base_min_col in nodes.columns:
            nodes[f"min_to_{category}{iso_suffix}"] = nodes[base_min_col]
        if base_name_col in nodes.columns:
            nodes[f"nearest_poi_{category}{iso_suffix}_name"] = nodes[base_name_col]
        if base_lat_col in nodes.columns:
            nodes[f"nearest_poi_{category}{iso_suffix}_lat"] = nodes[base_lat_col]
        if base_lng_col in nodes.columns:
            nodes[f"nearest_poi_{category}{iso_suffix}_lng"] = nodes[base_lng_col]
        if base_dist_col in nodes.columns:
            nodes[f"nearest_poi_{category}{iso_suffix}_dist_m"] = nodes[base_dist_col]
        return

    # Register POI coordinates with Pandana so nearest_pois can find them.
    poi_key = f"{category}{iso_suffix}"
    routing_x = processed_pois["x"]
    routing_y = processed_pois["y"]
    nearest_neighbor_count = 2
    net.set_pois(
        poi_key,
        maxdist=max_search_distance_m,
        maxitems=nearest_neighbor_count,
        x_col=routing_x,
        y_col=routing_y,
    )

    # Ask Pandana for the nearest POIs for every node.
    nearest_result = net.nearest_pois(
        distance=max_search_distance_m,
        category=poi_key,
        num_pois=nearest_neighbor_count,
        include_poi_ids=True,
    )
    include_ids = True

    def split_nearest_output(raw, need_ids):
        """Pandana v0.7 returns either (dist, ids) or a DataFrame with distance/id columns."""
        # Distances are the first N columns (named 1..N).
        distance_table = raw.iloc[:, :nearest_neighbor_count]
        poi_id_columns = [
            col for col in raw.columns if isinstance(col, str) and col.startswith("poi")
        ]
        poi_id_table = raw.loc[:, poi_id_columns[:nearest_neighbor_count]]
        return distance_table, poi_id_table

    dist_df, nearest_poi_ids = split_nearest_output(nearest_result, include_ids)

    # Pull the primary (closest) distance for each node.
    if dist_df is not None:
        primary_dist = pd.to_numeric(dist_df.iloc[:, 0], errors="coerce")
    else:
        primary_dist = pd.Series(np.nan, index=nodes.index)
    primary_dist = primary_dist.replace({np.inf: np.nan})

    # Column names for this category + suffix.
    min_col = f"min_to_{category}{iso_suffix}"
    dist_col = f"nearest_poi_{category}{iso_suffix}_dist_m"
    name_col = f"nearest_poi_{category}{iso_suffix}_name"
    lat_col = f"nearest_poi_{category}{iso_suffix}_lat"
    lng_col = f"nearest_poi_{category}{iso_suffix}_lng"

    # Store minutes and distances on the node table.
    nodes[min_col] = primary_dist / walk_speed_meters_per_minute
    nodes[dist_col] = primary_dist
    nodes[name_col] = pd.NA
    nodes[lat_col] = pd.NA
    nodes[lng_col] = pd.NA

    # Extract POI IDs so we can map them to names and coordinates.
    primary_idx = (
        pd.to_numeric(nearest_poi_ids.iloc[:, 0], errors="coerce")
        .round()
        .astype("Int64")
    )

    def map_lookup(idx_series, lookup_series):
        # Turn POI ids into actual values (name/lat/lon) for the nearest POI at each node.
        valid = idx_series.notna() & (idx_series >= 0)
        if pd.api.types.is_numeric_dtype(lookup_series):
            result = pd.Series(np.nan, index=nodes.index, dtype="float64")
        else:
            result = pd.Series(pd.NA, index=nodes.index, dtype="object")
        result.loc[valid] = idx_series.loc[valid].map(lookup_series)
        return result

    # Map POI IDs to names and coordinates.
    if include_ids:
        name_lookup = processed_pois["display_name"]
        lat_lookup = processed_pois["y"]
        lng_lookup = processed_pois["x"]
        primary_names = map_lookup(primary_idx, name_lookup)
        primary_lats = map_lookup(primary_idx, lat_lookup)
        primary_lngs = map_lookup(primary_idx, lng_lookup)

        nodes.loc[:, name_col] = primary_names
        nodes.loc[:, lat_col] = primary_lats
        nodes.loc[:, lng_col] = primary_lngs
    else:
        primary_names = pd.Series(pd.NA, index=nodes.index)
        primary_lats = pd.Series(pd.NA, index=nodes.index)
        primary_lngs = pd.Series(pd.NA, index=nodes.index)

# ===================== MAIN EXECUTION =====================
# Build the walking network and prepare amenity source data.
walking_graph = osmnx.graph_from_place(place_queries, network_type="walk")

nodes_geodf, edges_geodf = osmnx.graph_to_gdfs(walking_graph)
edges_table = edges_geodf.reset_index()

with open(os.devnull, "w") as _devnull, redirect_stdout(_devnull), redirect_stderr(_devnull):
    net = pdna.Network(
        node_x=nodes_geodf["x"],
        node_y=nodes_geodf["y"],
        edge_from=edges_table["u"],
        edge_to=edges_table["v"],
        edge_weights=pd.DataFrame({"weight": edges_table["length"].values}),
    )

nodes = nodes_geodf.copy()

osm_pois = osmnx.features_from_place(
    place_queries,
    tags={
        "amenity": True,
        "leisure": True,
        "shop": True,
        "public_transport": True,
        "tourism": True,
    },
)

population_dataset = None
population_lookup: Optional[Callable[[float, float], float]] = None
# I try to open the population raster so I can look up residents in the cell for each amenity. If it fails I fall back quietly.
print("Running step 1 - compute accessability")
population_dataset = rasterio.open(population_raster_path)
population_lookup = make_population_lookup(population_dataset)

# Compute node-level population so isolation mode can drop low-demand origins.
nodes = annotate_node_population(
    nodes,
    population_lookup,
    node_population_threshold,
)
poi_tables_all: Dict[str, pd.DataFrame] = {}
# The `_iso` view now reuses the same amenity set; isolated origins are removed later when we
# aggregate district averages.
for poi_category, poi_tags in pois.items():
    processed_pois = filter_normalize_label_pois_for_category(osm_pois, poi_tags)
    if poi_category == "parks":
        name_mask = processed_pois["name"].notna() & (
            processed_pois["name"].astype(str).str.strip() != ""
        )
        processed_pois = processed_pois[name_mask].reset_index(drop=True)
    annotated = annotate_amenity_population_and_isolation(
        processed_pois,
        population_lookup,
        amenity_population_threshold,
    )
    poi_tables_all[poi_category] = annotated.copy()

    compute_access_for_category(
        poi_category,
        annotated,
        iso_suffix="",
    )

    compute_access_for_category(
        poi_category,
        annotated,
        iso_suffix="_iso",
        # Isolation mode keeps minutes identical to baseline; flags reflect availability of non-isolated POIs.
    )

if population_dataset is not None:
    population_dataset.close()

# Merge all categories together so I can compute an "all POIs" travel time.
combined_tables_all = []
for df in poi_tables_all.values():
    if df is not None and not df.empty:
        combined_tables_all.append(df)
all_poi_table_all = pd.DataFrame()
all_poi_table_all = (
    pd.concat(combined_tables_all, ignore_index=True)
    .drop_duplicates(subset=["x", "y"])
    .reset_index(drop=True)
)
compute_access_for_category(
    "all_pois",
    all_poi_table_all,
    iso_suffix="",
)

# Same idea for the isolation-adjusted view, but now I keep all amenities and later drop nodes
# (population filtering happens in step 2 so ISO stats only use higher-population origins).
compute_access_for_category(
    "all_pois",
    all_poi_table_all,
    iso_suffix="_iso",
)

# all_pois mirrors
if "min_to_all_pois" in nodes.columns:
    nodes["min_to_all_pois_iso"] = nodes["min_to_all_pois"]
    for suffix in ["_name", "_lat", "_lng", "_dist_m"]:
        base_col = f"nearest_poi_all_pois{suffix}"
        iso_col = f"nearest_poi_all_pois_iso{suffix}"
        if base_col in nodes.columns:
            nodes[iso_col] = nodes[base_col]

# ===================== OUTPUT FILES =====================
# Drop nodes where population is unknown (NaN) to avoid empty popups and downstream artifacts.
nodes = nodes[nodes["node_pop_at_location"].notna()].copy()

def write_geojson(feature_collection: Dict, path: Path) -> None:
    """Persist GeoJSON data to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(feature_collection, fp, ensure_ascii=False)

feature_collection = build_feature_collection(
    poi_tables_all,
    amenity_population_threshold,
)
feature_count = len(feature_collection.get("features", []))
if feature_count:
    web_path = script_dir.parent / "web" / "poi_accessibility_output.geojson"
    try:
        write_geojson(feature_collection, web_path)
    except OSError:
        pass

out_csv = script_dir / "node_accessibility_output.csv"

# Ensure we always have a node_id column for downstream exports/joins.
if "node_id" not in nodes.columns:
    if "osmid" in nodes.columns:
        nodes = nodes.rename(columns={"osmid": "node_id"})
    else:
        nodes["node_id"] = nodes.index.astype(int)

# Final node table with every metric I computed above (this feeds step 2).
nodes.to_csv(out_csv, index=False)

# Export a GeoJSON overlay marking population-isolated nodes for the web map.
low_pop_fields = [
    "node_id",
    "x",
    "y",
    "node_pop_at_location",
    "node_population_threshold",
    "is_isolated_population",
]
low_pop_features = []
for index, row in nodes[low_pop_fields].iterrows():
    props = {}
    for field in low_pop_fields:
        if field not in {"x", "y"}:
            props[field] = normalise_for_json(row[field])
    lng = float(row["x"])
    lat = float(row["y"])
    low_pop_features.append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": props,
        }
    )
low_pop_geojson = {"type": "FeatureCollection", "features": low_pop_features}
low_pop_path = script_dir.parent / "web" / "low_population_overlay_output.geojson"
write_geojson(low_pop_geojson, low_pop_path)

# Export a compact JSON snapshot for the web map (minutes + distances per node).
network_json_columns = ["x", "y"]
minute_columns = []
for col in nodes.columns:
    if col.startswith("min_to_"):
        minute_columns.append(col)
minute_columns = sorted(minute_columns)

distance_columns = []
for col in nodes.columns:
    if col.startswith("nearest_poi_") and col.endswith("_dist_m"):
        distance_columns.append(col)
distance_columns = sorted(distance_columns)
network_json_columns.extend(minute_columns)
network_json_columns.extend(distance_columns)

network_table = nodes[network_json_columns]
network_data = []
for row in network_table.itertuples(index=False, name=None):
    row_values = []
    for value in row:
        row_values.append(normalise_for_json(value))
    network_data.append(row_values)
network_payload = {
    "columns": network_json_columns,
    "data": network_data,
}
network_json_path = script_dir.parent / "web" / "node_accessibility_output.json"
try:
    network_json_path.parent.mkdir(parents=True, exist_ok=True)
    with network_json_path.open("w", encoding="utf-8") as fp:
        json.dump(network_payload, fp, ensure_ascii=False, allow_nan=False)
except OSError:
    pass
