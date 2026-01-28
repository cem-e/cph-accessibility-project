"""Aggregate node-level minutes per district."""


from pathlib import Path
import warnings
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
logging.getLogger("pandana").setLevel(logging.ERROR)

# ===================== CONFIGURATION =====================
# Work inside the pipeline folder regardless of where the script runs.
pipeline_dir = Path(__file__).parent
nodes_csv = pipeline_dir / "node_accessibility_output.csv"  # node-level minutes produced in step 1
districts_path = pipeline_dir / "district_polygons.geojson"  # district polygons I aggregate over
output_csv = pipeline_dir / "district_accessibility_output.csv"  # final per-district table used by step 3
force_name_field = "kvarternavn"

# ===================== LOAD INPUT DATA =====================
nodes = pd.read_csv(nodes_csv)

# Wrap the raw DataFrame in a GeoDataFrame so spatial joins are painless.
g_nodes = gpd.GeoDataFrame(
    nodes,
    geometry=[Point(xy) for xy in zip(nodes["x"], nodes["y"])],
    crs="EPSG:4326",
)

districts = gpd.read_file(districts_path)

print("Running step 2 - aggregate accessibility")

# Pick the column that holds the district name.
name_field = force_name_field

districts = districts[[name_field, "geometry"]].rename(columns={name_field: "area_name"})

# ===================== SPATIAL JOIN =====================
# Attach each network node to the district polygon it lives in.
joined = gpd.sjoin(
    g_nodes,
    districts,
    how="left",
    predicate="within",
)

access_cols = []
for col in joined.columns:
    if col.startswith("min_to_"):
        access_cols.append(col)

category_base_keys = [
    "min_to_cafes_restaurants",
    "min_to_supermarkets",
    "min_to_education",
    "min_to_health",
    "min_to_parks",
    "min_to_playgrounds",
    "min_to_pt_stops",
    "min_to_culture",
]

category_cols = category_base_keys
category_cols_iso = [f"{col}_iso" for col in category_base_keys]
category_cols_no_parks = [col for col in category_cols if col != "min_to_parks"]
category_cols_iso_no_parks = [col for col in category_cols_iso if col != "min_to_parks_iso"]

pop_isolation_mask = (
    pd.Series(joined["is_isolated_population"].to_numpy(copy=False), dtype="boolean")
    .fillna(False)
    .to_numpy(dtype=bool)
)

# Average minutes across all the core categories so I have a single "overall" accessibility stat per node.
joined["node_avg_min_to_categories"] = (
    joined[category_cols]
    .apply(pd.to_numeric, errors="coerce")
    .mean(axis=1, skipna=True)
)

iso_numeric = joined[category_cols_iso].apply(pd.to_numeric, errors="coerce")
iso_numeric.loc[pop_isolation_mask, :] = np.nan
joined["node_avg_min_to_categories_iso"] = iso_numeric.mean(axis=1, skipna=True)

joined["node_avg_min_to_categories_no_parks"] = (
    joined[category_cols_no_parks]
    .apply(pd.to_numeric, errors="coerce")
    .mean(axis=1, skipna=True)
)

iso_np_numeric = joined[category_cols_iso_no_parks].apply(pd.to_numeric, errors="coerce")
iso_np_numeric.loc[pop_isolation_mask, :] = np.nan
joined["node_avg_min_to_categories_iso_no_parks"] = iso_np_numeric.mean(axis=1, skipna=True)

def filtered_minutes_series(grp, minute_col):
    """Return minutes for this column, applying isolation rules for _iso columns."""
    series = pd.to_numeric(grp[minute_col], errors="coerce")
    is_iso_minutes = (
        minute_col.startswith("min_to_") and minute_col.endswith("_iso")
    )
    if not is_iso_minutes:
        return series[np.isfinite(series)]

    # Build a boolean mask without triggering pandas' future downcast warning.
    pop_mask = (
        pd.Series(grp["is_isolated_population"].to_numpy(copy=False), dtype="boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    finite_mask = np.isfinite(series.to_numpy())
    keep_mask = (~pop_mask) & finite_mask
    return series[keep_mask]

rows = []
for area, grp in joined.groupby("area_name", dropna=False):
    # Build one summary row per district (medians, p90s, averages, node counts).
    rec = {"area_name": area, "n_nodes": int(len(grp))}
    for c in access_cols:
        s = filtered_minutes_series(grp, c)
        rec[f"median_{c}"] = float(s.median())
        rec[f"p90_{c}"] = float(s.quantile(0.9))
        rec[f"mean_{c}"] = float(s.mean())

    avg_col = "node_avg_min_to_categories"
    s = pd.to_numeric(grp[avg_col], errors="coerce").dropna()
    rec["mean_node_avg_min_to_categories"] = float(s.mean())
    rec["median_node_avg_min_to_categories"] = float(s.median())

    avg_col_iso = "node_avg_min_to_categories_iso"
    s_iso = pd.to_numeric(grp[avg_col_iso], errors="coerce").dropna()
    rec["mean_node_avg_min_to_categories_iso"] = float(s_iso.mean())
    rec["median_node_avg_min_to_categories_iso"] = float(s_iso.median())

    avg_col_np = "node_avg_min_to_categories_no_parks"
    s_np = grp[avg_col_np].dropna()
    rec["mean_node_avg_min_to_categories_no_parks"] = float(s_np.mean())
    rec["median_node_avg_min_to_categories_no_parks"] = float(s_np.median())

    avg_col_iso_np = "node_avg_min_to_categories_iso_no_parks"
    s_iso_np = grp[avg_col_iso_np].dropna()
    rec["mean_node_avg_min_to_categories_iso_no_parks"] = float(
        s_iso_np.mean()
    )
    rec["median_node_avg_min_to_categories_iso_no_parks"] = float(
        s_iso_np.median()
    )

    rows.append(rec)

df = pd.DataFrame(rows).sort_values("area_name")

output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)  # step 3 will merge income + geometry onto this table
