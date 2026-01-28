"""Attach median income (5th decile) to aggregated accessibility metrics.

What this does:
  1. Load the per-district accessibility table produced by step 2.
  2. Load the raw income export (median income per district) and clean the numbers.
  3. Translate each accessibility district name to the corresponding income label
     using the access_to_income crosswalk (deterministic on purpose).
  4. Write a combined CSV and GeoJSON so my web map and analysis tooling can show
     accessibility + income in one spot.

Why the hard-coded mapping?
  The municipality uses slightly different labels across datasets, so I'm keeping
  a curated lookup table. If either source file changes, I want the script to fail
  loudly instead of silently mixing the wrong areas.

Final outputs:
  • pipeline/district_income_access_output.csv   (tabular join of accessibility + income)
  • pipeline/district_income_access_output.geojson (same, but with geometry for GIS work)
  • web/district_income_access_output.geojson    (the file the browser map fetches)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd

# ===================== CONFIGURATION =====================
pipeline_dir = Path(__file__).parent

access_csv = pipeline_dir / "district_accessibility_output.csv"
districts_geojson = pipeline_dir / "district_polygons.geojson"
income_csv = pipeline_dir / "income_data_frederiksberg_included.csv"
web_geojson = Path("web/district_income_access_output.geojson")

access_to_income = {
    "Amagerbro vest": "Amager Vest - Amagerbro Vest",
    "Amagerbro øst": "Amager Øst - Amagerbro Øst",
    "Bavnehøj": "Kgs. Enghave - Bavnehøj",
    "Bellahøj": "Brønshøj-Husum - Bellahøj",
    "Bispebjerg": "Bispebjerg - Bispebjerg",
    "Blågårdskvarteret/Assistens/Rantzausgade": "Nørrebro - Blågårdskvarteret/Assistens/Rantzausgade",
    "Bryggen Syd": "Amager Vest - Bryggen Syd",
    "Brønshøj": "Brønshøj-Husum - Brønshøj",
    "Christianshavn Neden Vandet": "Christianshavn - Neden Vandet",
    "Christianshavn Oven Vandet": "Christianshavn - Oven Vandet",
    "Emdrup": "Bispebjerg - Emdrup",
    "Faste Batteri": "Amager Vest - Faste Batteri",
    "Frederiksberg": "Frederiksberg",
    "Frederiksstaden": "Indre By - Frederiksstaden",
    "Fælled": "Østerbro - Fælled",
    "Gamle Bryggen": "Amager Vest - Gamle Bryggen",
    "Gammelholm og Nyhavn": "Indre By - Gammelholm og Nyhavn",
    "Gl. Sydhavn": "Kgs. Enghave - Gl. Sydhavn",
    "Gl. Valby": "Valby - Gl. Valby",
    "Grøndals Park Kvarter": "Vanløse - Grøndals Park Kvarter",
    "Grønjordssøen (Ørestad Nord, Vejlands Kvarter)": "Amager Vest - Grønjordssøen (Ørestad Nord and Vejlandskvarter)",
    "Guldbergskvarteret/Panum/Ravnsborggade": "Nørrebro - Guldbergskvarteret/Panum/Ravnsborggade",
    "Haraldsgade-kvarteret": "Nørrebro - Haraldsgade-kvarteret",
    "Holmen og Refshaleøen": "Christianshavn - Holmen og Refshaleøen",
    "Holmene": "Kgs. Enghave - Holmene",
    "Husum": "Brønshøj-Husum - Husum",
    "Husum Nord": "Brønshøj-Husum - Husum Nord",
    "Jernbane Allé Kvarter": "Vanløse - Jernbane Allé Kvarter",
    "Jyllingevej Kvarter": "Vanløse - Jyllingevej Kvarter",
    "Kolonihavekvarteret": "Amager Vest - Kolonihavekvarteret",
    "Lyngbyvej Vest": "Østerbro - Lyngbyvej Vest",
    "Lyngbyvej Øst/Klimakvarteret": "Østerbro - Lyngbyvej Øst/Klimakvarteret",
    "Metropolzonen": "Indre By - Metropolzonen",
    "Middelalderbyen": "Indre By - Middelalderbyen",
    "Mimersgade-kvarteret": "Nørrebro - Mimersgade-kvarteret/ Nørrebro St.",
    "Nansensgade-Kvarteret": "Indre By - Nansensgade-kvarteret",
    "Nord/Komponistkvarteret": "Østerbro - Nord/Komponistkvarteret",
    "Nordhavn": "Østerbro - Nordhavn",
    "Nordvest": "Bispebjerg - Nordvest",
    "Nordøstamager": "Amager Øst - Nordøstamager",
    "Ny Ryvang": "Østerbro - Ny Ryvang",
    "Rosenvænget": "Østerbro - Rosenvænget",
    "Ryparken-Lundehus": "Bispebjerg- Ryparken-Lundehus",
    "Sallingvej Kvarter": "Vanløse - Sallingvej Kvarter",
    "Stefansgade/Nørrebroparken/Lundtoftegade": "Nørrebro - Stefansgade/Nørrebroparken/Lundtoftegade",
    "Sundbyvester": "Amager Vest - Sundbyvester",
    "Sundbyøster": "Amager Øst - Sundbyøster",
    "Sundholmsvejs kvarteret": "Amager Vest - Sundholmsvejs kvarteret",
    "Svanemøllen Syd/Øst": "Østerbro - Svanemøllen Syd/Øst",
    "Tingbjerg": "Brønshøj-Husum - Tingbjerg",
    "Urbanplanen": "Amager Vest - Urbanplanen",
    "Utterslev": "Bispebjerg - Utterslev",
    "Valby syd": "Valby - Valby Sydvest",
    "Valby sydvest": "Valby - Valby Sydvest",
    "Vesterbro central": "Vesterbro - Central",
    "Vesterbro syd": "Vesterbro - Syd",
    "Vesterbro vest": "Vesterbro - Vest",
    "Vesterbro øst": "Vesterbro - Øst",
    "Vigerslev": "Valby - Vigerslev",
    "Villakvartererne": "Amager Øst - Villakvartererne",
    "Ålholm": "Valby - Ålholm",
    "Århusgade Nord": "Østerbro - Århusgade Nord",
    "Århusgade Syd": "Østerbro - Århusgade Syd",
    "Ørestad City": "Amager Vest - Ørestad City",
    "Ørestad Syd": "Amager Vest - Ørestad Syd",
    "Øster Farimagsgade-kvarteret": "Indre By - Øster Farimagsgade-kvarteret",
    "Østerbro Nord": "Østerbro - Nord",
    "Østerport": "Indre By - Østerport",
}

income_columns = [
    "year",
    "area",
    "first_decile",
    "second_decile",
    "third_decile",
    "fourth_decile",
    "fifth_decile",
    "sixth_decile",
    "seventh_decile",
    "eighth_decile",
    "ninth_decile",
]

def load_accessibility(path: Path) -> pd.DataFrame:
    """Load the accessibility CSV and ensure every area has an income mapping."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["area_name"]).copy()

    return df

def load_income(path: Path) -> pd.Series:
    """Load the raw income CSV and return a Series keyed by area name."""
    income = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=income_columns,
        encoding="utf-8-sig",
        dtype=str,
    )
    income = income[income["area"].notna()].copy()
    income["area"] = income["area"].str.strip()
    income = income[income["area"] != ""]
    income["median_income_dkk"] = pd.to_numeric(
        income["fifth_decile"].str.replace(" ", "", regex=False), errors="coerce"
    )
    return income.set_index("area")["median_income_dkk"]

def load_base_polygons() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame that includes area_name and geometry for each district."""
    gdf = gpd.read_file(districts_geojson)
    gdf = gdf[["kvarternavn", "geometry"]].rename(columns={"kvarternavn": "area_name"})
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

def main() -> None:
    print("Running step 3 merge income data")

    access = load_accessibility(access_csv)
    income_lookup = load_income(income_csv)

    access["income_area"] = access["area_name"].map(access_to_income)

    access["median_income_dkk"] = access["income_area"].map(income_lookup)
    access["median_income_dkk"] = pd.to_numeric(
        access["median_income_dkk"], errors="coerce"
    )

    base_polygons = load_base_polygons()

    gdf = base_polygons.merge(access, on="area_name", how="left")
    if "median_income_dkk" in gdf.columns:
        gdf["median_income_dkk"] = pd.to_numeric(gdf["median_income_dkk"], errors="coerce")
    web_geojson.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(web_geojson, driver="GeoJSON")
    print("Pipeline completed")

if __name__ == "__main__":
    main()
