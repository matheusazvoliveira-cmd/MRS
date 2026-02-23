# %% [markdown]
# """
# Interactive rail map for MRS (CodigoFerr == 11)
# 
# Features:
# - Loads rail shapefile
# - Filters CodigoFerr == 11
# - Reprojects to EPSG:4326
# - Simplifies geometry (optional, recommended)
# - Interactive map with custom colors and commentaries via CSV
# - Exports to HTML
# """

# %%
# !pip -q install geopandas folium shapely pyproj branca pandas

# %% [markdown]
# Imports

# %%
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import Fullscreen, MeasureControl

# %% [markdown]
# Config

# %%
# --- REQUIRED ---
SHP_PATH = Path(r"C:\Users\matheus.deoliveira\OneDrive - Wabtec Corporation\Matheus\wabtec\Advanced Technology\MRS\Malha Ferroviária Federal (shp)\Malha Ferroviária Federal (shp)\Estacoes Code\LInhas\Linhas_BR.shp")  # <-- CHANGE THIS
FILTER_VALUE = 11                              # MRS

# --- OPTIONAL / RECOMMENDED ---
# If your shapefile has NO CRS defined, set it here (example: "EPSG:31983", "EPSG:4674", etc.)
# Leave as None if the CRS is already correct in the file.
SOURCE_CRS_IF_MISSING = "EPSG:4326"  # or EPSG:4674 if that's the correct one

# Column names
FILTER_COLUMN = "CodigoFerr"   # you said it's exact: CodigoFerr

# If you have a good identifier column for segments, put it here.
# If None, we will create one from the row index.
SEGMENT_ID_COLUMN = None  # e.g. "ID_TRECHO" or "CD_SEGMENTO" if you have one

# Optional: a "name" column to show on tooltip (if you have it)
NAME_COLUMN = None  # e.g. "NOME", "TRECHO", "LINHA", etc.

# Geometry simplification for performance:
# Increase for faster maps (e.g., 100-250), decrease for more detail (e.g., 10-50).
SIMPLIFY_TOLERANCE_METERS = 50

# Output files
OUT_HTML = Path("mrs_codigoFerr11_interactive.html")
OUT_GEOJSON = Path("mrs_codigoFerr11.geojson")
COMMENTS_CSV = Path("comments_colors.csv")

# %%
assert SHP_PATH.exists(), f"Shapefile not found: {SHP_PATH}"

gdf = gpd.read_file(SHP_PATH)
print("Loaded rows:", len(gdf))
print("CRS:", gdf.crs)
print("Columns:", list(gdf.columns))
gdf.head()

# %%
# If CRS missing, set it (ONLY if you know the correct CRS)
if gdf.crs is None and SOURCE_CRS_IF_MISSING is not None:
    gdf = gdf.set_crs(SOURCE_CRS_IF_MISSING)
    print("CRS was missing. Set CRS to:", SOURCE_CRS_IF_MISSING)

# Filter MRS
gdf_mrs = gdf[gdf[FILTER_COLUMN] == FILTER_VALUE].copy()
print("Rows after filter (CodigoFerr==11):", len(gdf_mrs))

# Basic sanity check
if len(gdf_mrs) == 0:
    raise ValueError("Filter returned 0 rows. Check if CodigoFerr is numeric/int or stored as string.")

# %%
gdf_mrs = gdf_mrs.to_crs(epsg=4326)
print("CRS after reprojection:", gdf_mrs.crs)

# %%
if SEGMENT_ID_COLUMN is None or SEGMENT_ID_COLUMN not in gdf_mrs.columns:
    # Create a stable ID from index (string)
    gdf_mrs = gdf_mrs.reset_index(drop=True)
    gdf_mrs["segment_id"] = gdf_mrs.index.astype(str)
    SEGMENT_ID_COLUMN = "segment_id"
    print("Created SEGMENT_ID_COLUMN:", SEGMENT_ID_COLUMN)
else:
    # Ensure it's string for merging with CSV
    gdf_mrs[SEGMENT_ID_COLUMN] = gdf_mrs[SEGMENT_ID_COLUMN].astype(str)

# Create a display name column for tooltip/popup
if NAME_COLUMN is None or NAME_COLUMN not in gdf_mrs.columns:
    gdf_mrs["display_name"] = "MRS segment " + gdf_mrs[SEGMENT_ID_COLUMN].astype(str)
    NAME_COLUMN = "display_name"
else:
    gdf_mrs[NAME_COLUMN] = gdf_mrs[NAME_COLUMN].astype(str)

gdf_mrs[[SEGMENT_ID_COLUMN, NAME_COLUMN]].head()

# %%
# Simplify in a metric CRS so tolerance is meaningful
gdf_m = gdf_mrs.to_crs(epsg=3857)
gdf_m["geometry"] = gdf_m.geometry.simplify(SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)

gdf_simpl = gdf_m.to_crs(epsg=4326)
print("Simplified geometry with tolerance (m):", SIMPLIFY_TOLERANCE_METERS)

# %%
# Create a template CSV if it doesn't exist
if not COMMENTS_CSV.exists():
    template = gdf_simpl[[SEGMENT_ID_COLUMN, NAME_COLUMN]].copy()
    template.rename(columns={SEGMENT_ID_COLUMN: "segment_id", NAME_COLUMN: "name"}, inplace=True)

    # Default fields you can edit later:
    template["color"] = "#2ca02c"   # default green
    template["comment"] = ""        # your notes here
    
    template.to_csv(COMMENTS_CSV, index=False, encoding="utf-8")
    print(f"Created template: {COMMENTS_CSV.resolve()}")
    print("Edit this CSV to set colors and comments, then re-run this cell.")
else:
    print(f"Using existing: {COMMENTS_CSV.resolve()}")

comments = pd.read_csv(COMMENTS_CSV, dtype={"segment_id": str})
comments.head()

# %%
# Ensure merge keys are string
gdf_simpl[SEGMENT_ID_COLUMN] = gdf_simpl[SEGMENT_ID_COLUMN].astype(str)
comments["segment_id"] = comments["segment_id"].astype(str)

gdf_map = gdf_simpl.merge(
    comments[["segment_id", "color", "comment", "name"]],
    left_on=SEGMENT_ID_COLUMN,
    right_on="segment_id",
    how="left"
)

# fallback defaults if any missing
gdf_map["color"] = gdf_map["color"].fillna("#2ca02c")
gdf_map["comment"] = gdf_map["comment"].fillna("")
gdf_map["name"] = gdf_map["name"].fillna(gdf_map[NAME_COLUMN])

gdf_map[[SEGMENT_ID_COLUMN, "name", "color", "comment"]].head()

# %%
# Center map on the dataset
centroid = gdf_map.geometry.unary_union.centroid
center = [centroid.y, centroid.x]

m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

Fullscreen().add_to(m)
MeasureControl().add_to(m)

def style_fn(feature):
    return {
        "color": feature["properties"].get("color", "#2ca02c"),
        "weight": 4,
        "opacity": 0.95
    }

tooltip = folium.GeoJsonTooltip(
    fields=["name"],
    aliases=["Line:"],
    sticky=True
)

popup = folium.GeoJsonPopup(
    fields=["name", "comment"],
    aliases=["Line:", "Commentary:"],
    max_width=450
)

folium.GeoJson(
    gdf_map,
    name="MRS (CodigoFerr = 11)",
    style_function=style_fn,
    tooltip=tooltip,
    popup=popup
).add_to(m)

folium.LayerControl().add_to(m)

m


