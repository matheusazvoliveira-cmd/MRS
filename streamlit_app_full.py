"""
MRS Interactive Corridor Builder — Streamlit Full Port

Ported from: MRS_Interactive_Corridor_Builder_v6_single_scroll_tooltip_free.ipynb

This app provides a complete routing UI on an interactive map with all the notebook's logic.
"""

import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress "missing ScriptRunContext" noise emitted during startup/cache init.
# Works regardless of which Streamlit sub-logger emits it.
class _NoScriptRunCtxFilter(logging.Filter):
    def filter(self, record):
        return 'missing ScriptRunContext' not in record.getMessage()

logging.getLogger('streamlit').addFilter(_NoScriptRunCtxFilter())

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time
import sys

import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from pyproj import Transformer
import hashlib
import shapely

import folium
from folium.plugins import Fullscreen
from folium.elements import JSCSSMixin
from streamlit_folium import st_folium
from jinja2 import Template

import routing
from routing import FILTER_COLUMN, MRS_CODE, NON_MRS_PENALTY_DEFAULT


# ============================================================================
# CONFIGURATION (match notebook settings)
# ============================================================================

def _app_base_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).parent.resolve()


def _resource_base_dir():
    return Path(getattr(sys, '_MEIPASS', _app_base_dir())).resolve()


def _resolve_resource(relative_path):
    rel = Path(relative_path)
    candidates = [
        _resource_base_dir() / rel,
        _app_base_dir() / rel,
        Path.cwd() / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RAILS_SHP_PATH = _resolve_resource(Path("LInhas") / "Linhas_BR.shp")
STATIONS_SHP_PATH = _resolve_resource(Path("Estacoes") / "Estacoes.shp")

# ---------------------------------------------------------------------------
# Operator ↔ ANTT rail-code mapping
# ---------------------------------------------------------------------------
RAIL_OPERATORS = {
    "MRS":            {"codes": [11],        "color": "#2ca02c",  "label": "MRS Logística"},
    "Rumo":           {"codes": [1, 5, 7, 9], "color": "#1f77b4", "label": "Rumo (Malha Sul / Paulista / Oeste / Norte)"},
    "Vale":           {"codes": [3, 4],       "color": "#9467bd",  "label": "Vale (EFC Carajás / EFVM Vitória-Minas)"},
    "VLI":            {"codes": [8, 12, 13],  "color": "#d62728",  "label": "VLI (FCA / FNS Norte / FNS Central)"},
    "Transnordestina": {"codes": [2],         "color": "#ff7f0e",  "label": "Transnordestina (CFN)"},
    "Ferroeste":      {"codes": [6],          "color": "#8c564b",  "label": "Ferroeste"},
    "FTC":            {"codes": [10],         "color": "#e377c2",  "label": "FTC (Tereza Cristina)"},
}

# Reverse lookup: code → operator name
CODE_TO_OPERATOR = {}
for _op_name, _op_info in RAIL_OPERATORS.items():
    for _code in _op_info["codes"]:
        CODE_TO_OPERATOR[_code] = _op_name

STRICT_MRS = "Strict MRS (11 only)"
PREFER_MRS = "Prefer MRS (penalize non-11)"
ALLOW_ALL = "Allow all (shortest distance)"

# UI label -> routing.py mode id
ROUTING_MODE_IDS = {
    STRICT_MRS: routing.STRICT_MRS,
    PREFER_MRS: routing.PREFER_MRS,
    ALLOW_ALL: routing.ALLOW_ALL,
}

# Routing graph disk cache (built once per shapefile version)
GRAPH_CACHE_DIR_NAME = ".cache"

# Styles
BASE_RAIL_COLOR = "#2ca02c"
BASE_RAIL_WEIGHT = 3

PREVIEW_COLOR = "#1f77b4"

HIGHLIGHT_WEIGHT = 5
HIGHLIGHT_OPACITY = 1.0

LABEL_FONT_SIZE = 17  # px — single source of truth for all station label text

# Map payload size
MAP_SIMPLIFY_TOLERANCE_DEG = 0.0005   # ~50 m — invisible at the zoom levels the map is used at
MAP_COORD_DECIMALS = 5                # ~1 m
HIGHLIGHT_DISPLAY_TOLERANCE_DEG = 0.0002  # ~20 m — "Simplify path coordinates" display option
STATION_LABEL_MIN_ZOOM = 10           # "All Stations" labels only appear from this zoom in

# Station label marker offset (px below the marker dot)
LABEL_MARGIN_TOP_DEFAULT = 12


# ============================================================================
# UTILITIES
# ============================================================================

to_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

# Default save file paths — absolute, anchored to app launch directory
_APP_DIR = _app_base_dir()
HIGHLIGHTS_SAVE_PATH = _APP_DIR / "committed_highlights.json"   # live auto-save (every commit/delete)
EXPORT_SAVE_PATH     = _APP_DIR / "exported_highlights.json"    # manual export snapshot only
STATION_GROUPS_SAVE_PATH = _APP_DIR / "station_groups.json"     # station group systems
STATION_GROUPS_EXPORT_PATH = _APP_DIR / "exported_station_groups.json"  # manual export snapshot

def save_highlights(highlights_dict):
    """Save committed highlights to disk as JSON."""
    if highlights_dict is None:
        return
    try:
        with open(str(HIGHLIGHTS_SAVE_PATH), 'w') as f:
            json.dump(highlights_dict, f, indent=2)
    except Exception as e:
        st.error(f"Save failed ({HIGHLIGHTS_SAVE_PATH}): {e}")

def load_highlights():
    """Load committed highlights from disk."""
    if HIGHLIGHTS_SAVE_PATH.exists():
        try:
            with open(str(HIGHLIGHTS_SAVE_PATH), 'r') as f:
                data = json.load(f)
            # Validate structure
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception as e:
            st.warning(f"Failed to load highlights from {HIGHLIGHTS_SAVE_PATH}: {e}")
    return {}


def save_station_group_systems(group_systems):
    """Save station group systems to disk as JSON."""
    if group_systems is None:
        return
    try:
        with open(str(STATION_GROUPS_SAVE_PATH), 'w') as f:
            json.dump(group_systems, f, indent=2)
    except Exception as e:
        st.error(f"Save failed ({STATION_GROUPS_SAVE_PATH}): {e}")


def _normalize_station_group_systems(raw):
    """Normalize station group payload, including migration from flat group format."""
    if not isinstance(raw, dict):
        return {}

    normalized = {}

    is_old_flat = all(
        isinstance(v, dict) and ('stations' in v or 'shape' in v or 'color' in v)
        for v in raw.values()
    ) if raw else False

    if is_old_flat:
        groups = {}
        for group_name, group_data in raw.items():
            if not isinstance(group_data, dict):
                continue
            groups[str(group_name)] = {
                'stations': list(group_data.get('stations', [])),
                'shape': str(group_data.get('shape', 'circle')).lower(),
                'color': str(group_data.get('color', '#d62728')),
            }
        if groups:
            normalized['Default'] = {'groups': groups}
        return normalized

    for system_name, system_data in raw.items():
        if not isinstance(system_data, dict):
            continue
        groups_in = system_data.get('groups', {})
        if not isinstance(groups_in, dict):
            continue

        groups_out = {}
        for group_name, group_data in groups_in.items():
            if not isinstance(group_data, dict):
                continue
            groups_out[str(group_name)] = {
                'stations': list(group_data.get('stations', [])),
                'shape': str(group_data.get('shape', 'circle')).lower(),
                'color': str(group_data.get('color', '#d62728')),
            }

        if groups_out:
            normalized[str(system_name)] = {'groups': groups_out}

    return normalized


def load_station_group_systems():
    """Load station group systems from disk."""
    if STATION_GROUPS_SAVE_PATH.exists():
        try:
            with open(str(STATION_GROUPS_SAVE_PATH), 'r') as f:
                data = json.load(f)
            return _normalize_station_group_systems(data)
        except Exception as e:
            st.warning(f"Failed to load station groups from {STATION_GROUPS_SAVE_PATH}: {e}")
    return {}


@st.cache_resource(show_spinner="Loading shapefiles...")
def load_data():
    """Load shapefiles — raises on failure so the caller can show the error in context."""
    if not RAILS_SHP_PATH.exists():
        raise FileNotFoundError(f"Rails shapefile not found: {RAILS_SHP_PATH}")
    if not STATIONS_SHP_PATH.exists():
        raise FileNotFoundError(f"Stations shapefile not found: {STATIONS_SHP_PATH}")

    # Read both shapefiles in parallel (I/O-bound — threading helps on first load)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_rails = ex.submit(routing.read_rails, RAILS_SHP_PATH)
        f_stations = ex.submit(routing.read_stations, STATIONS_SHP_PATH)
        return f_rails.result(), f_stations.result()


@st.cache_resource(show_spinner="Building routing graph (first run only)...")
def get_rail_graph(_rails_all_m, data_version_key):
    """National routing graph — loaded from the disk cache, or built once (~10 s) and cached."""
    _ = data_version_key
    return routing.load_or_build_graph(lambda: _rails_all_m, RAILS_SHP_PATH, _APP_DIR / GRAPH_CACHE_DIR_NAME)


def get_data_version_key():
    """Build a hashable key so derived-data cache invalidates when source files change."""
    rails_mtime = RAILS_SHP_PATH.stat().st_mtime_ns if RAILS_SHP_PATH.exists() else 0
    stations_mtime = STATIONS_SHP_PATH.stat().st_mtime_ns if STATIONS_SHP_PATH.exists() else 0
    return (str(RAILS_SHP_PATH), rails_mtime, str(STATIONS_SHP_PATH), stations_mtime)


# cache_resource (not cache_data): returns shared objects instead of unpickling a copy on every rerun
@st.cache_resource(show_spinner="Preparing data...")
def prepare_derived_data(_rails_all, _stations_all, data_version_key):
    """Compute all derived data (CRS transforms, GeoJSON conversions) — cached globally across all users."""
    _ = data_version_key
    # Phase 1: parallel CRS transforms + station prep
    # PROJ/GEOS are C libraries that release the GIL → real thread parallelism
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_rails_m = ex.submit(_rails_all.to_crs, epsg=3857)
        f_rails_ll = ex.submit(_rails_all.to_crs, epsg=4326)
        f_stations = ex.submit(prepare_stations, _stations_all)

        rails_all_m = f_rails_m.result()
        rails_all_ll = f_rails_ll.result()
        _, stations_near_ll, stations_lookup, labels = f_stations.result()

    centroid = stations_near_ll.geometry.union_all().centroid

    # Derive MRS subset from already-transformed data (no extra CRS transform)
    rails_mrs_ll = rails_all_ll[rails_all_ll[FILTER_COLUMN] == MRS_CODE].copy()

    # Phase 2: parallel GeoJSON conversions
    # Filter pre-transformed rails_all_ll instead of re-projecting per operator
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_mrs_json = ex.submit(_gdf_to_minimal_geojson, rails_mrs_ll)
        f_all_json = ex.submit(_gdf_to_minimal_geojson, rails_all_ll)

        # Per-operator GeoJSONs — filter already-transformed data
        op_futures = {}
        for op_name, op_info in RAIL_OPERATORS.items():
            op_mask = rails_all_ll[FILTER_COLUMN].isin(op_info['codes'])
            if op_mask.any():
                op_futures[op_name] = ex.submit(
                    _gdf_to_minimal_geojson, rails_all_ll[op_mask]
                )
            else:
                op_futures[op_name] = None

        rails_mrs_geojson = f_mrs_json.result()
        rails_all_geojson = f_all_json.result()
        operator_geojsons = {}
        for op_name, fut in op_futures.items():
            if fut is not None:
                operator_geojsons[op_name] = fut.result()
            else:
                operator_geojsons[op_name] = {'type': 'FeatureCollection', 'features': []}

    stations_geojson = _stations_to_geojson(stations_near_ll)
    stations_by_label = {
        f['properties']['label']: {
            'label': f['properties']['label'],
            'lat': f['geometry']['coordinates'][1],
            'lon': f['geometry']['coordinates'][0],
        }
        for f in stations_geojson['features']
    }

    # Versioned files for the browser to fetch and cache (None -> embed in the map)
    static_layer_urls = {
        'rails_all': _publish_static_geojson('rails_all', rails_all_geojson),
        'stations': _publish_static_geojson('stations', stations_geojson),
    }
    for op_name, op_geojson in operator_geojsons.items():
        if op_geojson['features']:
            static_layer_urls[f"op_{op_name}"] = _publish_static_geojson(f"op_{op_name}", op_geojson)

    return {
        'rails_all_m': rails_all_m,
        'rails_mrs_ll': rails_mrs_ll,
        'rails_all_ll': rails_all_ll,
        'stations_near_ll': stations_near_ll,
        'stations_lookup': stations_lookup,
        'labels': labels,
        'station_counts_by_operator': {
            op_name: int((stations_lookup['operator_name'] == op_name).sum())
            for op_name in RAIL_OPERATORS.keys()
        },
        'map_center': [centroid.y, centroid.x],
        'rails_mrs_geojson': rails_mrs_geojson,
        'rails_all_geojson': rails_all_geojson,
        'operator_geojsons': operator_geojsons,
        'stations_geojson': stations_geojson,
        'stations_by_label': stations_by_label,
        'static_layer_urls': static_layer_urls,
    }


def prepare_stations(stations_all):
    """Point stations with labels (routing.prepare_stations) plus operator names and lat/lon copy."""
    stations_near = routing.prepare_stations(stations_all)
    stations_near['operator_name'] = stations_near[FILTER_COLUMN].map(CODE_TO_OPERATOR).fillna('Unknown')

    stations_near_ll = stations_near.to_crs(epsg=4326)
    stations_lookup = stations_near.set_index('station_label')
    labels = sorted(list(stations_lookup.index))

    return stations_near, stations_near_ll, stations_lookup, labels


def _path_station_count(path_info):
    """Return the number of stations represented by a preview/committed path."""
    stored = path_info.get('station_count')
    if stored is not None and stored is not pd.NA:
        try:
            return int(stored)
        except (TypeError, ValueError):
            pass

    preview_points = path_info.get('preview_station_points', [])
    if isinstance(preview_points, list) and preview_points:
        return len(preview_points)

    return len(path_info.get('order', []))


def stations_along_path(coords_ll, stations_lookup, tolerance_m=150.0):
    """Find stations that lie near a route polyline (meters in EPSG:3857).

    Memoized by a hash of the coordinates: the sidebar asks for every committed
    path on every rerun, but the answer only changes when the path does."""
    if not coords_ll or len(coords_ll) < 2:
        return []
    try:
        coords_key = hashlib.md5(np.asarray(coords_ll, dtype=float).tobytes()).hexdigest()
    except Exception:
        return []
    return _stations_along_path_cached(coords_key, float(tolerance_m), coords_ll, stations_lookup)


@st.cache_data(max_entries=512, show_spinner=False)
def _stations_along_path_cached(coords_key, tolerance_m, _coords_ll, _stations_lookup):
    coords_ll = _coords_ll
    stations_lookup = _stations_lookup
    try:
        route_xy = [to_3857.transform(float(lon), float(lat)) for lat, lon in coords_ll]
        route_line = LineString(route_xy)
    except Exception:
        return []

    if route_line.length == 0:
        return []

    stations_m = stations_lookup[['geometry']].copy()
    candidates = stations_m[stations_m.geometry.distance(route_line) <= float(tolerance_m)].copy()
    if candidates.empty:
        return []

    candidates['route_proj'] = candidates.geometry.apply(route_line.project)
    candidates = candidates.sort_values('route_proj')

    # Vectorized coordinate extraction — avoids iterrows() overhead
    _xs = candidates.geometry.x.values
    _ys = candidates.geometry.y.values
    _labels = candidates.index.values
    out = []
    for i in range(len(candidates)):
        lon, lat = to_4326.transform(float(_xs[i]), float(_ys[i]))
        out.append({'label': str(_labels[i]), 'lat': float(lat), 'lon': float(lon)})
    return out


def compute_label_offset(lat, lon, layout_state=None, collision_radius_m=5000.0):
    """Compute vertical/horizontal offsets with alternating left/right pattern along route."""
    if layout_state is None:
        return 8, -50

    # Simple alternating pattern: even stations text ends at marker, odd starts at marker
    count = len(layout_state)
    
    if count % 2 == 0:
        # Even: text ends at marker (right edge of label at station dot)
        translate_x_pct = -100
    else:
        # Odd: text starts at marker (left edge of label at station dot)
        translate_x_pct = 0
    
    margin_top = 8
    layout_state.append({'count': count})
    return margin_top, translate_x_pct


def add_station_shape_marker(
    layer,
    lat,
    lon,
    label,
    shape="circle",
    color="#d62728",
    size=8,
    font_size=LABEL_FONT_SIZE,
    label_layout_state=None,
):
    """Render a station marker using circle/square/triangle shape with label beneath."""
    marker_shape = (shape or "circle").lower()
    marker_size = max(6, int(size))

    if marker_shape == "circle":
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            color="#333333",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=label,
            popup=label,
        ).add_to(layer)

    elif marker_shape == "square":
        side = marker_size * 2
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=(
                    f"<div style='width: {side}px; height: {side}px; "
                    f"background: {color}; border: 1.5px solid #333333; "
                    f"opacity: 0.95;'></div>"
                ),
                icon_size=(side, side),
                icon_anchor=(side // 2, side // 2),
            ),
            tooltip=label,
            popup=label,
        ).add_to(layer)

    else:  # triangle
        triangle_height = marker_size * 2
        triangle_half = marker_size
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=(
                    f"<div style='width: 0; height: 0; "
                    f"border-left: {triangle_half}px solid transparent; "
                    f"border-right: {triangle_half}px solid transparent; "
                    f"border-bottom: {triangle_height}px solid {color}; "
                    f"filter: drop-shadow(0 0 1px #333333);'></div>"
                ),
                icon_size=(triangle_half * 2, triangle_height),
                icon_anchor=(triangle_half, triangle_height),
            ),
            tooltip=label,
            popup=label,
        ).add_to(layer)

    label_margin_top, label_translate_x = compute_label_offset(
        lat,
        lon,
        layout_state=label_layout_state,
    )

    # Add label text near marker with collision-avoidance offset
    _add_label_marker(layer, lat, lon, label,
                      margin_top=label_margin_top,
                      translate_x_pct=label_translate_x,
                      font_size=font_size)


def _add_label_marker(layer, lat, lon, text, margin_top=LABEL_MARGIN_TOP_DEFAULT,
                      translate_x_pct=-50, font_size=LABEL_FONT_SIZE):
    """Add a text label DivIcon marker to *layer*.

    Centralises the repeated label-rendering pattern used for station
    endpoint labels, committed highlight labels, and station group labels.
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=(
                f"<div style='display:inline-block;font-size:{font_size}px;font-weight:bold;"
                f"background-color:transparent;color:black;padding:2px 4px;"
                f"border-radius:2px;border:none;"
                f"white-space:nowrap;transform:translateX({translate_x_pct}%);"
                f"margin-top:{margin_top}px;'>{text}</div>"
            ),
            icon_size=(0, 0),
            icon_anchor=(0, 0),
        ),
    ).add_to(layer)


def _make_path_payload(path_info):
    """Create a shallow copy of a path payload dict with list fields copied."""
    return {
        'coords_ll': list(path_info.get('coords_ll', [])),
        'order': list(path_info.get('order', [])),
        'preview_station_points': list(path_info.get('preview_station_points', [])),
        'selected_station_markers': list(path_info.get('selected_station_markers', [])),
        'total_km': float(path_info.get('total_km', 0.0)),
        'route_color': path_info.get('route_color', PREVIEW_COLOR),
        'highlight_name': path_info.get('highlight_name'),
        'path_name': path_info.get('path_name', ''),
        'station_count': _path_station_count(path_info),
    }


def _station_render_key(lat, lon):
    """Stable key for station deduplication based on position."""
    return (round(float(lat), 6), round(float(lon), 6))


def _reverse_path_payload(path_info):
    p = _make_path_payload(path_info)
    p['coords_ll'].reverse()
    p['order'].reverse()
    p['preview_station_points'].reverse()
    p['selected_station_markers'].reverse()
    return p


def _orient_path_to_shared(path_info, shared_station, must_end_at_shared):
    order = path_info.get('order', [])
    if not order:
        return None

    starts_shared = order[0] == shared_station
    ends_shared = order[-1] == shared_station

    if must_end_at_shared:
        if ends_shared:
            return _make_path_payload(path_info)
        if starts_shared:
            return _reverse_path_payload(path_info)
    else:
        if starts_shared:
            return _make_path_payload(path_info)
        if ends_shared:
            return _reverse_path_payload(path_info)

    return None


def merge_paths_at_shared_station(path_a, path_b, system_name):
    order_a = path_a.get('order', [])
    order_b = path_b.get('order', [])
    if len(order_a) < 2 or len(order_b) < 2:
        return None, "Both paths must have at least 2 stations."

    endpoints_a = {order_a[0], order_a[-1]}
    endpoints_b = {order_b[0], order_b[-1]}
    shared = list(endpoints_a.intersection(endpoints_b))

    if len(shared) != 1:
        return None, "Selected paths must share exactly one endpoint station."

    shared_station = shared[0]
    first = _orient_path_to_shared(path_a, shared_station, must_end_at_shared=True)
    second = _orient_path_to_shared(path_b, shared_station, must_end_at_shared=False)

    if first is None or second is None:
        return None, "Could not orient paths to merge at the shared station."

    coords_first = first.get('coords_ll', [])
    coords_second = second.get('coords_ll', [])
    if not coords_first or not coords_second:
        return None, "Both paths must contain coordinates."

    merged_coords = list(coords_first)
    if coords_first[-1] == coords_second[0]:
        merged_coords.extend(coords_second[1:])
    else:
        merged_coords.extend(coords_second)

    merged_order = list(first.get('order', []))
    if second.get('order', []):
        merged_order.extend(second['order'][1:])

    stations_first = list(first.get('preview_station_points', []))
    stations_second = list(second.get('preview_station_points', []))
    merged_stations = stations_first + stations_second[1:] if stations_second else stations_first

    markers_first = list(first.get('selected_station_markers', []))
    markers_second = list(second.get('selected_station_markers', []))
    marker_seen = {m.get('label') for m in markers_first if isinstance(m, dict)}
    merged_markers = list(markers_first)
    for marker in markers_second:
        label = marker.get('label') if isinstance(marker, dict) else None
        if label and label in marker_seen:
            continue
        merged_markers.append(marker)
        if label:
            marker_seen.add(label)

    merged_path = {
        'coords_ll': merged_coords,
        'order': merged_order,
        'total_km': float(first.get('total_km', 0.0)) + float(second.get('total_km', 0.0)),
        'route_color': first.get('route_color', PREVIEW_COLOR),
        'highlight_name': system_name,
        'path_name': f"Merged ({merged_order[0]} → {merged_order[-1]})",
        'preview_station_points': merged_stations,
        'selected_station_markers': merged_markers,
        'station_count': len(merged_stations),
    }
    return merged_path, None


def _gdf_to_minimal_geojson(gdf):
    """Convert EPSG:4326 line geometries to a compact GeoJSON FeatureCollection.

    Strips all attribute columns, simplifies (~50 m), rounds coordinates (~1 m)
    and packs every line into ONE MultiLineString feature — a fraction of the
    size of gdf.to_json().  Used to render rail layers as a single folium.GeoJson.
    """
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    geoms = geoms[~shapely.is_missing(geoms)]
    parts = shapely.get_parts(shapely.simplify(geoms, MAP_SIMPLIFY_TOLERANCE_DEG))
    parts = parts[shapely.get_type_id(parts) == 1]  # LineStrings only
    lines = [np.round(shapely.get_coordinates(p), MAP_COORD_DECIMALS).tolist() for p in parts]
    if not lines:
        return {'type': 'FeatureCollection', 'features': []}
    return {'type': 'FeatureCollection', 'features': [{
        'type': 'Feature',
        'geometry': {'type': 'MultiLineString', 'coordinates': lines},
        'properties': {},
    }]}


def _stations_to_geojson(stations_ll):
    """Point FeatureCollection of all stations (label + popup text) for one GeoJson layer."""
    labels = stations_ll['station_label'].astype(str).values
    codes = (
        stations_ll['station_code'].astype(str).values
        if 'station_code' in stations_ll.columns
        else np.full(len(stations_ll), '', dtype=object)
    )
    lons = np.round(stations_ll.geometry.x.values, MAP_COORD_DECIMALS)
    lats = np.round(stations_ll.geometry.y.values, MAP_COORD_DECIMALS)
    features = [
        {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [float(lons[i]), float(lats[i])]},
            'properties': {
                'label': labels[i],
                'popup': f"{labels[i]} | Code: {codes[i]}" if codes[i] else labels[i],
            },
        }
        for i in range(len(stations_ll))
    ]
    return {'type': 'FeatureCollection', 'features': features}


def _simplify_latlon(coords, tolerance=None, max_points=None):
    """Douglas-Peucker simplification of [(lat, lon), ...] that keeps the route's shape
    (unlike keeping every Nth point).  With *max_points*, the tolerance grows until
    the result fits.  Returns [[lat, lon], ...]."""
    if len(coords) < 3:
        return [list(c) for c in coords]
    line = LineString([(c[1], c[0]) for c in coords])

    def run(tol):
        return [[lat, lon] for lon, lat in line.simplify(tol, preserve_topology=False).coords]

    if max_points is None:
        return run(tolerance)
    tol = 1e-5
    out = run(tol)
    while len(out) > max_points and tol < 1.0:
        tol *= 1.5
        out = run(tol)
    return out


# ============================================================================
# MAP UI CONSTANTS (module-level — avoid re-creating every Streamlit rerun)
# ============================================================================

class _ZoomDisplay(JSCSSMixin, folium.MacroElement):
    """Leaflet control that shows the current map zoom level."""
    _template = Template("""
    {% macro script(this, kwargs) %}
        var zoomDisplay = L.Control.extend({
            options: { position: 'topleft' },
            onAdd: function (map) {
                var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                container.style.backgroundColor = 'white';
                container.style.padding = '5px 10px';
                container.style.fontSize = '14px';
                container.style.fontWeight = 'bold';
                container.style.cursor = 'default';
                container.innerHTML = 'Zoom: ' + map.getZoom().toFixed(2);
                map.on('zoomend', function() {
                    container.innerHTML = 'Zoom: ' + map.getZoom().toFixed(2);
                });
                return container;
            }
        });
        {{ this._parent.get_name() }}.addControl(new zoomDisplay());
    {% endmacro %}
    """)


class _RemoteGeoJson(JSCSSMixin, folium.MacroElement):
    """Static GeoJSON layer the browser loads by URL instead of having it embedded
    in the map HTML on every rerun.

    The file name carries a content hash, so the browser downloads it once and
    reuses its HTTP cache until the data changes.  Data is only fetched the first
    time the parent FeatureGroup is shown, and drawn on canvas.  Falls back to
    embedding *data* when static serving is unavailable (url=None).

    kind='lines'    -> polylines styled with *style*
    kind='stations' -> circle markers (*style*) with popup + permanent label that
                       is hidden below STATION_LABEL_MIN_ZOOM
    """
    _template = Template("""
    {% macro header(this, kwargs) %}
        {% if this.kind == 'stations' %}
        <style>
            .leaflet-tooltip.station-label {
                background: transparent; border: none; box-shadow: none; padding: 0 4px;
                font-size: {{ this.font_size }}px; font-weight: bold; color: black; white-space: nowrap;
            }
            .leaflet-tooltip.station-label:before { display: none; }
            .hide-station-labels .leaflet-tooltip.station-label { display: none; }
        </style>
        {% endif %}
    {% endmacro %}
    {% macro script(this, kwargs) %}
    (function () {
        var group = {{ this._parent.get_name() }};
        var renderer = L.canvas({padding: 0.5});
        var started = false;
        var labelled = [];
        function draw(data) {
            L.geoJSON(data, {
            {% if this.kind == 'stations' %}
                pointToLayer: function (feature, latlng) {
                    return L.circleMarker(latlng, Object.assign({renderer: renderer}, {{ this.style|tojson }}));
                },
                onEachFeature: function (feature, layer) {
                    layer.bindPopup(feature.properties.popup);
                    layer.bindTooltip(feature.properties.label,
                        {permanent: true, direction: 'bottom', className: 'station-label'});
                    labelled.push(layer);
                }
            {% else %}
                renderer: renderer,
                style: function () { return {{ this.style|tojson }}; }
            {% endif %}
            }).addTo(group);
            {% if this.kind == 'stations' %}declutter();{% endif %}
        }
        function load() {
            if (started) return;
            started = true;
            {% if this.url %}
            fetch({{ this.url|tojson }}).then(function (r) { return r.json(); }).then(draw);
            {% else %}
            draw({{ this.data|tojson }});
            {% endif %}
        }
        group.on('add', load);
        // Layers start on the map and LayerControl removes hidden ones afterwards,
        // so only load once all scripts have run.
        setTimeout(function () { if (group._map) load(); }, 0);
        {% if this.kind == 'stations' %}
        var map = {{ this._parent._parent.get_name() }};
        function toggleLabels() {
            map.getContainer().classList.toggle('hide-station-labels', map.getZoom() < {{ this.min_label_zoom }});
        }
        // Hide labels that would overlap one already shown (like OpenLayers' declutter).
        // All writes, then all reads, then all writes — keeps it to one layout pass.
        function declutter() {
            if (!group._map || map.getZoom() < {{ this.min_label_zoom }}) return;
            var els = [];
            labelled.forEach(function (layer) {
                var el = layer.getTooltip() && layer.getTooltip().getElement();
                if (el) { el.style.visibility = ''; els.push(el); }
            });
            var view = map.getContainer().getBoundingClientRect();
            var rects = els.map(function (el) { return el.getBoundingClientRect(); });
            var placed = [];
            rects.forEach(function (r, i) {
                if (r.right < view.left || r.left > view.right || r.bottom < view.top || r.top > view.bottom) return;
                var hit = placed.some(function (p) {
                    return r.left < p.right && r.right > p.left && r.top < p.bottom && r.bottom > p.top;
                });
                if (hit) { els[i].style.visibility = 'hidden'; } else { placed.push(r); }
            });
        }
        map.on('zoomend', toggleLabels);
        map.on('zoomend moveend', declutter);
        group.on('add', function () { setTimeout(declutter, 0); });
        toggleLabels();
        {% endif %}
    })();
    {% endmacro %}
    """)

    def __init__(self, url, data, kind, style, font_size=LABEL_FONT_SIZE, min_label_zoom=STATION_LABEL_MIN_ZOOM):
        super().__init__()
        self._name = 'RemoteGeoJson'
        self.url = url
        self.data = data if url is None else None
        self.kind = kind
        self.style = style
        self.font_size = font_size
        self.min_label_zoom = min_label_zoom


# Generated GeoJSON served by Streamlit at <base>/app/static/ (server.enableStaticServing).
# Must live next to the main script — that is where Streamlit looks for it.
STATIC_DIR = Path(__file__).parent.resolve() / "static"


def _publish_static_geojson(name, data):
    """Write *data* to STATIC_DIR under a content-hashed file name and return its URL,
    or None if static serving is disabled or the folder is not writable."""
    try:
        if not st.get_option("server.enableStaticServing"):
            return None
        payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
        file_name = f"{name}.{hashlib.md5(payload).hexdigest()[:12]}.json"
        path = STATIC_DIR / file_name
        if not path.exists():
            STATIC_DIR.mkdir(exist_ok=True)
            for old in STATIC_DIR.glob(f"{name}.*.json"):
                old.unlink(missing_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(payload)
            tmp.replace(path)
        base = (st.get_option("server.baseUrlPath") or "").strip('/')
        return f"{'/' + base if base else ''}/app/static/{file_name}"
    except Exception:
        return None


# Brazil bounding box — tiles outside this rectangle are never requested
_BR_BOUNDS = [[-33.75, -73.99], [5.27, -34.79]]
_ORM_ATTR = (
    'Map tiles by <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a> '
    '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>) | '
    'Data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

_LAYER_SCROLL_FIX = """
<style>
    .leaflet-control-layers{width:auto;min-width:200px;max-width:300px;max-height:75vh;overflow:visible;bottom:10px!important}
    .leaflet-control-layers-expanded{width:auto;min-width:200px;max-width:300px;max-height:75vh;overflow:visible}
    .leaflet-control-layers-list{max-height:70vh!important;min-height:auto;overflow-y:scroll!important;overflow-x:hidden!important;margin-bottom:0;padding:4px 8px}
    .leaflet-control-layers-item{margin:4px 0;word-break:break-word}
    .leaflet-control-layers-list::-webkit-scrollbar{width:10px}
    .leaflet-control-layers-list::-webkit-scrollbar-track{background:#f1f1f1}
    .leaflet-control-layers-list::-webkit-scrollbar-thumb{background:#888;border-radius:5px}
    .leaflet-control-layers-list::-webkit-scrollbar-thumb:hover{background:#555}
</style>
<script>
window.addEventListener('load',function(){var l=document.querySelector('.leaflet-control-layers-list');if(l){l.style.maxHeight='70vh';l.style.overflowY='scroll';l.style.overflowX='hidden'}});
var l=document.querySelector('.leaflet-control-layers-list');if(l){l.style.maxHeight='70vh';l.style.overflowY='scroll';l.style.overflowX='hidden'}
</script>
"""


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="MRS Corridor Builder", layout="wide")
    st.title("MRS Interactive Corridor Builder — Streamlit")

    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    if 'committed_highlights' not in st.session_state:
        st.session_state.committed_highlights = load_highlights()
    if 'station_group_systems' not in st.session_state:
        st.session_state.station_group_systems = load_station_group_systems()
    if 'active_station_group_systems' not in st.session_state:
        st.session_state.active_station_group_systems = []
    if 'active_operator_layers' not in st.session_state:
        st.session_state.active_operator_layers = ['MRS']  # MRS shown by default

    # Load raw data (instant after first call — @st.cache_data)
    try:
        rails_all, stations_all = load_data()
    except Exception as e:
        st.error(f"Could not load shapefiles: {e}")
        return

    # Compute derived data — globally cached across all users (@st.cache_data)
    # First user triggers computation (~10-15s), all subsequent users get instant cached result
    data_version_key = get_data_version_key()
    derived_data = prepare_derived_data(rails_all, stations_all, data_version_key)
    st.session_state.derived_data = derived_data  # Store in session_state for convenience
    st.success(f"Loaded {derived_data['labels'].__len__()} stations, {len(rails_all)} rail features")

    d = derived_data
    rails_all_m      = d['rails_all_m']
    rails_mrs_ll     = d['rails_mrs_ll']
    rails_all_ll     = d['rails_all_ll']
    stations_near_ll = d['stations_near_ll']
    stations_lookup  = d['stations_lookup']
    labels           = d['labels']
    station_counts_by_operator = d['station_counts_by_operator']
    map_center       = d['map_center']

    # Sidebar controls
    with st.sidebar:
        st.header("Route Builder")

        # A form: picking stations/options does not rerun the app (and re-send the
        # map) until one of the buttons is pressed.
        with st.form("route_form", border=False):
            col1, col2 = st.columns(2)
            with col1:
                origin = st.selectbox("Station A", options=labels, key="origin")
            with col2:
                destination = st.selectbox("Station B", options=labels, index=min(1, len(labels)-1), key="dest")

            vias = st.multiselect("VIA stations (optional)", options=labels, key="vias")

            routing_mode = st.radio("Routing mode", [STRICT_MRS, PREFER_MRS, ALLOW_ALL], index=2)
            penalty_input = st.slider("Non-MRS penalty (Prefer MRS mode)", 1.0, 100.0, NON_MRS_PENALTY_DEFAULT, key="penalty")
            penalty = penalty_input if routing_mode == PREFER_MRS else 1.0

            color = st.color_picker("Route color", "#1f77b4", key="color")
            highlight_name = st.text_input("System name", "Corridor 1", key="name")
            path_name = st.text_input("Path name (optional)", "", key="path_name",
                                       placeholder="e.g. Direct, Via Campinas…")

            run_btn = st.form_submit_button("Compute Preview")
            commit_btn = st.form_submit_button("Commit Highlight")
        progress_bar = st.progress(0)
        status_txt = st.empty()

        st.divider()
        st.subheader("🎨 Rendering Options")
        render_committed = st.checkbox("Show committed highlights on map", value=True, key="render_committed",
                                       help="Toggle off to speed up map rendering with many paths")
        if render_committed:
            simplify_coords = st.checkbox("Simplify path coordinates", value=False, key="simplify",
                                         help="Reduce coordinate density for faster rendering (slight visual quality loss)")
        else:
            simplify_coords = False

        st.divider()
        st.subheader("📍 Station Finder")
        with st.form("station_finder_form", border=False):
            find_station = st.selectbox(
                "Find station on map:",
                options=["(Select a station...)"] + labels,
                key="find_station",
                help="Highlight a station on the map to find its location"
            )
            mark_btn = st.form_submit_button("📌 Mark on Map")
        if mark_btn and find_station != "(Select a station...)":
            st.session_state.highlighted_station = find_station
            st.success(f"Marked: {find_station}")
        if st.session_state.get('highlighted_station'):
            if st.button("🔄 Clear Marker", key="clear_marker"):
                st.session_state.highlighted_station = None
                st.info("Marker cleared")

        st.divider()
        st.subheader("� Operator Rail Layers")
        st.caption("Toggle rail network layers by operator/concessionaire.")
        operator_names = sorted(RAIL_OPERATORS.keys())
        st.multiselect(
            "Show operators on map",
            options=operator_names,
            default=st.session_state.active_operator_layers,
            key="active_operator_layers",
            format_func=lambda op: f"{op} — {RAIL_OPERATORS[op]['label']} ({station_counts_by_operator.get(op, 0)} stations)",
        )
        active_operator_station_total = sum(
            station_counts_by_operator.get(op, 0)
            for op in st.session_state.get('active_operator_layers', [])
        )
        if st.session_state.get('active_operator_layers'):
            st.caption(
                "Selected operators: "
                + " | ".join(
                    f"{op}: {station_counts_by_operator.get(op, 0)}"
                    for op in st.session_state['active_operator_layers']
                )
            )
            st.caption(f"Total stations across selected operators: {active_operator_station_total}")

        st.divider()
        st.subheader("�🗂️ Station Group Systems")
        st.caption("Create systems and add multiple station groups per system (each group with its own marker shape/color).")

        # Form: typing names / picking stations does not rerun the app until Save
        with st.form("station_group_form", border=False):
            group_system_name_input = st.text_input(
                "System name",
                key="group_system_name_input",
                placeholder="e.g. MRS Crew Operations"
            )
            group_name_input = st.text_input(
                "Group name",
                key="group_name_input",
                placeholder="e.g. Crew Change"
            )
            selected_group_stations = st.multiselect(
                "Stations in group",
                options=labels,
                key="group_station_selection"
            )
            group_marker_shape = st.selectbox(
                "Marker shape",
                options=["circle", "square", "triangle"],
                key="group_marker_shape"
            )
            group_marker_color = st.color_picker(
                "Marker color",
                "#d62728",
                key="group_marker_color"
            )
            save_group_btn = st.form_submit_button("💾 Save/Update Group", use_container_width=True)

        col_g1 = st.container()  # save feedback
        col_g2 = st.container()  # delete controls (below the form)
        with col_g1:
            if save_group_btn:
                system_name_clean = (group_system_name_input or "").strip()
                group_name_clean = (group_name_input or "").strip()
                if not system_name_clean:
                    st.warning("Enter a system name.")
                elif not group_name_clean:
                    st.warning("Enter a group name.")
                elif not selected_group_stations:
                    st.warning("Select at least one station.")
                else:
                    if system_name_clean not in st.session_state.station_group_systems:
                        st.session_state.station_group_systems[system_name_clean] = {'groups': {}}

                    st.session_state.station_group_systems[system_name_clean]['groups'][group_name_clean] = {
                        'stations': list(selected_group_stations),
                        'shape': group_marker_shape,
                        'color': group_marker_color,
                    }
                    # Ensure system is in active list
                    if system_name_clean not in st.session_state.get('active_station_group_systems', []):
                        # Just save - don't modify the bound session state variable
                        # The multiselect widget will handle it on next render
                        pass
                    
                    save_station_group_systems(st.session_state.station_group_systems)
                    st.success(f"Saved group '{group_name_clean}' in system '{system_name_clean}'.")
                    st.rerun()

        system_names = sorted(st.session_state.station_group_systems.keys())
        # Clean up active systems - keep only those that still exist
        filtered_active = [
            s for s in st.session_state.get('active_station_group_systems', [])
            if s in system_names
        ]
        if filtered_active != st.session_state.get('active_station_group_systems', []):
            st.session_state.active_station_group_systems = filtered_active

        col_sg_exp, col_sg_hint = st.columns([1, 1])
        with col_sg_exp:
            if st.button("📤 Export Station Groups", key="export_station_groups_btn", use_container_width=True):
                try:
                    with open(str(STATION_GROUPS_EXPORT_PATH), 'w') as f:
                        json.dump(st.session_state.station_group_systems, f, indent=2)
                    st.success(f"Exported → {STATION_GROUPS_EXPORT_PATH.name}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        with col_sg_hint:
            st.caption("Exports all station group systems")

        st.markdown("**Import station group systems (JSON)**")
        station_groups_files = st.file_uploader(
            "Import station groups JSON (select one or more)",
            type=["json"],
            key="import_station_groups_file",
            label_visibility="collapsed",
            accept_multiple_files=True
        )

        all_imported_station_groups = {}
        total_systems_preview = 0
        total_groups_preview = 0
        total_size_kb = 0.0

        if station_groups_files:
            for station_groups_file in station_groups_files:
                try:
                    file_bytes = station_groups_file.getvalue()
                    total_size_kb += len(file_bytes) / 1024
                    parsed_station_groups = json.loads(file_bytes)
                    normalized_station_groups = _normalize_station_group_systems(parsed_station_groups)
                    total_systems_preview += len(normalized_station_groups)
                    total_groups_preview += sum(
                        len(system_data.get('groups', {}))
                        for system_data in normalized_station_groups.values()
                        if isinstance(system_data, dict)
                    )
                    for system_name, system_data in normalized_station_groups.items():
                        all_imported_station_groups[system_name] = system_data
                except Exception:
                    st.warning(f"Could not parse {station_groups_file.name}.")

            if all_imported_station_groups:
                st.info(
                    f"📄 **{len(station_groups_files)} file(s)** — {total_size_kb:.1f} KB\n\n"
                    f"{total_systems_preview} system(s), {total_groups_preview} group(s)"
                )
                if st.button("📥 Load Station Groups", key="load_station_groups_btn", use_container_width=True):
                    merged_count = 0
                    for imported_system_name, imported_system_data in all_imported_station_groups.items():
                        if imported_system_name not in st.session_state.station_group_systems:
                            st.session_state.station_group_systems[imported_system_name] = {'groups': {}}

                        imported_groups = imported_system_data.get('groups', {}) if isinstance(imported_system_data, dict) else {}
                        for imported_group_name, imported_group_data in imported_groups.items():
                            st.session_state.station_group_systems[imported_system_name]['groups'][imported_group_name] = {
                                'stations': list(imported_group_data.get('stations', [])),
                                'shape': str(imported_group_data.get('shape', 'circle')).lower(),
                                'color': str(imported_group_data.get('color', '#d62728')),
                            }
                            merged_count += 1

                    save_station_group_systems(st.session_state.station_group_systems)
                    st.success(f"Loaded {merged_count} station group(s) from {len(station_groups_files)} file(s) into session.")
                    st.rerun()

        if system_names:
            st.multiselect(
                "Show systems on map",
                options=system_names,
                default=st.session_state.active_station_group_systems,
                key="active_station_group_systems"
            )

            with col_g2:
                delete_system = st.selectbox(
                    "System",
                    options=system_names,
                    key="delete_group_system_pick",
                    label_visibility="collapsed"
                )
                group_names = sorted(
                    st.session_state.station_group_systems.get(delete_system, {}).get('groups', {}).keys()
                )
                if group_names:
                    delete_group = st.selectbox(
                        "Group",
                        options=group_names,
                        key="delete_group_pick",
                        label_visibility="collapsed"
                    )
                    if st.button("🗑️ Delete Group", key="delete_group_btn", use_container_width=True):
                        st.session_state.station_group_systems[delete_system]['groups'].pop(delete_group, None)
                        if not st.session_state.station_group_systems[delete_system]['groups']:
                            st.session_state.station_group_systems.pop(delete_system, None)
                        save_station_group_systems(st.session_state.station_group_systems)
                        st.success(f"Deleted group '{delete_group}' from '{delete_system}'.")
                        st.rerun()

                if st.button("🧹 Delete System", key="delete_system_btn", use_container_width=True):
                    st.session_state.station_group_systems.pop(delete_system, None)
                    save_station_group_systems(st.session_state.station_group_systems)
                    st.success(f"Deleted system '{delete_system}'.")
                    st.rerun()

            st.caption("Existing systems and groups")
            for system_name in system_names:
                groups_map = st.session_state.station_group_systems.get(system_name, {}).get('groups', {})
                if not groups_map:
                    continue
                group_summaries = [
                    f"{group_name} ({len(group_data.get('stations', []))} stations, {group_data.get('shape', 'circle')})"
                    for group_name, group_data in sorted(groups_map.items())
                ]
                st.caption(f"• {system_name}: " + " | ".join(group_summaries))
        else:
            st.info("No station group systems yet.")
        
        st.divider()
        if st.button("📤 Export All", key="export_btn"):
            try:
                with open(str(EXPORT_SAVE_PATH), 'w') as f:
                    json.dump(st.session_state.committed_highlights, f, indent=2)
                st.success(f"Exported → {EXPORT_SAVE_PATH.name}")
            except Exception as e:
                st.error(f"Export failed: {e}")

        st.markdown("**Import JSON highlights**")
        import_files = st.file_uploader("Import JSON file (select one or more)", type=["json"], key="import_file", label_visibility="collapsed", accept_multiple_files=True)

        all_previews = {}
        total_import_systems = 0
        total_import_paths = 0
        total_import_kb = 0.0

        if import_files:
            for import_file in import_files:
                try:
                    file_bytes = import_file.getvalue()
                    total_import_kb += len(file_bytes) / 1024
                    preview = json.loads(file_bytes)
                    if isinstance(preview, dict):
                        total_import_systems += len(preview)
                        total_import_paths += sum(len(v) if isinstance(v, list) else 0 for v in preview.values())
                        all_previews.update(preview)
                except Exception:
                    st.warning(f"Could not parse {import_file.name}.")

            if all_previews:
                est_sec = max(0.1, total_import_kb / 800)
                st.info(
                    f"📄 **{len(import_files)} file(s)** — {total_import_kb:.1f} KB\n\n"
                    f"{total_import_systems} system(s), {total_import_paths} path(s) · est. ~{est_sec:.1f}s"
                )
                if st.button("📥 Load into session", key="import_btn"):
                    t0 = time.time()
                    file_names = [f.name for f in import_files]
                    for imported_system_name, imported_paths in all_previews.items():
                        if imported_system_name not in st.session_state.committed_highlights:
                            st.session_state.committed_highlights[imported_system_name] = []
                        existing_orders = {
                            tuple(p.get('order', []))
                            for p in st.session_state.committed_highlights[imported_system_name]
                        }
                        for path in imported_paths:
                            if isinstance(path, dict):
                                key = tuple(path.get('order', []))
                                if key not in existing_orders:
                                    st.session_state.committed_highlights[imported_system_name].append(path)
                                    existing_orders.add(key)
                    save_highlights(st.session_state.committed_highlights)
                    elapsed = time.time() - t0
                    st.session_state.last_import_info = (
                        f"✅ Imported {total_import_systems} system(s), {total_import_paths} path(s) "
                        f"from {len(import_files)} file(s) in {elapsed:.2f}s"
                    )
                    st.rerun()  # needed: forces sidebar to re-render with new committed data

        if st.session_state.get("last_import_info"):
            st.success(st.session_state.last_import_info)
            st.session_state.last_import_info = None  # clear after showing once

        st.divider()
        n_committed = sum(len(v) for v in st.session_state.committed_highlights.values())
        n_systems_committed = len(st.session_state.committed_highlights)
        
        # Calculate total coordinates for performance diagnostic
        total_coords = sum(
            len(path.get('coords_ll', []))
            for paths in st.session_state.committed_highlights.values()
            for path in paths
        )
        
        st.subheader(f"Committed Systems ({n_systems_committed} | {n_committed} paths)")
        if total_coords > 0:
            perf_color = "🟢" if total_coords < 5000 else "🟡" if total_coords < 20000 else "🔴"
            st.caption(f"{perf_color} Total coordinates: {total_coords:,} | Avg: {total_coords//max(1,n_committed):,}/path")

        col_clear_all, col_clear_hint = st.columns([1, 1])
        with col_clear_all:
            if st.button("🗑️ Clear All Systems", key="clear_all_systems", use_container_width=True):
                st.session_state.committed_highlights = {}
                save_highlights(st.session_state.committed_highlights)
                st.success("All systems cleared.")
                st.rerun()
        with col_clear_hint:
            st.caption("Removes all committed systems")

        # Display committed highlights grouped by name
        if st.session_state.committed_highlights:
            highlight_groups = st.session_state.committed_highlights
            for system_name in sorted(highlight_groups.keys()):
                paths_list = highlight_groups[system_name]
                color_indicator = paths_list[0].get('route_color', PREVIEW_COLOR) if paths_list else PREVIEW_COLOR
                
                with st.expander(f"📍 {system_name} ({len(paths_list)} paths)", expanded=False):
                    st.markdown(f"<div style='display:flex; align-items:center; gap:8px;'><div style='width:20px; height:20px; background-color:{color_indicator}; border-radius:2px;'></div><span>System Color</span></div>", unsafe_allow_html=True)

                    col_rn, col_rb = st.columns([4, 1])
                    with col_rn:
                        new_name = st.text_input("Rename system", value=system_name, key=f"rename_{system_name}", label_visibility="collapsed")
                    with col_rb:
                        if st.button("✏️", key=f"rename_btn_{system_name}", help="Rename system"):
                            new_name = (new_name or "").strip()
                            if not new_name:
                                st.warning("Name cannot be empty.")
                            elif new_name == system_name:
                                st.info("Same name — nothing changed.")
                            elif new_name in highlight_groups:
                                st.warning(f"'{new_name}' already exists.")
                            else:
                                highlight_groups[new_name] = highlight_groups.pop(system_name)
                                for p in highlight_groups[new_name]:
                                    p['highlight_name'] = new_name
                                save_highlights(st.session_state.committed_highlights)
                                st.rerun()

                    col_sj, col_sc, col_sd = st.columns([1, 1, 1])
                    with col_sj:
                        if st.button("📄 Export JSON", key=f"export_json_{system_name}"):
                            sys_export_path = _APP_DIR / f"{system_name}.json"
                            try:
                                with open(str(sys_export_path), 'w') as f:
                                    json.dump({system_name: paths_list}, f, indent=2)
                                st.success(f"Exported → {sys_export_path.name}")
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                    with col_sc:
                        st.caption("JSON only")
                    with col_sd:
                        if st.button("🗑️ Clear System", key=f"clear_system_{system_name}"):
                            if system_name in highlight_groups:
                                del highlight_groups[system_name]
                                save_highlights(st.session_state.committed_highlights)
                                st.success(f"System '{system_name}' cleared.")
                                st.rerun()

                    if len(paths_list) >= 2:
                        st.caption("Merge two paths that share a start/end station")
                        path_options = [
                            f"{idx + 1}. {(p.get('path_name') or 'Unnamed Path')}"
                            for idx, p in enumerate(paths_list)
                        ]
                        col_m1, col_m2, col_m3 = st.columns([3, 3, 2])
                        with col_m1:
                            merge_sel_a = st.selectbox(
                                "Path A",
                                options=path_options,
                                key=f"merge_a_{system_name}",
                                label_visibility="collapsed"
                            )
                        with col_m2:
                            merge_sel_b = st.selectbox(
                                "Path B",
                                options=path_options,
                                index=1 if len(path_options) > 1 else 0,
                                key=f"merge_b_{system_name}",
                                label_visibility="collapsed"
                            )
                        with col_m3:
                            if st.button("🔗 Merge", key=f"merge_btn_{system_name}"):
                                idx_a = int(merge_sel_a.split('.', 1)[0]) - 1
                                idx_b = int(merge_sel_b.split('.', 1)[0]) - 1

                                if idx_a == idx_b:
                                    st.warning("Select two different paths to merge.")
                                else:
                                    merged_path, merge_error = merge_paths_at_shared_station(
                                        paths_list[idx_a],
                                        paths_list[idx_b],
                                        system_name
                                    )
                                    if merge_error:
                                        st.warning(merge_error)
                                    else:
                                        for idx in sorted([idx_a, idx_b], reverse=True):
                                            paths_list.pop(idx)
                                        paths_list.append(merged_path)
                                        save_highlights(st.session_state.committed_highlights)
                                        st.success("Paths merged successfully.")
                                        st.rerun()
                    
                    st.divider()
                    
                    for i, path_info in enumerate(paths_list):
                        route_text = " → ".join(path_info.get('order', []))
                        dist_text = f"{path_info.get('total_km', 0.0):.1f} km"
                        station_count = _path_station_count(path_info)
                        n_coords = len(path_info.get('coords_ll', []))
                        path_key = f"{system_name}_{i}"
                        
                        # Performance indicator
                        if n_coords > 1000:
                            perf_icon = "🔴"
                            perf_hint = f" (⚠️ {n_coords:,} coords - may cause lag)"
                        elif n_coords > 500:
                            perf_icon = "🟡"
                            perf_hint = f" ({n_coords:,} coords)"
                        else:
                            perf_icon = "🟢"
                            perf_hint = f" ({n_coords:,} coords)"
                        
                        auto_label = f"{route_text} ({dist_text})"
                        current_path_name = path_info.get('path_name') or auto_label

                        col_pn, col_pe, col_pd = st.columns([5, 1, 1])
                        with col_pn:
                            new_path_name = st.text_input(
                                "Path name",
                                value=current_path_name,
                                key=f"pname_{system_name}_{i}",
                                label_visibility="collapsed"
                            )
                        with col_pe:
                            if st.button("✏️", key=f"pname_btn_{system_name}_{i}", help="Rename path"):
                                saved = (new_path_name or "").strip()
                                path_info['path_name'] = saved if saved else auto_label
                                save_highlights(st.session_state.committed_highlights)
                                st.rerun()
                        with col_pd:
                            if st.button("✕", key=f"del_{system_name}_{i}", help="Delete path"):
                                paths_list.pop(i)
                                if not paths_list:
                                    del highlight_groups[system_name]
                                save_highlights(st.session_state.committed_highlights)
                                st.rerun()
                        st.caption(f"{perf_icon} {route_text} · {dist_text} · {station_count} stations{perf_hint}")

                        col_pc1, col_pc2 = st.columns([3, 1])
                        with col_pc1:
                            edited_path_color = st.color_picker(
                                "Path color",
                                value=path_info.get('route_color', PREVIEW_COLOR),
                                key=f"path_color_{path_key}",
                            )
                        with col_pc2:
                            if st.button("🎨", key=f"path_color_btn_{path_key}", help="Apply path color"):
                                path_info['route_color'] = edited_path_color
                                save_highlights(st.session_state.committed_highlights)
                                st.success("Path color updated.")
                                st.rerun()

                        # Manual station markers along this path
                        pending_key = f"pending_path_stations_{path_key}"
                        station_color_key = f"station_marker_color_{path_key}"

                        if pending_key not in st.session_state:
                            st.session_state[pending_key] = []

                        confirmed_markers = list(path_info.get('selected_station_markers', []))
                        confirmed_labels = {
                            m.get('label')
                            for m in confirmed_markers
                            if isinstance(m, dict) and m.get('label')
                        }

                        pending_labels = [
                            label for label in st.session_state[pending_key]
                            if label not in confirmed_labels
                        ]
                        st.session_state[pending_key] = pending_labels

                        stations_candidates = stations_along_path(path_info.get('coords_ll', []), stations_lookup, tolerance_m=150.0)
                        station_lookup = {s['label']: s for s in stations_candidates}
                        available_labels = [
                            s['label'] for s in stations_candidates
                            if s['label'] not in confirmed_labels and s['label'] not in pending_labels
                        ]

                        st.caption("Add specific stations along this path")
                        col_sa, col_sb = st.columns([4, 1])
                        with col_sa:
                            if available_labels:
                                station_to_add = st.selectbox(
                                    "Stations along path",
                                    options=available_labels,
                                    key=f"station_pick_{path_key}",
                                    label_visibility="collapsed"
                                )
                            else:
                                station_to_add = None
                                st.caption("No additional stations available for this path")
                        with col_sb:
                            if st.button("➕ Add", key=f"add_station_{path_key}", disabled=station_to_add is None):
                                st.session_state[pending_key].append(station_to_add)
                                st.rerun()

                        if pending_labels:
                            st.caption("Pending stations: " + " | ".join(pending_labels))
                            default_marker_color = path_info.get('route_color', PREVIEW_COLOR)
                            marker_color = st.color_picker(
                                "Marker color",
                                default_marker_color,
                                key=station_color_key
                            )

                            col_sc1, col_sc2 = st.columns(2)
                            with col_sc1:
                                if st.button("✅ Confirm Stations", key=f"confirm_stations_{path_key}", use_container_width=True):
                                    new_markers = []
                                    for label in pending_labels:
                                        p = station_lookup.get(label)
                                        if p:
                                            new_markers.append({
                                                'label': label,
                                                'lat': p['lat'],
                                                'lon': p['lon'],
                                                'color': marker_color,
                                            })

                                    if new_markers:
                                        if 'selected_station_markers' not in path_info:
                                            path_info['selected_station_markers'] = []
                                        path_info['selected_station_markers'].extend(new_markers)
                                        st.session_state[pending_key] = []
                                        save_highlights(st.session_state.committed_highlights)
                                        st.success(f"Added {len(new_markers)} station marker(s).")
                                        st.rerun()
                            with col_sc2:
                                if st.button("Clear Pending", key=f"clear_pending_{path_key}", use_container_width=True):
                                    st.session_state[pending_key] = []
                                    st.rerun()

                        confirmed_markers = list(path_info.get('selected_station_markers', []))
                        if confirmed_markers:
                            st.caption("Confirmed markers")
                            marker_options = []
                            for marker_idx, marker in enumerate(confirmed_markers):
                                if not isinstance(marker, dict):
                                    continue
                                marker_label = marker.get('label', f"Marker {marker_idx + 1}")
                                marker_options.append(f"{marker_idx + 1}. {marker_label}")

                            if marker_options:
                                col_rm1, col_rm2, col_rm3 = st.columns([3, 1, 1])
                                with col_rm1:
                                    marker_to_remove = st.selectbox(
                                        "Remove marker",
                                        options=marker_options,
                                        key=f"remove_marker_pick_{path_key}",
                                        label_visibility="collapsed"
                                    )
                                with col_rm2:
                                    if st.button("🗑️ Remove", key=f"remove_marker_btn_{path_key}"):
                                        if marker_to_remove:
                                            marker_idx = int(marker_to_remove.split('.', 1)[0]) - 1
                                            if 0 <= marker_idx < len(path_info.get('selected_station_markers', [])):
                                                path_info['selected_station_markers'].pop(marker_idx)
                                                save_highlights(st.session_state.committed_highlights)
                                                st.success("Marker removed.")
                                                st.rerun()
                                with col_rm3:
                                    if st.button("🧹 Clear All", key=f"clear_all_markers_btn_{path_key}"):
                                        path_info['selected_station_markers'] = []
                                        save_highlights(st.session_state.committed_highlights)
                                        st.success("All markers cleared for this path.")
                                        st.rerun()
                        
                        # Offer simplification for high-coordinate paths
                        if n_coords > 500:
                            st.warning(f"⚠️ This path has {n_coords:,} coordinates. Consider simplification for better performance.")
                            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
                            with col_s1:
                                simplify_factor = st.slider(
                                    "Reduce to points:",
                                    min_value=50,
                                    max_value=min(500, n_coords),
                                    value=min(200, n_coords // 2),
                                    step=50,
                                    key=f"simplify_slider_{system_name}_{i}",
                                    help="Reduce coordinate density to improve performance"
                                )
                            with col_s2:
                                if st.button("💾 Backup", key=f"backup_btn_{system_name}_{i}", help="Export backup before simplifying"):
                                    backup_path = _APP_DIR / f"{system_name}_path{i}_backup.json"
                                    try:
                                        with open(str(backup_path), 'w') as f:
                                            json.dump({f"{system_name}_backup": [path_info]}, f, indent=2)
                                        st.success(f"Backed up → {backup_path.name}")
                                    except Exception as e:
                                        st.error(f"Backup failed: {e}")
                            with col_s3:
                                if st.button("⚡ Simplify", key=f"simplify_btn_{system_name}_{i}", help="⚠️ PERMANENT - Cannot undo!"):
                                    coords = path_info['coords_ll']
                                    if len(coords) > simplify_factor:
                                        # Auto-backup before simplification
                                        backup_path = _APP_DIR / f"{system_name}_path{i}_backup_{int(time.time())}.json"
                                        backup_note = f"Backup: {backup_path.name}"
                                        try:
                                            with open(str(backup_path), 'w') as f:
                                                json.dump({f"{system_name}_original": [path_info.copy()]}, f, indent=2)
                                        except (OSError, TypeError, ValueError) as e:
                                            backup_note = f"⚠️ Backup failed: {e}"  # simplify anyway, as before

                                        path_info['coords_ll'] = _simplify_latlon(coords, max_points=simplify_factor)
                                        save_highlights(st.session_state.committed_highlights)
                                        st.success(f"✅ Reduced from {n_coords:,} to {len(path_info['coords_ll']):,} coords | {backup_note}")
                                        st.rerun()
                            st.caption("⚠️ Simplification is PERMANENT. Original coordinates cannot be restored. Backup first!")
        else:
            st.info("No committed highlights yet. Compute and commit a route above.")


    # Main map area
    st.subheader("Interactive Map")
    map_zoom = 6  # Initial zoom level

    m = folium.Map(
        location=map_center,
        zoom_start=int(map_zoom),
        height=700,
        zoom_snap=0.25,
        zoom_delta=0.25,
        wheel_px_per_zoom_level=120,
        prefer_canvas=True,  # draw vectors on one canvas instead of thousands of SVG nodes
    )
    Fullscreen(position='topleft', title='Fullscreen', title_cancel='Exit fullscreen', force_separate_button=True).add_to(m)
    _ZoomDisplay().add_to(m)

    folium.TileLayer(
        tiles='https://{s}.tiles.openrailwaymap.org/gauge/{z}/{x}/{y}.png',
        attr=_ORM_ATTR,
        name='Rail Gauge (OpenRailwayMap)',
        overlay=True,
        control=True,
        show=False,
        min_zoom=2,
        max_zoom=19,
        opacity=0.8,
        subdomains='abc',
        bounds=_BR_BOUNDS,
    ).add_to(m)

    all_rails_layer  = folium.FeatureGroup(name="All Brazil Rails", show=False)
    stations_layer   = folium.FeatureGroup(name="All Stations", show=False)
    route_layer      = folium.FeatureGroup(name="Preview Route Track", show=True)
    route_stations_layer = folium.FeatureGroup(name="Preview Stations", show=True)

    station_group_layers = []
    active_station_group_systems = st.session_state.get('active_station_group_systems', [])
    station_group_systems = st.session_state.get('station_group_systems', {})
    if not isinstance(active_station_group_systems, list):
        active_station_group_systems = []

    # Static layers (all rails, operators, all stations) are loaded by URL from
    # versioned files the browser caches — see _RemoteGeoJson.  Hidden layers
    # cost nothing until switched on in the layer control.
    static_urls = d['static_layer_urls']
    _RemoteGeoJson(
        static_urls['rails_all'], d['rails_all_geojson'], 'lines',
        {'color': '#ff7f0e', 'weight': BASE_RAIL_WEIGHT, 'opacity': 0.85},
    ).add_to(all_rails_layer)

    # Per-operator rail layers — only render operators selected in sidebar
    active_operators = st.session_state.get('active_operator_layers', ['MRS'])
    operator_layers = []
    operator_geojsons = d.get('operator_geojsons', {})
    for op_name in sorted(RAIL_OPERATORS.keys()):
        if op_name not in active_operators:
            continue
        op_geojson = operator_geojsons.get(op_name)
        if not op_geojson or not op_geojson.get('features'):
            continue
        op_layer = folium.FeatureGroup(name=f"🚂 {op_name} Rails", show=True)
        _RemoteGeoJson(
            static_urls.get(f"op_{op_name}"), op_geojson, 'lines',
            {'color': RAIL_OPERATORS[op_name]['color'], 'weight': BASE_RAIL_WEIGHT, 'opacity': 0.85},
        ).add_to(op_layer)
        operator_layers.append(op_layer)

    # All stations (off by default): circle markers + labels shown from STATION_LABEL_MIN_ZOOM
    _RemoteGeoJson(
        static_urls['stations'], d['stations_geojson'], 'stations',
        {'radius': 4, 'color': '#0077cc', 'fill': True, 'fillOpacity': 0.75},
    ).add_to(stations_layer)

    stations_by_label = d['stations_by_label']
    group_labels_layout_state = []
    for system_name in active_station_group_systems:
        system_data = station_group_systems.get(system_name, {})
        groups_map = system_data.get('groups', {}) if isinstance(system_data, dict) else {}
        if not isinstance(groups_map, dict) or not groups_map:
            continue

        system_layer = folium.FeatureGroup(name=f"Station System: {system_name}", show=True)
        for group_name, group_data in sorted(groups_map.items()):
            if not isinstance(group_data, dict):
                continue
            group_labels = set(group_data.get('stations', []))
            group_shape = group_data.get('shape', 'circle')
            group_color = group_data.get('color', '#d62728')

            for station_label in group_labels:
                station = stations_by_label.get(station_label)
                if not station:
                    continue
                add_station_shape_marker(
                    system_layer,
                    lat=station['lat'],
                    lon=station['lon'],
                    label=station_label,
                    shape=group_shape,
                    color=group_color,
                    size=8,
                    font_size=LABEL_FONT_SIZE,
                    label_layout_state=group_labels_layout_state,
                )

        station_group_layers.append(system_layer)

    # Run routing on button click
    if run_btn:
        t_start = time.time()
        with st.spinner("Computing route..."):
            order = [origin] + list(vias) + [destination]

            pts = []
            for lab in order:
                row = stations_lookup.loc[lab]
                pts.append((float(row.geometry.x), float(row.geometry.y)))

            progress_bar.progress(10)
            status_txt.text("Loading routing graph...")
            t_graph_start = time.time()
            graph = get_rail_graph(rails_all_m, data_version_key)
            t_graph = time.time() - t_graph_start

            progress_bar.progress(60)
            status_txt.text("Computing route...")
            t_route_start = time.time()
            try:
                result = routing.route(graph, pts, ROUTING_MODE_IDS[routing_mode], penalty)
            except routing.NoPathError as e:
                progress_bar.progress(0)
                status_txt.text("")
                msg = f"No path found for segment {order[e.segment_index]} → {order[e.segment_index + 1]}"
                if routing_mode == STRICT_MRS:
                    msg += (" — Strict MRS only uses track coded as MRS (11); some track in the"
                            " shapefile has no operator code. Try Prefer MRS.")
                st.error(msg)
                return
            t_route = time.time() - t_route_start

            total_km = result['total_km']
            coords_ll = result['coords_ll']

            # Draw route on map
            preview_station_points = []
            for lab in order:
                row = stations_lookup.loc[lab]
                lon, lat = to_4326.transform(float(row.geometry.x), float(row.geometry.y))
                preview_station_points.append({
                    'label': lab,
                    'lat': lat,
                    'lon': lon
                })

            st.session_state.preview_data = {
                'coords_ll': coords_ll,
                'order': order,
                'total_km': total_km,
                'route_color': color,
                'highlight_name': highlight_name,
                'path_name': path_name.strip(),
                'preview_station_points': preview_station_points,
                'station_count': len(preview_station_points),
            }

            progress_bar.progress(100)
            t_total = time.time() - t_start

            # Show performance breakdown
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Graph Load", f"{t_graph:.2f}s")
            with perf_col2:
                st.metric("Routing", f"{t_route:.2f}s")
            with perf_col3:
                st.metric("Total Time", f"{t_total:.2f}s")

            status_txt.text(f"✅ Route complete: {total_km:.1f} km")

            # Show route info
            station_count = len(preview_station_points)
            st.success(f"**Route:** {' → '.join(order)}")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"**Total Distance:** {total_km:.1f} km")
            with info_col2:
                st.info(f"**Stations in path:** {station_count}")

            # Show code breakdown
            st.subheader("Rail Code Breakdown (km)")
            seg_diag = []
            for i, (km, codes) in enumerate(zip(result['segment_km'], result['segment_codes'])):
                codes = {k: round(v, 1) for k, v in codes.items()}
                seg_diag.append({'Segment': f"{order[i]} → {order[i+1]}", 'Distance (km)': f"{km:.1f}", 'Codes': str(codes)})
            st.table(pd.DataFrame(seg_diag))

    # Commit current preview as a persistent highlight
    if commit_btn:
        preview_data = st.session_state.get('preview_data')
        if not preview_data:
            st.warning("Compute a preview first, then click Commit Highlight.")
        else:
            system_name = preview_data.get('highlight_name', 'Committed Route')
            if system_name not in st.session_state.committed_highlights:
                st.session_state.committed_highlights[system_name] = []

            auto_label = " → ".join(preview_data.get('order', []))
            committed_path_name = preview_data.get('path_name', '').strip() or auto_label
            # Build payload with updated highlight_name and path_name
            temp_data = dict(preview_data)
            temp_data['highlight_name'] = system_name
            temp_data['path_name'] = committed_path_name
            payload = _make_path_payload(temp_data)
            st.session_state.committed_highlights[system_name].append(payload)
            save_highlights(st.session_state.committed_highlights)
            st.success(
                f"Committed: **{system_name}** · {payload['station_count']} stations · "
                f"{payload.get('total_km', 0.0):.1f} km → {HIGHLIGHTS_SAVE_PATH.name}"
            )
            st.rerun()

    # Draw persisted preview layers
    preview_data = st.session_state.get('preview_data')
    if preview_data:
        folium.PolyLine(
            preview_data['coords_ll'],
            color=preview_data.get('route_color', PREVIEW_COLOR),
            weight=HIGHLIGHT_WEIGHT + 2,
            opacity=HIGHLIGHT_OPACITY,
            popup=preview_data.get('highlight_name', 'Route Preview')
        ).add_to(route_layer)

        for p in preview_data.get('preview_station_points', []):
            folium.CircleMarker(
                location=[p['lat'], p['lon']],
                radius=6,
                color=preview_data.get('route_color', PREVIEW_COLOR),
                fill=True,
                fill_opacity=1.0,
                popup=p['label']
            ).add_to(route_stations_layer)

        pts = preview_data.get('preview_station_points', [])
        if pts:
            preview_labels_layout = []  # Layout state for alternating labels
            endpoint_idxs = [0] if len(pts) == 1 else [0, len(pts) - 1]
            for idx in endpoint_idxs:
                p = pts[idx]
                folium.CircleMarker(
                    location=[p['lat'], p['lon']],
                    radius=10,
                    color="#ffffff",
                    weight=2,
                    fill=True,
                    fill_color=preview_data.get('route_color', PREVIEW_COLOR),
                    fill_opacity=1.0,
                    tooltip=p['label'],
                    popup=p['label']
                ).add_to(route_stations_layer)
                # Add station name label with alternating offset
                margin_top, translate_x = compute_label_offset(p['lat'], p['lon'], layout_state=preview_labels_layout)
                _add_label_marker(route_stations_layer, p['lat'], p['lon'], p['label'],
                                margin_top=margin_top, translate_x_pct=translate_x)

    # Draw committed highlights grouped by system.
    # FeatureGroup objects must be freshly created each rerun — they hold a
    # reference to the parent map and cannot be safely reused across reruns.
    # The data (coords_ll, colors, etc.) is already plain Python so this loop
    # is fast even with many committed paths.
    # PERFORMANCE: Only render if toggle is enabled
    if render_committed:
        # Global alternation so left/right pattern is consistent across systems
        route_labels_layout = []
        for system_name in sorted(st.session_state.get('committed_highlights', {}).keys()):
            system_layer = folium.FeatureGroup(name=f"System: {system_name}", show=False)  # Hidden by default for performance
            # Per-system dedup: prevents duplicate markers across paths within
            # the same system, but every system keeps its own complete markers
            # so toggling a single system always shows all its stations.
            rendered_station_keys = set()
            for h in st.session_state.committed_highlights[system_name]:
                # Optionally simplify coordinates for better performance
                coords = h.get('coords_ll', [])
                if simplify_coords and len(coords) > 100:
                    coords = _simplify_latlon(coords, tolerance=HIGHLIGHT_DISPLAY_TOLERANCE_DEG)
                if coords:
                    folium.PolyLine(
                        coords,
                        color=h.get('route_color', PREVIEW_COLOR),
                        weight=HIGHLIGHT_WEIGHT,
                        opacity=0.9,
                        popup=f"{h.get('path_name') or h.get('highlight_name', 'Route')} ({h.get('total_km', 0.0):.1f} km)"
                    ).add_to(system_layer)
                pts = h.get('preview_station_points', [])
                if pts:
                    endpoint_idxs = [0] if len(pts) == 1 else [0, len(pts) - 1]
                    for idx in endpoint_idxs:
                        p = pts[idx]
                        station_key = _station_render_key(p['lat'], p['lon'])
                        if station_key in rendered_station_keys:
                            continue
                        rendered_station_keys.add(station_key)
                        folium.CircleMarker(
                            location=[p['lat'], p['lon']],
                            radius=8,
                            color=h.get('route_color', PREVIEW_COLOR),
                            weight=2,
                            fill=True,
                            fill_color="#ffffff",
                            fill_opacity=1.0,
                            tooltip=f"{system_name}: {p['label']}",
                            popup=p['label']
                        ).add_to(system_layer)
                        # Add label with alternating offset
                        margin_top, translate_x = compute_label_offset(p['lat'], p['lon'], layout_state=route_labels_layout)
                        _add_label_marker(system_layer, p['lat'], p['lon'], p['label'],
                                        margin_top=margin_top, translate_x_pct=translate_x)

                for station_marker in h.get('selected_station_markers', []):
                    if not isinstance(station_marker, dict):
                        continue
                    lat = station_marker.get('lat')
                    lon = station_marker.get('lon')
                    label = station_marker.get('label', 'Selected station')
                    marker_color = station_marker.get('color', h.get('route_color', PREVIEW_COLOR))
                    if lat is None or lon is None:
                        continue
                    station_key = _station_render_key(lat, lon)
                    if station_key in rendered_station_keys:
                        continue
                    rendered_station_keys.add(station_key)

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=7,
                        color=marker_color,
                        weight=2,
                        fill=True,
                        fill_color=marker_color,
                        fill_opacity=0.95,
                        tooltip=f"{system_name}: {label}",
                        popup=label
                    ).add_to(system_layer)
                    # Add label with alternating offset (shares layout state with route endpoints)
                    margin_top, translate_x = compute_label_offset(lat, lon, layout_state=route_labels_layout)
                    _add_label_marker(system_layer, lat, lon, label,
                                    margin_top=margin_top, translate_x_pct=translate_x)
            system_layer.add_to(m)

    # Draw station finder marker if selected
    highlighted_station = st.session_state.get('highlighted_station')
    if highlighted_station:
        try:
            station_row = stations_lookup.loc[highlighted_station]
            # Convert from EPSG:3857 (meters) to EPSG:4326 (lat/lon) for Folium
            lon4326, lat4326 = to_4326.transform(station_row.geometry.x, station_row.geometry.y)
            
            marker_layer = folium.FeatureGroup(name="🎯 Selected Station", show=True)
            folium.CircleMarker(
                location=[lat4326, lon4326],
                radius=20,
                color="#FF0000",
                weight=3,
                fill=True,
                fill_color="#FF0000",
                fill_opacity=0.5,
                popup=highlighted_station,
                tooltip=f"📍 {highlighted_station}"
            ).add_to(marker_layer)
            folium.Marker(
                location=[lat4326, lon4326],
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                popup=highlighted_station,
                tooltip=highlighted_station
            ).add_to(marker_layer)
            marker_layer.add_to(m)
        except Exception as e:
            st.warning(f"Station marker error: {e}")

    all_rails_layer.add_to(m)
    for op_layer in operator_layers:
        op_layer.add_to(m)
    stations_layer.add_to(m)
    route_layer.add_to(m)
    route_stations_layer.add_to(m)
    for group_layer in station_group_layers:
        group_layer.add_to(m)
    
    # Keep layer control expanded so system visibility is a direct click toggle
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # Add comprehensive CSS/JS for layer control scrolling
    m.get_root().html.add_child(folium.Element(_LAYER_SCROLL_FIX))

    # Display map
    st_folium(m, width=1200, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
