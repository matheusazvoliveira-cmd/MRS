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
import json
import time
import sys

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import scipy.spatial as scipy_spatial
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, snap, nearest_points
from pyproj import Transformer
import math

import folium
from folium.plugins import Fullscreen
from folium.elements import JSCSSMixin
from streamlit_folium import st_folium
from jinja2 import Template


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

FILTER_COLUMN = "CodigoFerr"
MRS_CODE = 11

STATION_NAME_FIELD = "NomeEstaca"
STATION_ID_FIELD = None

SOURCE_CRS_IF_MISSING = "EPSG:4674"

STRICT_MRS = "Strict MRS (11 only)"
PREFER_MRS = "Prefer MRS (penalize non-11)"
ALLOW_ALL = "Allow all (shortest distance)"

NON_MRS_PENALTY_DEFAULT = 10.0

# Topology repair
ROUND_M = 25
SNAP_ENDPOINTS = True
SNAP_TOLERANCE_M = 75

# Intersection noding
NODE_INTERSECTIONS_DEFAULT = True
ASSIGN_CODIGO_MAX_DIST_M = 10.0

# Waypoint snapping
SNAP_WAYPOINTS_TO_EDGES_DEFAULT = True
EDGE_SNAP_SEARCH_RADIUS_M = 2000

# Corridor constraint
CONSTRAIN_CORRIDOR_DEFAULT = True
CORRIDOR_BUFFER_KM_DEFAULT = 20.0

# Styles
BASE_RAIL_COLOR = "#2ca02c"
BASE_RAIL_WEIGHT = 3

PREVIEW_COLOR = "#1f77b4"

HIGHLIGHT_WEIGHT = 5
HIGHLIGHT_OPACITY = 1.0

LABEL_FONT_SIZE = 20  # px — single source of truth for all station label text

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


@st.cache_data
def load_data():
    """Load shapefiles — raises on failure so the caller can show the error in context."""
    if not RAILS_SHP_PATH.exists():
        raise FileNotFoundError(f"Rails shapefile not found: {RAILS_SHP_PATH}")
    if not STATIONS_SHP_PATH.exists():
        raise FileNotFoundError(f"Stations shapefile not found: {STATIONS_SHP_PATH}")

    rails_all = gpd.read_file(RAILS_SHP_PATH)
    stations_all = gpd.read_file(STATIONS_SHP_PATH)

    # Set CRS if missing
    if rails_all.crs is None and SOURCE_CRS_IF_MISSING:
        rails_all = rails_all.set_crs(SOURCE_CRS_IF_MISSING)
    if stations_all.crs is None and SOURCE_CRS_IF_MISSING:
        stations_all = stations_all.set_crs(SOURCE_CRS_IF_MISSING)

    # Normalize rail codes
    rails_all[FILTER_COLUMN] = pd.to_numeric(rails_all[FILTER_COLUMN], errors='coerce').round().astype('Int64')

    return rails_all, stations_all


def prepare_stations(stations_all):
    """Extract point stations and build labels with 3-letter codes."""
    stations_m = stations_all.to_crs(epsg=3857)
    stations_pts_m = stations_m[stations_m.geometry.type == 'Point'].copy()

    stations_near = stations_pts_m.copy()

    # Name series
    if STATION_NAME_FIELD in stations_near.columns:
        names = stations_near[STATION_NAME_FIELD].astype(str).str.strip()
    else:
        names = stations_near.index.astype(str)

    # 3-letter code
    if 'CodigoTres' in stations_near.columns:
        stations_near['station_code'] = stations_near['CodigoTres'].astype(str).str.upper().fillna('')
    elif 'Codigo_Tres' in stations_near.columns:
        stations_near['station_code'] = stations_near['Codigo_Tres'].astype(str).str.upper().fillna('')
    else:
        stations_near['station_code'] = ''

    # Station label
    if STATION_ID_FIELD and STATION_ID_FIELD in stations_near.columns:
        ids = stations_near[STATION_ID_FIELD].astype(str)
        stations_near['station_label'] = names + ' [' + ids + ']'
    else:
        dup = names.duplicated(keep=False)
        stations_near['station_label'] = names
        stations_near.loc[dup, 'station_label'] = names[dup] + ' (idx=' + stations_near.loc[dup].index.astype(str) + ')'

    # Append code to label (vectorized — avoids row-by-row apply)
    _has_code = stations_near['station_code'].astype(bool)
    stations_near.loc[_has_code, 'station_label'] = (
        stations_near.loc[_has_code, 'station_label'] + ' [' + stations_near.loc[_has_code, 'station_code'] + ']'
    )

    stations_near_ll = stations_near.to_crs(epsg=4326)
    stations_lookup = stations_near.set_index('station_label')
    labels = sorted(list(stations_lookup.index))

    return stations_near, stations_near_ll, stations_lookup, labels


def stations_along_path(coords_ll, stations_lookup, tolerance_m=150.0):
    """Find stations that lie near a route polyline (meters in EPSG:3857)."""
    if not coords_ll or len(coords_ll) < 2:
        return []

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


def compute_label_offset(lat, lon, layout_state=None, collision_radius_m=5000.0, label_text=None):
    """Compute vertical/horizontal offsets to reduce collisions among nearby labels.
    
    Checks session state for custom label offsets first, then falls back to automatic collision avoidance.
    """
    # Check for manual override
    if label_text and 'custom_label_offsets' in st.session_state:
        custom = st.session_state.custom_label_offsets.get(label_text)
        if custom is not None:
            return custom  # (margin_top, translate_x_pct) tuple
    
    if layout_state is None:
        return 8, -50

    x, y = to_3857.transform(float(lon), float(lat))
    radius_sq = float(collision_radius_m) ** 2
    overlaps = 0
    for entry in layout_state:
        dx = x - entry['x']
        dy = y - entry['y']
        if (dx * dx + dy * dy) <= radius_sq:
            overlaps += 1

    # Expanded offset pattern with more varied positions
    # (margin_top_px, translate_x_pct) — spreads labels in radial pattern
    offset_pattern = [
        (8, -50),      # center below (default)
        (8, -100),     # far left
        (8, 0),        # far right
        (25, -50),     # center below, further down
        (25, -100),    # far left, further down
        (25, 0),       # far right, further down
        (-10, -50),    # center above
        (-10, -100),   # far left above
        (-10, 0),      # far right above
        (40, -50),     # center, even further down
        (40, -120),    # extra far left down
        (40, 20),      # extra far right down
        (-25, -50),    # center, further above
        (-25, -100),   # far left, further above
        (-25, 0),      # far right, further above
        (8, -75),      # mid-left
        (8, -25),      # mid-right
        (25, -75),     # mid-left, further down
        (25, -25),     # mid-right, further down
        (55, -50),     # very far down center
        (55, -90),     # very far down left
        (55, -10),     # very far down right
    ]
    margin_top, translate_x_pct = offset_pattern[overlaps % len(offset_pattern)]
    layout_state.append({'x': x, 'y': y})
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
        label_text=label,
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
                f"<div style='font-size:{font_size}px;font-weight:bold;"
                f"background-color:white;color:black;padding:2px 4px;"
                f"border-radius:2px;border:1px solid #999;"
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
    }


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
    }
    return merged_path, None


def edge_cost(length, codigo, routing_mode, penalty_mult):
    """Compute edge traversal cost, applying penalty for non-MRS segments."""
    if routing_mode == PREFER_MRS and codigo is not None and codigo is not pd.NA and codigo != MRS_CODE:
        return length * float(penalty_mult)
    return length


def build_graph(points_xy, routing_mode, penalty_mult, rails_all_m, node_intersections=True,
                constrain_corridor=True, corridor_buffer_m=20_000.0, progress_cb=None):
    """Build graph from rail network."""

    def cb(msg):
        if progress_cb:
            progress_cb(msg)

    cb("Extracting bounding box...")
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    margin = 200_000
    minx, miny = min(xs) - margin, min(ys) - margin
    maxx, maxy = max(xs) + margin, max(ys) + margin

    rails = rails_all_m.cx[minx:maxx, miny:maxy].copy()

    if constrain_corridor:
        cb("Applying corridor constraint...")
        centerline = LineString(points_xy)
        corridor = centerline.buffer(float(corridor_buffer_m))
        rails = rails[rails.geometry.intersects(corridor)]

    if routing_mode == STRICT_MRS:
        rails = rails[rails[FILTER_COLUMN] == MRS_CODE].copy()

    cb(f"Rails in bbox/corridor: {len(rails)} | exploding...")
    rails = rails.explode(index_parts=False).reset_index(drop=True)
    rails = rails[rails.geometry.notnull()]
    rails = rails[rails.geometry.type == 'LineString'].copy()

    if len(rails) == 0:
        return None, None, None, None

    if SNAP_ENDPOINTS:
        cb("Snapping linework...")
        u = unary_union(list(rails.geometry.values))
        rails['geometry'] = rails.geometry.apply(lambda g: snap(g, u, SNAP_TOLERANCE_M))

    if node_intersections:
        cb("Noding intersections...")
        rails = node_and_assign_codigo(rails)

    cb("Building graph...")

    def round_xy(xy, r=ROUND_M):
        return (round(xy[0] / r) * r, round(xy[1] / r) * r)

    node_index = {}
    node_coords = []

    def get_node_id(xy):
        xy_r = round_xy(xy)
        if xy_r not in node_index:
            node_index[xy_r] = len(node_coords)
            node_coords.append(xy_r)
        return node_index[xy_r]

    G = nx.Graph()
    edge_rows = []

    # Pre-resolve penalty flag outside the hot loop so the inner iterations
    # do a simple boolean check instead of a string comparison + pd.notna()
    _apply_penalty = (routing_mode == PREFER_MRS)
    _penalty_f = float(penalty_mult)
    _hypot = math.hypot  # local alias avoids module lookup per iteration

    # Direct array access — avoids pandas iterrows() overhead
    _rail_geoms = rails.geometry.values
    _rail_codigos = rails[FILTER_COLUMN].values
    for i in range(len(_rail_geoms)):
        codigo = _rail_codigos[i]
        coords = _rail_geoms[i].coords[:]
        for j in range(len(coords) - 1):
            a, b = coords[j], coords[j + 1]
            u_id = get_node_id(a)
            v_id = get_node_id(b)
            length_m = _hypot(b[0] - a[0], b[1] - a[1])

            # Inlined edge_cost — avoids function-call overhead per segment
            if _apply_penalty and codigo is not None and codigo is not pd.NA and codigo != MRS_CODE:
                w = length_m * _penalty_f
            else:
                w = length_m

            if G.has_edge(u_id, v_id):
                if w < G[u_id][v_id]['weight']:
                    G[u_id][v_id].update({'weight': w, 'length_m': length_m, 'codigo': codigo})
            else:
                G.add_edge(u_id, v_id, weight=w, length_m=length_m, codigo=codigo)

            edge_rows.append({'u': u_id, 'v': v_id, 'geometry': LineString([a, b]), 'codigo': codigo, 'length_m': length_m})

    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry='geometry', crs='EPSG:3857')
    edges_sidx = edges_gdf.sindex

    node_xy = np.array(node_coords)
    tree = scipy_spatial.KDTree(node_xy)

    cb(f"Graph ready: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    return G, node_coords, tree, (edges_gdf, edges_sidx)


def node_and_assign_codigo(rails_sub_m):
    """Noding and codigo assignment."""
    merged = unary_union(list(rails_sub_m.geometry.values))
    noded = gpd.GeoSeries([merged], crs=rails_sub_m.crs).explode(index_parts=False)
    segs = gpd.GeoDataFrame(geometry=noded).reset_index(drop=True)
    segs = segs[segs.geometry.notnull()]
    segs = segs[segs.geometry.type == 'LineString'].copy()

    sidx = rails_sub_m.sindex
    codigos = []

    for geom in segs.geometry.values:
        cand_idx = list(sidx.intersection(geom.bounds))
        if not cand_idx:
            codigos.append(pd.NA)
            continue
        cand = rails_sub_m.iloc[cand_idx]
        dists = cand.distance(geom)
        j = int(dists.idxmin())
        min_dist = float(dists.loc[j])
        if min_dist <= ASSIGN_CODIGO_MAX_DIST_M:
            codigos.append(cand.loc[j, FILTER_COLUMN])
        else:
            codigos.append(pd.NA)

    segs[FILTER_COLUMN] = pd.array(codigos, dtype='Int64')
    return segs


def insert_waypoint_node_on_edge(G, node_coords, edges_gdf, edges_sidx, x, y, routing_mode, penalty_mult):
    """Snap waypoint to nearest edge."""
    pt = Point(x, y)
    r = EDGE_SNAP_SEARCH_RADIUS_M

    cand_idx = list(edges_sidx.intersection((x - r, y - r, x + r, y + r)))
    if not cand_idx:
        return None, None, None

    cand = edges_gdf.iloc[cand_idx].copy()
    cand['dist'] = cand.geometry.distance(pt)
    best = cand.loc[cand['dist'].idxmin()]
    if best['dist'] > r:
        return None, None, None

    u = int(best['u'])
    v = int(best['v'])
    line = best.geometry
    codigo = best.get('codigo')

    snapped_pt = nearest_points(line, pt)[0]
    sx, sy = float(snapped_pt.x), float(snapped_pt.y)

    new_id = len(node_coords)
    node_coords.append((sx, sy))

    if G.has_edge(u, v):
        G.remove_edge(u, v)

    seg1 = LineString([node_coords[u], (sx, sy)])
    seg2 = LineString([(sx, sy), node_coords[v]])
    len1, len2 = seg1.length, seg2.length

    G.add_edge(u, new_id, weight=edge_cost(len1, codigo, routing_mode, penalty_mult), length_m=len1, codigo=codigo)
    G.add_edge(new_id, v, weight=edge_cost(len2, codigo, routing_mode, penalty_mult), length_m=len2, codigo=codigo)

    return new_id, float(best['dist']), codigo


def stitch_paths(paths):
    """Concatenate path segments."""
    out = []
    for i, p in enumerate(paths):
        if i == 0:
            out.extend(p)
        else:
            out.extend(p[1:])
    return out


def nodes_to_latlon(node_coords, path_nodes):
    """Convert node IDs to (lat, lon) coordinates."""
    coords = []
    for n in path_nodes:
        x, y = node_coords[n]
        lon, lat = to_4326.transform(x, y)
        coords.append((lat, lon))
    return coords


def path_length_km(G, path_nodes):
    """Total path distance in km."""
    return sum(float(G[u][v].get('length_m', 0.0)) for u, v in zip(path_nodes[:-1], path_nodes[1:])) / 1000.0


def codigo_breakdown_km(G, path_nodes):
    """Distance breakdown by rail code."""
    by = {}
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        codigo = G[u][v].get('codigo')
        by[codigo] = by.get(codigo, 0.0) + float(G[u][v].get('length_m', 0.0)) / 1000.0
    return dict(sorted(by.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def iter_lines(geom):
    """Yield LineString parts from LineString/MultiLineString geometry."""
    if geom is None:
        return
    gtype = geom.geom_type
    if gtype == 'LineString':
        yield geom
    elif gtype == 'MultiLineString':
        for part in geom.geoms:
            yield part


def _gdf_to_minimal_geojson(gdf):
    """Convert GeoDataFrame geometries to a minimal GeoJSON FeatureCollection.

    Strips all attribute columns — much smaller than gdf.to_json().
    Used to render rail layers as a single folium.GeoJson layer
    instead of N individual PolyLine objects.
    """
    features = []
    for geom in gdf.geometry.values:
        if geom is None:
            continue
        gtype = geom.geom_type
        if gtype == 'LineString':
            features.append({
                'type': 'Feature',
                'geometry': geom.__geo_interface__,
                'properties': {},
            })
        elif gtype == 'MultiLineString':
            for part in geom.geoms:
                features.append({
                    'type': 'Feature',
                    'geometry': part.__geo_interface__,
                    'properties': {},
                })
    return {'type': 'FeatureCollection', 'features': features}


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
    if 'custom_label_offsets' not in st.session_state:
        st.session_state.custom_label_offsets = {}

    # Load raw data (instant after first call — @st.cache_data)
    try:
        rails_all, stations_all = load_data()
    except Exception as e:
        st.error(f"Could not load shapefiles: {e}")
        return

    # Compute and cache all derived data once per session
    if 'derived_data' not in st.session_state or 'rails_all_ll' not in st.session_state.derived_data:
        with st.spinner("Preparing data..."):
            rails_all_m = rails_all.to_crs(epsg=3857)
            rails_mrs = rails_all[rails_all[FILTER_COLUMN] == MRS_CODE].copy()
            rails_mrs_ll = rails_mrs.to_crs(epsg=4326)
            rails_all_ll = rails_all.to_crs(epsg=4326)
            _, stations_near_ll, stations_lookup, labels = prepare_stations(stations_all)
            centroid = stations_near_ll.geometry.union_all().centroid
            # Pre-compute minimal GeoJSON for rail layers
            # (single folium.GeoJson layer vs N individual PolyLines)
            rails_mrs_geojson = _gdf_to_minimal_geojson(rails_mrs_ll)
            rails_all_geojson = _gdf_to_minimal_geojson(rails_all_ll)
            st.session_state.derived_data = {
                'rails_all_m': rails_all_m,
                'rails_mrs_ll': rails_mrs_ll,
                'rails_all_ll': rails_all_ll,
                'stations_near_ll': stations_near_ll,
                'stations_lookup': stations_lookup,
                'labels': labels,
                'map_center': [centroid.y, centroid.x],
                'rails_mrs_geojson': rails_mrs_geojson,
                'rails_all_geojson': rails_all_geojson,
            }
        st.success(f"Loaded {len(labels)} stations, {len(rails_all)} rail features")

    d = st.session_state.derived_data
    rails_all_m      = d['rails_all_m']
    rails_mrs_ll     = d['rails_mrs_ll']
    rails_all_ll     = d['rails_all_ll']
    stations_near_ll = d['stations_near_ll']
    stations_lookup  = d['stations_lookup']
    labels           = d['labels']
    map_center       = d['map_center']

    # Sidebar controls
    with st.sidebar:
        st.header("Route Builder")

        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Station A", options=labels, key="origin")
        with col2:
            destination = st.selectbox("Station B", options=labels, index=min(1, len(labels)-1), key="dest")

        vias = st.multiselect("VIA stations (optional)", options=labels, key="vias")

        routing_mode = st.radio("Routing mode", [STRICT_MRS, PREFER_MRS, ALLOW_ALL], index=2)
        if routing_mode == PREFER_MRS:
            penalty = st.slider("Non-MRS penalty", 1.0, 100.0, NON_MRS_PENALTY_DEFAULT, key="penalty")
        else:
            penalty = 1.0

        do_node = st.checkbox("Node intersections", value=NODE_INTERSECTIONS_DEFAULT, key="node", 
                             help="⚠️ SLOW - Expensive geometry operation")
        do_edge = st.checkbox("Snap waypoints to edges", value=SNAP_WAYPOINTS_TO_EDGES_DEFAULT, key="edge",
                             help="Spatial search - can add 30-60 seconds")
        do_corr = st.checkbox("Constrain to corridor", value=CONSTRAIN_CORRIDOR_DEFAULT, key="corr",
                             help="⚠️ SLOW for complex networks")
        if do_corr:
            buf_km = st.slider("Corridor buffer (km)", 5.0, 80.0, CORRIDOR_BUFFER_KM_DEFAULT, key="buf",
                             help="Smaller = faster")
        else:
            buf_km = CORRIDOR_BUFFER_KM_DEFAULT

        color = st.color_picker("Route color", "#1f77b4", key="color")
        highlight_name = st.text_input("System name", "Corridor 1", key="name")
        path_name = st.text_input("Path name (optional)", "", key="path_name",
                                   placeholder="e.g. Direct, Via Campinas…")

        run_btn = st.button("Compute Preview", key="run")
        commit_btn = st.button("Commit Highlight", key="commit")
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
        find_station = st.selectbox(
            "Find station on map:",
            options=["(Select a station...)"] + labels,
            key="find_station",
            help="Highlight a station on the map to find its location"
        )
        if find_station != "(Select a station...)":
            if 'highlighted_station' not in st.session_state:
                st.session_state.highlighted_station = None
            if st.button("📌 Mark on Map", key="mark_station"):
                st.session_state.highlighted_station = find_station
                st.success(f"Marked: {find_station}")
            if st.button("🔄 Clear Marker", key="clear_marker"):
                st.session_state.highlighted_station = None
                st.info("Marker cleared")
        
        st.divider()
        st.subheader("🎯 Manual Label Positioning")
        st.caption("Override automatic label placement for specific stations")
        
        adjust_station = st.selectbox(
            "Station to adjust:",
            options=["(Select...)" ] + sorted(labels),
            key="adjust_label_station"
        )
        
        if adjust_station != "(Select...)":
            # Preset position options
            position_presets = {
                "Auto (collision-avoid)": None,
                "Center below": (8, -50),
                "Far left": (8, -100),
                "Far right": (8, 0),
                "Further down center": (25, -50),
                "Further down left": (25, -100),
                "Further down right": (25, 0),
                "Above center": (-10, -50),
                "Above left": (-10, -100),
                "Above right": (-10, 0),
                "Very far down": (55, -50),
                "Mid-left": (8, -75),
                "Mid-right": (8, -25),
            }
            
            current = st.session_state.custom_label_offsets.get(adjust_station)
            current_key = "Auto (collision-avoid)"
            for k, v in position_presets.items():
                if v == current:
                    current_key = k
                    break
            
            selected_preset = st.selectbox(
                "Label position",
                options=list(position_presets.keys()),
                index=list(position_presets.keys()).index(current_key),
                key="label_preset_pick"
            )
            
            if st.button("✅ Apply Position", key="apply_label_pos"):
                preset_value = position_presets[selected_preset]
                if preset_value is None:
                    st.session_state.custom_label_offsets.pop(adjust_station, None)
                    st.success(f"Reset {adjust_station} to auto")
                else:
                    st.session_state.custom_label_offsets[adjust_station] = preset_value
                    st.success(f"Set {adjust_station} → {selected_preset}")
                st.rerun()
            
            # Fine-tune with sliders
            with st.expander("🎨 Custom offset (advanced)"):
                st.caption("Fine-tune position with sliders")
                if current:
                    default_top, default_x = current
                else:
                    default_top, default_x = 8, -50
                
                custom_top = st.slider(
                    "Vertical (px)",
                    min_value=-40,
                    max_value=80,
                    value=int(default_top),
                    step=5,
                    key="custom_margin_top",
                    help="Negative = above marker, Positive = below"
                )
                custom_x = st.slider(
                    "Horizontal (%)",
                    min_value=-120,
                    max_value=40,
                    value=int(default_x),
                    step=5,
                    key="custom_translate_x",
                    help="-100 = far left, -50 = center, 0 = far right"
                )
                
                if st.button("💾 Save Custom", key="save_custom_offset"):
                    st.session_state.custom_label_offsets[adjust_station] = (custom_top, custom_x)
                    st.success(f"Custom offset saved for {adjust_station}")
                    st.rerun()
        
        # Show currently customized stations
        if st.session_state.custom_label_offsets:
            st.caption(f"**{len(st.session_state.custom_label_offsets)} custom position(s)**")
            if st.button("🔄 Reset All Custom Positions", key="reset_all_custom"):
                st.session_state.custom_label_offsets = {}
                st.success("All custom positions cleared")
                st.rerun()

        st.divider()
        st.subheader("🗂️ Station Group Systems")
        st.caption("Create systems and add multiple station groups per system (each group with its own marker shape/color).")

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

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if st.button("💾 Save/Update Group", key="save_group_btn", use_container_width=True):
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
                        st.caption(f"{perf_icon} {route_text} · {dist_text}{perf_hint}")

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
                                        try:
                                            with open(str(backup_path), 'w') as f:
                                                json.dump({f"{system_name}_original": [path_info.copy()]}, f, indent=2)
                                        except:
                                            pass  # Continue even if backup fails
                                        
                                        step = len(coords) // simplify_factor
                                        path_info['coords_ll'] = coords[::step] + [coords[-1]]
                                        save_highlights(st.session_state.committed_highlights)
                                        st.success(f"✅ Reduced from {n_coords:,} to {len(path_info['coords_ll']):,} coords | Backup: {backup_path.name}")
                                        st.rerun()
                            st.caption("⚠️ Simplification is PERMANENT. Original coordinates cannot be restored. Backup first!")
        else:
            st.info("No committed highlights yet. Compute and commit a route above.")


    # Main map area
    st.subheader("Interactive Map")
    map_zoom = 6  # Initial zoom level

    # ------------------------------------------------------------------
    # Pre-process station data ONCE per session.  Rail geometries now use
    # pre-computed GeoJSON (in derived_data) rendered as a single
    # folium.GeoJson layer instead of N individual PolyLines.
    # ------------------------------------------------------------------
    if 'stations_data_cache' not in st.session_state:
        # Vectorized extraction — avoids iterrows() overhead
        _lats = stations_near_ll.geometry.y.values
        _lons = stations_near_ll.geometry.x.values
        _slabels = stations_near_ll['station_label'].values
        _codes = (
            stations_near_ll['station_code'].values
            if 'station_code' in stations_near_ll.columns
            else np.full(len(stations_near_ll), '', dtype=object)
        )
        _cache = []
        for i in range(len(stations_near_ll)):
            label = str(_slabels[i])
            code = str(_codes[i])
            popup_text = f"{label} | Code: {code}" if code else label
            _cache.append({
                'lat': float(_lats[i]),
                'lon': float(_lons[i]),
                'label': label,
                'popup': popup_text,
            })
        st.session_state.stations_data_cache = _cache
        # Invalidate derived lookup so it gets rebuilt
        st.session_state.pop('stations_by_label', None)

    m = folium.Map(
        location=map_center, 
        zoom_start=int(map_zoom), 
        height=700,
        zoom_snap=0.25,
        zoom_delta=0.25,
        wheel_px_per_zoom_level=120
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
    folium.TileLayer(
        tiles='https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
        attr=_ORM_ATTR,
        name='Rail Network (OpenRailwayMap)',
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
    rails_layer      = folium.FeatureGroup(name="MRS Rails (Code 11)", show=True)
    stations_layer   = folium.FeatureGroup(name="All Stations", show=False)
    route_layer      = folium.FeatureGroup(name="Preview Route Track", show=True)
    route_stations_layer = folium.FeatureGroup(name="Preview Stations", show=True)

    station_group_layers = []
    active_station_group_systems = st.session_state.get('active_station_group_systems', [])
    station_group_systems = st.session_state.get('station_group_systems', {})
    if not isinstance(active_station_group_systems, list):
        active_station_group_systems = []

    # All Brazil rails — single GeoJSON layer (much faster than N individual PolyLines)
    folium.GeoJson(
        d['rails_all_geojson'],
        style_function=lambda x: {
            'color': '#ff7f0e',
            'weight': BASE_RAIL_WEIGHT,
            'opacity': 0.85,
        },
    ).add_to(all_rails_layer)

    # MRS rails (Code 11) — single GeoJSON layer
    folium.GeoJson(
        d['rails_mrs_geojson'],
        style_function=lambda x: {
            'color': BASE_RAIL_COLOR,
            'weight': BASE_RAIL_WEIGHT,
            'opacity': 0.85,
        },
    ).add_to(rails_layer)

    all_stations_label_layout = []

    # Add station markers (off by default) with persistent labels (from cache)
    for s in st.session_state.stations_data_cache:
        folium.CircleMarker(
            location=[s['lat'], s['lon']],
            radius=4,
            color="#0077cc",
            fill=True,
            fill_opacity=0.75,
            popup=s['popup']
        ).add_to(stations_layer)

        label_margin_top, label_translate_x = compute_label_offset(s['lat'], s['lon'], layout_state=all_stations_label_layout)

        _add_label_marker(stations_layer, s['lat'], s['lon'], s['label'],
                          margin_top=label_margin_top,
                          translate_x_pct=label_translate_x)

    if 'stations_by_label' not in st.session_state:
        st.session_state.stations_by_label = {
            s.get('label'): s for s in st.session_state.stations_data_cache
        }
    stations_by_label = st.session_state.stations_by_label
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
        with st.spinner("Building graph..."):
            order = [origin] + list(vias) + [destination]

            # Progress callback
            def prog_cb(msg):
                status_txt.text(msg)
                progress_bar.progress(min(100, int(time.time() * 10) % 100))

            pts = []
            for lab in order:
                row = stations_lookup.loc[lab]
                pts.append((float(row.geometry.x), float(row.geometry.y)))

            t_graph_start = time.time()
            G, node_coords, tree, edges_pack = build_graph(
                pts, routing_mode, penalty, rails_all_m,
                node_intersections=do_node,
                constrain_corridor=do_corr,
                corridor_buffer_m=buf_km * 1000.0,
                progress_cb=prog_cb
            )
            t_graph = time.time() - t_graph_start

            if G is None or len(G) == 0:
                st.error("No graph built. Try relaxing corridor constraint or increasing buffer.")
                return

            if edges_pack is None or tree is None:
                st.error("Graph indexing failed. Try again or disable advanced options.")
                return

            progress_bar.progress(50)
            status_txt.text("Snapping waypoints...")

            edges_gdf, edges_sidx = edges_pack
            wp_nodes = []

            for (x, y), lab in zip(pts, order):
                if do_edge:
                    new_id, _, _ = insert_waypoint_node_on_edge(G, node_coords, edges_gdf, edges_sidx, x, y, routing_mode, penalty)
                    if new_id is not None:
                        wp_nodes.append(new_id)
                        continue

                _, n = tree.query([x, y], k=1)
                wp_nodes.append(int(n))

            progress_bar.progress(75)
            status_txt.text("Computing route...")

            # Shortest paths
            w_metric = 'length_m' if routing_mode == ALLOW_ALL else 'weight'
            seg_paths = []
            total_km = 0.0
            t_route_start = time.time()

            for i in range(len(wp_nodes) - 1):
                s, t = wp_nodes[i], wp_nodes[i + 1]
                try:
                    p = nx.shortest_path(G, s, t, weight=w_metric)
                    km = path_length_km(G, p)
                    total_km += km
                    seg_paths.append(p)
                except nx.NetworkXNoPath:
                    st.error(f"No path found for segment {order[i]} → {order[i+1]}")
                    return

            t_route = time.time() - t_route_start

            full_path = stitch_paths(seg_paths)
            coords_ll = nodes_to_latlon(node_coords, full_path)

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
                'preview_station_points': preview_station_points
            }

            progress_bar.progress(100)
            t_total = time.time() - t_start
            
            # Show performance breakdown
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Graph Build", f"{t_graph:.1f}s")
            with perf_col2:
                st.metric("Routing", f"{t_route:.1f}s")
            with perf_col3:
                st.metric("Total Time", f"{t_total:.1f}s")
            
            if t_total > 60:
                st.warning(
                    f"⚠️ **Computation took {t_total:.0f}s**. To speed up:\n"
                    f"- Uncheck '**Node intersections**' (most time-consuming)\n"
                    f"- Uncheck '**Snap waypoints to edges**'\n"
                    f"- Uncheck '**Constrain to corridor**' or reduce buffer\n"
                    f"- Use fewer VIA stations"
                )
            
            status_txt.text(f"✅ Route complete: {total_km:.1f} km")

            # Show route info
            st.success(f"**Route:** {' → '.join(order)}")
            st.info(f"**Total Distance:** {total_km:.1f} km")

            # Show code breakdown
            if do_edge or do_node:
                st.subheader("Rail Code Breakdown (km)")
                seg_diag = []
                for i, p in enumerate(seg_paths):
                    km = path_length_km(G, p)
                    codes = codigo_breakdown_km(G, p)
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
            st.success(f"Committed: **{system_name}** → {HIGHLIGHTS_SAVE_PATH.name} (auto-saved)")
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
                # Add station name label below the preview endpoint marker
                _add_label_marker(route_stations_layer, p['lat'], p['lon'], p['label'])

    # Draw committed highlights grouped by system.
    # FeatureGroup objects must be freshly created each rerun — they hold a
    # reference to the parent map and cannot be safely reused across reruns.
    # The data (coords_ll, colors, etc.) is already plain Python so this loop
    # is fast even with many committed paths.
    # PERFORMANCE: Only render if toggle is enabled
    if render_committed:
        for system_name in sorted(st.session_state.get('committed_highlights', {}).keys()):
            system_layer = folium.FeatureGroup(name=f"System: {system_name}", show=False)  # Hidden by default for performance
            for h in st.session_state.committed_highlights[system_name]:
                # Optionally simplify coordinates for better performance
                coords = h.get('coords_ll', [])
                if simplify_coords and len(coords) > 100:
                    # Keep every Nth point (adaptive based on path length)
                    step = max(2, len(coords) // 100)
                    coords = coords[::step] + [coords[-1]]  # Always include last point
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
                        _add_label_marker(system_layer, p['lat'], p['lon'], p['label'])

                for station_marker in h.get('selected_station_markers', []):
                    if not isinstance(station_marker, dict):
                        continue
                    lat = station_marker.get('lat')
                    lon = station_marker.get('lon')
                    label = station_marker.get('label', 'Selected station')
                    marker_color = station_marker.get('color', h.get('route_color', PREVIEW_COLOR))
                    if lat is None or lon is None:
                        continue

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
                    # Add station name label below the marker
                    _add_label_marker(system_layer, lat, lon, label)
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
    rails_layer.add_to(m)
    stations_layer.add_to(m)
    route_layer.add_to(m)
    route_stations_layer.add_to(m)
    for group_layer in station_group_layers:
        group_layer.add_to(m)
    
    # Create layer control - try collapsed by default for easier use
    folium.LayerControl(collapsed=True, position='topright').add_to(m)

    # Add comprehensive CSS/JS for layer control scrolling
    m.get_root().html.add_child(folium.Element(_LAYER_SCROLL_FIX))

    # Display map
    st_folium(m, width=1200, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
