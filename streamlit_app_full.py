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

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx

from shapely.geometry import LineString, Point
from shapely.ops import unary_union, snap, nearest_points
from scipy.spatial import cKDTree
from pyproj import Transformer

import folium
from streamlit_folium import st_folium


# ============================================================================
# CONFIGURATION (match notebook settings)
# ============================================================================

RAILS_SHP_PATH = Path(r"/home/matheusazv/Documentos/Advance technology/Estacoes Code/LInhas/Linhas_BR.shp")
STATIONS_SHP_PATH = Path(r"/home/matheusazv/Documentos/Advance technology/Estacoes Code/Estacoes/Estacoes.shp")

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


# ============================================================================
# UTILITIES
# ============================================================================

to_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)

# Default save file paths — always absolute, anchored to this script's directory
_SCRIPT_DIR = Path(__file__).parent.resolve()
HIGHLIGHTS_SAVE_PATH = _SCRIPT_DIR / "committed_highlights.json"   # live auto-save (every commit/delete)
EXPORT_SAVE_PATH     = _SCRIPT_DIR / "exported_highlights.json"    # manual export snapshot only

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


@st.cache_data
def load_data():
    """Load shapefiles — raises on failure so the caller can show the error in context."""
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

    # Append code to label
    stations_near['station_label'] = stations_near.apply(
        lambda r: f"{r['station_label']} [{r['station_code']}]" if r['station_code'] else r['station_label'],
        axis=1
    )

    stations_near_ll = stations_near.to_crs(epsg=4326)
    stations_lookup = stations_near.set_index('station_label')
    labels = sorted(list(stations_lookup.index))

    return stations_near, stations_near_ll, stations_lookup, labels


def edge_cost(length, codigo, routing_mode, penalty_mult):
    """Compute edge traversal cost, applying penalty for non-MRS segments."""
    if routing_mode == PREFER_MRS and codigo != MRS_CODE:
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

    for _, row in rails.iterrows():
        codigo = row.get(FILTER_COLUMN)
        coords = list(row.geometry.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            u_id = get_node_id(a)
            v_id = get_node_id(b)
            seg = LineString([a, b])
            length_m = seg.length
            w = edge_cost(length_m, codigo, routing_mode, penalty_mult)

            if G.has_edge(u_id, v_id):
                if w < G[u_id][v_id]['weight']:
                    G[u_id][v_id].update({'weight': w, 'length_m': length_m, 'codigo': codigo})
            else:
                G.add_edge(u_id, v_id, weight=w, length_m=length_m, codigo=codigo)

            edge_rows.append({'u': u_id, 'v': v_id, 'geometry': seg, 'codigo': codigo, 'length_m': length_m})

    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry='geometry', crs='EPSG:3857')
    edges_sidx = edges_gdf.sindex

    node_xy = np.array(node_coords)
    tree = cKDTree(node_xy)

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

    # Load raw data (instant after first call — @st.cache_data)
    try:
        rails_all, stations_all = load_data()
    except Exception as e:
        st.error(f"Could not load shapefiles: {e}")
        return

    # Compute and cache all derived data once per session
    if 'derived_data' not in st.session_state:
        with st.spinner("Preparing data..."):
            rails_all_m = rails_all.to_crs(epsg=3857)
            rails_mrs = rails_all[rails_all[FILTER_COLUMN] == MRS_CODE].copy()
            rails_mrs_ll = rails_mrs.to_crs(epsg=4326)
            _, stations_near_ll, stations_lookup, labels = prepare_stations(stations_all)
            centroid = stations_near_ll.geometry.union_all().centroid
            st.session_state.derived_data = {
                'rails_all_m': rails_all_m,
                'rails_mrs_ll': rails_mrs_ll,
                'stations_near_ll': stations_near_ll,
                'stations_lookup': stations_lookup,
                'labels': labels,
                'map_center': [centroid.y, centroid.x],
            }
        st.success(f"Loaded {len(labels)} stations, {len(rails_all)} rail features")

    d = st.session_state.derived_data
    rails_all_m    = d['rails_all_m']
    rails_mrs_ll   = d['rails_mrs_ll']
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

        do_node = st.checkbox("Node intersections", value=NODE_INTERSECTIONS_DEFAULT, key="node")
        do_edge = st.checkbox("Snap waypoints to edges", value=SNAP_WAYPOINTS_TO_EDGES_DEFAULT, key="edge")
        do_corr = st.checkbox("Constrain to corridor", value=CONSTRAIN_CORRIDOR_DEFAULT, key="corr")
        if do_corr:
            buf_km = st.slider("Corridor buffer (km)", 5.0, 80.0, CORRIDOR_BUFFER_KM_DEFAULT, key="buf")
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

        if st.button("📤 Export All", key="export_btn"):
            try:
                with open(str(EXPORT_SAVE_PATH), 'w') as f:
                    json.dump(st.session_state.committed_highlights, f, indent=2)
                st.success(f"Exported → {EXPORT_SAVE_PATH.name}")
            except Exception as e:
                st.error(f"Export failed: {e}")

        st.markdown("**Import JSON highlights**")
        import_file = st.file_uploader("Import JSON file", type=["json"], key="import_file", label_visibility="collapsed")

        if import_file is not None:
            # Show file info immediately (before import button click)
            file_bytes = import_file.getvalue()
            file_kb = len(file_bytes) / 1024
            n_systems, n_paths = 0, 0
            try:
                preview = json.loads(file_bytes)
                n_systems = len(preview)
                n_paths = sum(len(v) for v in preview.values())
                est_sec = max(0.1, file_kb / 800)  # ~800 KB/s effective merge rate
                st.info(
                    f"📄 **{import_file.name}** — {file_kb:.1f} KB\n\n"
                    f"{n_systems} system(s), {n_paths} path(s) · est. ~{est_sec:.1f}s"
                )
            except Exception:
                st.warning("Could not preview file.")
                preview = None

            if preview is not None and st.button("📥 Load into session", key="import_btn"):
                t0 = time.time()
                for imported_system_name, imported_paths in preview.items():
                    if imported_system_name not in st.session_state.committed_highlights:
                        st.session_state.committed_highlights[imported_system_name] = []
                    existing_orders = {
                        tuple(p.get('order', []))
                        for p in st.session_state.committed_highlights[imported_system_name]
                    }
                    for path in imported_paths:
                        key = tuple(path.get('order', []))
                        if key not in existing_orders:
                            st.session_state.committed_highlights[imported_system_name].append(path)
                            existing_orders.add(key)
                save_highlights(st.session_state.committed_highlights)
                elapsed = time.time() - t0
                st.session_state.last_import_info = (
                    f"✅ Imported {n_systems} system(s), {n_paths} path(s) "
                    f"from **{import_file.name}** in {elapsed:.2f}s"
                )
                st.rerun()  # needed: forces sidebar to re-render with new committed data

        if st.session_state.get("last_import_info"):
            st.success(st.session_state.last_import_info)
            st.session_state.last_import_info = None  # clear after showing once

        st.divider()
        n_committed = sum(len(v) for v in st.session_state.committed_highlights.values())
        n_systems_committed = len(st.session_state.committed_highlights)
        st.subheader(f"Committed Systems ({n_systems_committed} | {n_committed} paths)")

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

                    col_sj, col_sc = st.columns(2)
                    with col_sj:
                        if st.button("📄 Export JSON", key=f"export_json_{system_name}"):
                            sys_export_path = _SCRIPT_DIR / f"{system_name}.json"
                            try:
                                with open(str(sys_export_path), 'w') as f:
                                    json.dump({system_name: paths_list}, f, indent=2)
                                st.success(f"Exported → {sys_export_path.name}")
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                    with col_sc:
                        st.caption("JSON only")
                    
                    st.divider()
                    
                    for i, path_info in enumerate(paths_list):
                        route_text = " → ".join(path_info.get('order', []))
                        dist_text = f"{path_info.get('total_km', 0.0):.1f} km"
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
                        st.caption(f"↳ {route_text} · {dist_text}")
        else:
            st.info("No committed highlights yet. Compute and commit a route above.")


    # Main map area
    st.subheader("Interactive Map")

    # ------------------------------------------------------------------
    # Pre-process geodataframe geometries ONCE per session into plain
    # Python lists so the expensive iterrows / coord extraction runs
    # only on first load and never again on subsequent reruns.
    # ------------------------------------------------------------------
    if 'rails_coords_cache' not in st.session_state:
        _cache = []
        for _, row in rails_mrs_ll.iterrows():
            for line in iter_lines(row.geometry):
                _cache.append([(lat, lon) for lon, lat in line.coords])
        st.session_state.rails_coords_cache = _cache

    if 'stations_data_cache' not in st.session_state:
        _cache = []
        for _, row in stations_near_ll.iterrows():
            popup_text = str(row['station_label'])
            if row.get('station_code'):
                popup_text += f" | Code: {row['station_code']}"
            _cache.append({
                'lat': row.geometry.y,
                'lon': row.geometry.x,
                'label': str(row['station_label']),
                'popup': popup_text,
            })
        st.session_state.stations_data_cache = _cache

    m = folium.Map(location=map_center, zoom_start=6, height=700)

    # Brazil bounding box — tiles outside this rectangle are never requested
    _BR_BOUNDS = [[-33.75, -73.99], [5.27, -34.79]]
    _orm_attr = (
        'Map tiles by <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a> '
        '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>) | '
        'Data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )
    folium.TileLayer(
        tiles='https://{s}.tiles.openrailwaymap.org/gauge/{z}/{x}/{y}.png',
        attr=_orm_attr,
        name='Rail Gauge (OpenRailwayMap)',
        overlay=True,
        control=True,
        show=False,
        min_zoom=2,
        max_zoom=19,
        opacity=0.8,
        subdomains=['a', 'b', 'c'],
        bounds=_BR_BOUNDS,
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
        attr=_orm_attr,
        name='Rail Network (OpenRailwayMap)',
        overlay=True,
        control=True,
        show=False,
        min_zoom=2,
        max_zoom=19,
        opacity=0.8,
        subdomains=['a', 'b', 'c'],
        bounds=_BR_BOUNDS,
    ).add_to(m)

    rails_layer = folium.FeatureGroup(name="MRS Rails (Code 11)", show=True)
    stations_layer = folium.FeatureGroup(name="All Stations", show=False)
    route_layer = folium.FeatureGroup(name="Preview Route Track", show=True)
    route_stations_layer = folium.FeatureGroup(name="Preview Stations", show=True)

    # Add MRS rails base layer (from session-state coord cache)
    for coords in st.session_state.rails_coords_cache:
        folium.PolyLine(
            coords,
            color=BASE_RAIL_COLOR,
            weight=BASE_RAIL_WEIGHT,
            opacity=0.85
        ).add_to(rails_layer)

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

        folium.Marker(
            location=[s['lat'], s['lon']],
            popup=s['popup'],
            icon=folium.DivIcon(
                html=f"<div style='font-size: 10px; background-color: white; "
                     f"padding: 2px 4px; border-radius: 2px; border: 1px solid #999; "
                     f"white-space: nowrap;'>{s['label']}</div>"
            )
        ).add_to(stations_layer)

    # Run routing on button click
    if run_btn:
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

            G, node_coords, tree, edges_pack = build_graph(
                pts, routing_mode, penalty, rails_all_m,
                node_intersections=do_node,
                constrain_corridor=do_corr,
                corridor_buffer_m=buf_km * 1000.0,
                progress_cb=prog_cb
            )

            if G is None or len(G) == 0:
                st.error("No graph built. Try relaxing corridor constraint or increasing buffer.")
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
            st.session_state.committed_highlights[system_name].append({
                'coords_ll': list(preview_data.get('coords_ll', [])),
                'order': list(preview_data.get('order', [])),
                'total_km': float(preview_data.get('total_km', 0.0)),
                'route_color': preview_data.get('route_color', PREVIEW_COLOR),
                'highlight_name': system_name,
                'path_name': committed_path_name,
                'preview_station_points': list(preview_data.get('preview_station_points', []))
            })
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

    # Draw committed highlights grouped by system.
    # FeatureGroup objects must be freshly created each rerun — they hold a
    # reference to the parent map and cannot be safely reused across reruns.
    # The data (coords_ll, colors, etc.) is already plain Python so this loop
    # is fast even with many committed paths.
    for system_name in sorted(st.session_state.get('committed_highlights', {}).keys()):
        system_layer = folium.FeatureGroup(name=f"System: {system_name}", show=True)
        for h in st.session_state.committed_highlights[system_name]:
            if h.get('coords_ll'):
                folium.PolyLine(
                    h['coords_ll'],
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
        system_layer.add_to(m)

    rails_layer.add_to(m)
    stations_layer.add_to(m)
    route_layer.add_to(m)
    route_stations_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Display map
    st_folium(m, width=1200, height=700, returned_objects=[])


if __name__ == "__main__":
    main()
