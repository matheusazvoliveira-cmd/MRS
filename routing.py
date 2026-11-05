"""
Rail routing for the Streamlit app (streamlit_app_full.py).

The national rail network is noded ONCE into a junction-to-junction graph and
cached on disk (keyed by the shapefile's path, size and mtime).  Each route
request is then a Dijkstra search on that graph — no geometry work per request.

All geometry here is in EPSG:3857 metres, matching the original app.  Mercator
stretches lengths by 1/cos(latitude) (~6-8 % in the South-East), so reported
distances are converted to true metres (true_length_m); routes saved before that
change carry the stretched figure.
"""

import hashlib
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.ops import substring


# ============================================================================
# CONFIGURATION
# ============================================================================

FILTER_COLUMN = "CodigoFerr"
MRS_CODE = 11

STATION_NAME_FIELD = "NomeEstaca"
STATION_ID_FIELD = None

SOURCE_CRS_IF_MISSING = "EPSG:4674"

# Operator <-> ANTT rail code (CodigoFerr) mapping
RAIL_OPERATORS = {
    "MRS":            {"codes": [11],        "color": "#2ca02c",  "label": "MRS Logística"},
    "Rumo":           {"codes": [1, 5, 7, 9], "color": "#1f77b4", "label": "Rumo (Malha Sul / Paulista / Oeste / Norte)"},
    "Vale":           {"codes": [3, 4],       "color": "#9467bd",  "label": "Vale (EFC Carajás / EFVM Vitória-Minas)"},
    "VLI":            {"codes": [8, 12, 13],  "color": "#d62728",  "label": "VLI (FCA / FNS Norte / FNS Central)"},
    "Transnordestina": {"codes": [2],         "color": "#ff7f0e",  "label": "Transnordestina (CFN)"},
    "Ferroeste":      {"codes": [6],          "color": "#8c564b",  "label": "Ferroeste"},
    "FTC":            {"codes": [10],         "color": "#e377c2",  "label": "FTC (Tereza Cristina)"},
}
CODE_TO_OPERATOR = {code: name for name, info in RAIL_OPERATORS.items() for code in info["codes"]}

# Routing modes, relative to a chosen set of rail networks (operators)
STRICT = "strict"        # only the chosen networks
PREFER = "prefer"        # other networks cost penalty_mult x their length
ALLOW_ALL = "allow_all"  # shortest distance on any track
LEGACY_MODES = {"strict_mrs": STRICT, "prefer_mrs": PREFER}  # older ids, network = MRS

OTHER_NETWORK_PENALTY_DEFAULT = 10.0


def network_codes(operators):
    """Sorted CodigoFerr codes of the given operator names (see RAIL_OPERATORS)."""
    return tuple(sorted({c for op in operators for c in RAIL_OPERATORS[op]["codes"]}))

# Topology repair: line ends within this distance of other track get a connector
SNAP_TOLERANCE_M = 75.0
# Precision grid used when noding intersections (makes noding robust)
NODING_GRID_M = 0.5
# Nodes closer than this are one junction (grid snapping can leave two line ends one grid cell apart)
NODE_MERGE_M = 1.0
# Max distance between a noded piece and its source line when assigning CodigoFerr
ASSIGN_CODIGO_MAX_DIST_M = 10.0

# Bump when the graph-building logic changes so stale disk caches are ignored
GRAPH_CACHE_VERSION = 3

to_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
_EARTH_R = 6378137.0  # EPSG:3857 sphere radius


def true_length_m(xy):
    """Length in true metres of a polyline of EPSG:3857 points ([N, 2] array): each
    segment's Mercator length times cos(latitude) at its middle."""
    if len(xy) < 2:
        return 0.0
    lat = 2 * np.arctan(np.exp(xy[:, 1] / _EARTH_R)) - np.pi / 2
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return float(np.sum(seg * np.cos((lat[1:] + lat[:-1]) / 2)))


# ============================================================================
# DATA LOADING
# ============================================================================

def _read_vector(path):
    try:
        return gpd.read_file(path, engine='pyogrio', use_arrow=True)
    except Exception:
        # pyarrow missing or unsupported — fall back to the default reader
        return gpd.read_file(path)


def read_rails(path):
    """Read the rails shapefile, set CRS if missing and normalise rail codes to Int64."""
    rails = _read_vector(path)
    if rails.crs is None and SOURCE_CRS_IF_MISSING:
        rails = rails.set_crs(SOURCE_CRS_IF_MISSING)
    rails[FILTER_COLUMN] = pd.to_numeric(rails[FILTER_COLUMN], errors='coerce').round().astype('Int64')
    return rails


def read_stations(path):
    """Read the stations shapefile and set CRS if missing."""
    stations = _read_vector(path)
    if stations.crs is None and SOURCE_CRS_IF_MISSING:
        stations = stations.set_crs(SOURCE_CRS_IF_MISSING)
    return stations


def prepare_stations(stations_all):
    """Extract point stations (EPSG:3857) and build unique labels with 3-letter codes."""
    stations_m = stations_all.to_crs(epsg=3857)
    stations_near = stations_m[stations_m.geometry.type == 'Point'].copy()

    if FILTER_COLUMN in stations_near.columns:
        stations_near[FILTER_COLUMN] = pd.to_numeric(
            stations_near[FILTER_COLUMN], errors='coerce'
        ).round().astype('Int64')
    else:
        stations_near[FILTER_COLUMN] = pd.Series(pd.NA, index=stations_near.index, dtype='Int64')

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
    return stations_near


# ============================================================================
# GRAPH BUILDING (once per shapefile version)
# ============================================================================

class RailGraph:
    """Junction-to-junction rail graph.  Edge i runs from node u[i] to node v[i]
    along geoms[i] (EPSG:3857); code[i] is its CodigoFerr (NaN if unknown)."""

    def __init__(self, node_xy, u, v, length, code, geoms):
        self.node_xy = node_xy
        self.u = u
        self.v = v
        self.length = length
        self.code = code
        self.geoms = geoms
        self._trees = {}

    @property
    def n_nodes(self):
        return len(self.node_xy)

    @property
    def n_edges(self):
        return len(self.u)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_trees'] = {}  # STRtrees are rebuilt on demand
        return state

    def edge_tree(self, mode, codes):
        """STRtree over the edges usable in *mode* (only the *codes* networks in strict mode)."""
        key = tuple(codes) if mode == STRICT else 'all'
        if key not in self._trees:
            idx = np.flatnonzero(np.isin(self.code, codes)) if mode == STRICT else np.arange(self.n_edges)
            self._trees[key] = (shapely.STRtree(self.geoms[idx]), idx)
        return self._trees[key]


def _gap_connectors(geoms, codes):
    """Short connector lines joining each loose line end to the nearest other line
    within SNAP_TOLERANCE_M.  Replaces the old snap()-against-union step."""
    n = len(geoms)
    ends = np.concatenate([shapely.get_point(geoms, 0), shapely.get_point(geoms, -1)])
    owner = np.concatenate([np.arange(n), np.arange(n)])

    tree = shapely.STRtree(geoms)
    end_idx, line_idx = tree.query(ends, predicate='dwithin', distance=SNAP_TOLERANCE_M)
    other = line_idx != owner[end_idx]
    end_idx, line_idx = end_idx[other], line_idx[other]
    if len(end_idx) == 0:
        return np.array([], dtype=object), np.array([], dtype=float)

    dist = shapely.distance(ends[end_idx], geoms[line_idx])
    # Nearest other line per end
    order = np.lexsort((dist, end_idx))
    end_idx, line_idx, dist = end_idx[order], line_idx[order], dist[order]
    first = np.unique(end_idx, return_index=True)[1]
    end_idx, line_idx, dist = end_idx[first], line_idx[first], dist[first]

    # Ends already touching another line are connected by noding
    loose = dist > NODING_GRID_M
    end_idx, line_idx = end_idx[loose], line_idx[loose]
    connectors = shapely.shortest_line(ends[end_idx], geoms[line_idx])
    return connectors, codes[owner[end_idx]]


def build_rail_graph(rails_m, progress_cb=None):
    """Node the whole rail network (EPSG:3857 GeoDataFrame) into a RailGraph."""
    def cb(msg):
        if progress_cb:
            progress_cb(msg)

    cb("Preparing rail lines...")
    lines = rails_m[[FILTER_COLUMN, 'geometry']].explode(index_parts=False)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty & (lines.geom_type == 'LineString')]
    # Identical copies of a line (e.g. the Corrego do Feijao and Suzano/Imperatriz branches) would
    # make each other's loose ends look connected, so they never got a gap connector
    lines = lines[~pd.Series(shapely.to_wkb(lines.geometry.array), index=lines.index).duplicated().to_numpy()]
    geoms = np.asarray(lines.geometry.array, dtype=object)
    codes = lines[FILTER_COLUMN].to_numpy(dtype=float, na_value=np.nan)

    cb("Closing gaps between line ends...")
    conn_geoms, conn_codes = _gap_connectors(geoms, codes)
    src_geoms = np.concatenate([geoms, conn_geoms])
    src_codes = np.concatenate([codes, conn_codes])

    cb("Noding intersections...")
    noded = shapely.union_all(src_geoms, grid_size=NODING_GRID_M)
    pieces = shapely.get_parts(noded)
    pieces = pieces[(shapely.get_type_id(pieces) == 1) & (shapely.length(pieces) > 0)]

    cb("Assigning rail codes...")
    mids = shapely.line_interpolate_point(pieces, 0.5, normalized=True)
    in_idx, src_idx = shapely.STRtree(src_geoms).query_nearest(
        mids, max_distance=ASSIGN_CODIGO_MAX_DIST_M, all_matches=False
    )
    piece_codes = np.full(len(pieces), np.nan)
    piece_codes[in_idx] = src_codes[src_idx]

    cb("Building graph...")
    start_xy = shapely.get_coordinates(shapely.get_point(pieces, 0))
    end_xy = shapely.get_coordinates(shapely.get_point(pieces, -1))
    node_xy, inv = np.unique(np.concatenate([start_xy, end_xy]), axis=0, return_inverse=True)
    inv = inv.ravel()
    # Merge nodes within NODE_MERGE_M: e.g. Rondonopolis was cut off from the network by two
    # line ends 0.5 m apart that snapped to neighbouring grid cells
    pairs = cKDTree(node_xy).query_pairs(NODE_MERGE_M, output_type='ndarray')
    if len(pairs):
        n_nodes = len(node_xy)
        adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
        _, cluster = connected_components(adj, directed=False)
        node_xy = node_xy[np.unique(cluster, return_index=True)[1]]  # first node of each cluster
        inv = cluster[inv]
    u, v = inv[:len(pieces)], inv[len(pieces):]

    keep = u != v  # closed loops can never shorten a route
    graph = RailGraph(
        node_xy=node_xy,
        u=u[keep].astype(np.int64),
        v=v[keep].astype(np.int64),
        length=shapely.length(pieces[keep]),
        code=piece_codes[keep],
        geoms=pieces[keep],
    )
    cb(f"Graph ready: {graph.n_nodes:,} nodes / {graph.n_edges:,} edges")
    return graph


def _graph_cache_path(cache_dir, rails_path):
    st_ = Path(rails_path).stat()
    key = f"{Path(rails_path).resolve()}|{st_.st_size}|{st_.st_mtime_ns}|v{GRAPH_CACHE_VERSION}"
    return Path(cache_dir) / f"rail_graph_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"


def load_or_build_graph(load_rails_m, rails_path, cache_dir, progress_cb=None):
    """Load the RailGraph from the disk cache, or build and cache it.

    *load_rails_m* is a callable returning the rails GeoDataFrame in EPSG:3857;
    it is only called when the graph has to be built."""
    cache_path = _graph_cache_path(cache_dir, rails_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                graph = pickle.load(f)
            if isinstance(graph, RailGraph):
                return graph
        except Exception:
            pass  # corrupt/incompatible cache — rebuild below

    graph = build_rail_graph(load_rails_m(), progress_cb=progress_cb)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for old in cache_path.parent.glob('rail_graph_*.pkl'):
            old.unlink(missing_ok=True)
        tmp = cache_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except OSError:
        pass  # read-only location — keep the in-memory graph
    return graph


# ============================================================================
# ROUTING (per request)
# ============================================================================

class NoPathError(Exception):
    def __init__(self, segment_index):
        super().__init__(f"No path for segment {segment_index}")
        self.segment_index = segment_index


def edge_weights(graph, mode, penalty_mult=OTHER_NETWORK_PENALTY_DEFAULT, codes=(MRS_CODE,)):
    """Traversal cost per edge for a routing mode relative to the *codes* networks (inf = not allowed)."""
    if mode == ALLOW_ALL:
        return graph.length
    outside = ~np.isin(graph.code, codes)  # NaN codes count as outside here
    if mode == STRICT:
        return np.where(outside, np.inf, graph.length)
    # Prefer: unknown codes are not penalised (same as the original app)
    penalised = outside & ~np.isnan(graph.code)
    return np.where(penalised, graph.length * float(penalty_mult), graph.length)


def _snap_to_edge(graph, mode, codes, x, y):
    """Nearest usable edge and the distance along it of the projected point."""
    tree, idx = graph.edge_tree(mode, codes)
    if len(idx) == 0:
        return None, None
    pt = shapely.Point(x, y)
    hit = tree.query_nearest(pt, all_matches=False)
    e = int(idx[hit[0]])
    return e, float(shapely.line_locate_point(graph.geoms[e], pt))


def route(graph, points_xy, mode, penalty_mult=OTHER_NETWORK_PENALTY_DEFAULT, codes=(MRS_CODE,)):
    """Shortest route through the waypoints (EPSG:3857 x/y), in order.

    *mode* is STRICT / PREFER / ALLOW_ALL relative to the rail networks with
    CodigoFerr in *codes* (see network_codes()); legacy 'strict_mrs' /
    'prefer_mrs' are accepted.

    Returns dict(total_km, segment_km, coords_ll [(lat, lon)], segment_codes
    [{codigo: km}] per segment, segment_ends [index in coords_ll after each segment's last point]).  Raises NoPathError(segment_index).
    """
    mode = LEGACY_MODES.get(mode, mode)
    codes = tuple(sorted(codes))
    w = edge_weights(graph, mode, penalty_mult, codes)
    usable = np.isfinite(w)
    n = graph.n_nodes

    # Virtual edges split the snapped edge at each waypoint: (from, to, edge, d_from, d_to)
    virtual = []
    snapped = []
    for k, (x, y) in enumerate(points_xy):
        e, d = _snap_to_edge(graph, mode, codes, x, y)
        if e is None:
            raise NoPathError(max(0, k - 1))
        node = n + k
        virtual.append((int(graph.u[e]), node, e, 0.0, d))
        virtual.append((node, int(graph.v[e]), e, d, float(graph.length[e])))
        for j, (e2, d2) in enumerate(snapped):
            if e2 == e:  # two waypoints on the same edge
                virtual.append((n + j, node, e, d2, d))
        snapped.append((e, d))

    real = np.flatnonzero(usable)
    vu = np.array([t[0] for t in virtual], dtype=np.int64)
    vv = np.array([t[1] for t in virtual], dtype=np.int64)
    ve = np.array([t[2] for t in virtual], dtype=np.int64)
    vlen = np.array([abs(t[4] - t[3]) for t in virtual])
    rate = np.where(graph.length[ve] > 0, w[ve] / np.maximum(graph.length[ve], 1e-9), 1.0)

    U = np.concatenate([graph.u[real], vu])
    V = np.concatenate([graph.v[real], vv])
    W = np.maximum(np.concatenate([w[real], vlen * rate]), 1e-6)  # explicit zeros are dropped
    ref = np.concatenate([real, -1 - np.arange(len(virtual))])     # >=0 real edge, <0 virtual

    # Keep the cheapest edge per node pair (sparse matrices would sum duplicates)
    lo, hi = np.minimum(U, V), np.maximum(U, V)
    n_total = n + len(points_xy)
    pair = lo * n_total + hi
    order = np.lexsort((W, pair))
    first = order[np.unique(pair[order], return_index=True)[1]]
    lo, hi, W, ref = lo[first], hi[first], W[first], ref[first]
    pair_ref = dict(zip((lo * n_total + hi).tolist(), ref.tolist()))
    csr = coo_matrix((W, (lo, hi)), shape=(n_total, n_total)).tocsr()

    coords = []
    segment_km = []
    segment_codes = []
    segment_ends = []
    for k in range(len(points_xy) - 1):
        s, t = n + k, n + k + 1
        dist, pred = dijkstra(csr, directed=False, indices=s, return_predecessors=True)
        if not np.isfinite(dist[t]):
            raise NoPathError(k)
        path = [t]
        while path[-1] != s:
            path.append(int(pred[path[-1]]))
        path.reverse()

        seg_m = 0.0
        codes_km = {}
        for a, b in zip(path[:-1], path[1:]):
            r = pair_ref[min(a, b) * n_total + max(a, b)]
            if r >= 0:
                geom, code = graph.geoms[r], graph.code[r]
                forward = graph.u[r] == a
            else:
                fa, fb, e, d0, d1 = virtual[-1 - r]
                geom = substring(graph.geoms[e], min(d0, d1), max(d0, d1))
                code = graph.code[e]
                forward = (fa == a) == (d0 <= d1)
            xy = shapely.get_coordinates(geom)
            if not forward:
                xy = xy[::-1]
            length = true_length_m(xy)
            if coords and len(xy):
                xy = xy[1:]
            coords.extend(map(tuple, xy))
            seg_m += length
            key = None if np.isnan(code) else int(code)
            codes_km[key] = codes_km.get(key, 0.0) + float(length) / 1000.0
        segment_km.append(seg_m / 1000.0)
        segment_ends.append(len(coords))
        segment_codes.append(dict(sorted(codes_km.items(), key=lambda kv: (-kv[1], str(kv[0])))))

    if coords:
        xs, ys = np.asarray(coords).T
        lons, lats = to_4326.transform(xs, ys)
        coords_ll = list(zip(lats.tolist(), lons.tolist()))
    else:
        coords_ll = []
    return {
        'total_km': float(sum(segment_km)),
        'segment_km': segment_km,
        'segment_codes': segment_codes,
        'coords_ll': coords_ll,
        'segment_ends': segment_ends,
    }
