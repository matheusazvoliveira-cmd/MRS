"""
Track elevation from the subdivision TrackData XML (the onboard track database the KMZ
exports are made from).  The KMZ has no heights; the XML has, per track block, grade
points (offset along the block -> elevation) and located points along the block.

Units, checked against the located points (ECEF centimetres, so their own height is known):
offsets are feet, Elevation is hundredths of a foot in FormatVersion 137 files.  The
elevation unit is detected per file rather than assumed, because older (FormatVersion 6)
files exist too.  SpeedAndWeightUnits = 2 in every file; the speeds are km/h.

elevation_profile() samples a route (lat/lon) against the blocks of every loaded
subdivision and returns distance / elevation rows plus summary figures.
"""

import hashlib
import pickle
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import substring
from pyproj import Transformer

NS = '{http://www.wabtec.com/WRE/SystemTest/Track}'
TRACKDATA_VERSION = 1   # bump when load_trackdata() output changes

FT = 0.3048
# Candidate Elevation units (metres per unit); the one matching the located points' height wins
_ELEV_UNITS = {'0.01 ft': FT / 100, 'ft': FT, 'cm': 0.01, 'm': 1.0}

PROFILE_STEP_M = 100.0       # route sampled every ~100 m
PROFILE_MAX_SAMPLES = 4000   # ...but never more samples than this
MATCH_DISTANCE_M = 60.0      # a sample farther than this from any block has no elevation
GRADE_WINDOW_M = 1000.0      # max grade is reported over this distance
JUMP_MAX_GRADE = 0.08        # a height change steeper than 8 % between samples is a jump between lines
JUMP_MIN_M = 5.0             # ...and at least this many metres

_ecef_to_lla = Transformer.from_crs('EPSG:4978', 'EPSG:4979', always_xy=True)
_to_3857 = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


def find_trackdata(kmz_path, subdiv_id, version):
    """The TrackData XML next to a subdivision KMZ: '<RR>.<id>.<ver>.TrackData.xml', '<RR>.<id>.<ver>.xml'
    or '<RR>.<id>.<ver>_Xml.zip' (all three naming styles occur).  None if there is none."""
    folder = Path(kmz_path).parent
    tag = f".{int(subdiv_id):05d}.{version}".lower()
    found = {}
    for f in folder.iterdir():
        name = f.name.lower()
        if tag + '.' not in name and tag + '_' not in name:
            continue
        if name.endswith('.trackdata.xml'):
            found[0] = f
        elif name.endswith('.xml'):
            found[1] = f
        elif name.endswith('_xml.zip'):
            found[2] = f
    return found[min(found)] if found else None


def _read_xml(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            data = z.read(next(n for n in z.namelist() if n.lower().endswith('.xml')))
    else:
        data = path.read_bytes()
    if re.search(rb'<!(DOCTYPE|ENTITY)', data[:100_000]):
        raise ValueError(f"{path.name}: DOCTYPE/ENTITY declarations are not accepted")
    # Some exports write empty sign texts as a NUL character reference, which is not legal XML
    return ET.fromstring(re.sub(rb'&#(x0+|0+);', b'', data))


def load_trackdata(path):
    """Blocks with located points and grade points from one TrackData XML.

    Returns dict(speed_units, elevation_unit, blocks=[dict(id, lonlat [N,2], offsets_ft [N],
    grade_ft [M], grade_m [M])]) — only blocks that have both."""
    root = _read_xml(path)
    raw = []
    for b in root.iter(NS + 'BlockFeature'):
        pts = [(int(h.findtext(NS + 'Offset')), int(h.findtext(NS + 'LocationX')),
                int(h.findtext(NS + 'LocationY')), int(h.findtext(NS + 'LocationZ')))
               for h in b.iter(NS + 'BlockHeadingFeature')]
        grade = [(int(e.findtext(NS + 'Offset')), int(e.findtext(NS + 'Elevation')))
                 for e in b.iter(NS + 'BlockElevationFeature')]
        if len(pts) >= 2 and grade:
            raw.append((b.findtext(NS + 'BlockId'), np.array(pts, dtype=float), np.array(grade, dtype=float)))

    # One vectorised ECEF (cm) -> lon/lat/height conversion for the whole file
    allpts = np.concatenate([p for _, p, _ in raw]) if raw else np.zeros((0, 4))
    lon, lat, h = _ecef_to_lla.transform(allpts[:, 1] / 100, allpts[:, 2] / 100, allpts[:, 3] / 100)

    # Elevation unit: the one that makes the grade points agree with the located points' height
    ref_h, ref_e = [], []
    i = 0
    for _, p, g in raw:
        n = len(p)
        ref_h.append(np.interp(g[:, 0], p[:, 0], h[i:i + n]))
        ref_e.append(g[:, 1])
        i += n
    unit = 'm'
    if raw:
        ref_h, ref_e = np.concatenate(ref_h), np.concatenate(ref_e)
        unit = min(_ELEV_UNITS, key=lambda u: np.median(np.abs(ref_e * _ELEV_UNITS[u] - ref_h)))
    to_m = _ELEV_UNITS[unit]

    blocks = []
    i = 0
    for block_id, p, g in raw:
        n = len(p)
        order = np.argsort(g[:, 0])
        blocks.append({
            'id': block_id,
            'lonlat': np.column_stack([lon[i:i + n], lat[i:i + n]]).astype(np.float64),
            'offsets_ft': p[:, 0],
            'grade_ft': g[order, 0],
            'grade_m': (g[order, 1] * to_m).astype(np.float32),
        })
        i += n
    units = root.findtext(f'.//{NS}SpeedAndWeightUnits')
    return {'speed_units': units, 'elevation_unit': unit, 'blocks': blocks}


def _cache_path(path, cache_dir):
    path = Path(path)
    st_ = path.stat()
    key = f"{path.resolve()}|{st_.st_size}|{st_.st_mtime_ns}|t{TRACKDATA_VERSION}"
    return Path(cache_dir) / f"track_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"


def is_cached(path, cache_dir):
    """True if load_trackdata_cached() would read *path* from the disk cache."""
    return _cache_path(path, cache_dir).exists()


def load_trackdata_cached(path, cache_dir):
    """load_trackdata() with a disk cache keyed by file path, size, mtime and TRACKDATA_VERSION."""
    path = Path(path)
    cache_path = _cache_path(path, cache_dir)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupt cache — parse again
    data = load_trackdata(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except OSError:
        pass
    return data


class ElevationIndex:
    """Every block of the loaded subdivisions, searchable by location (EPSG:3857)."""

    def __init__(self, datasets):
        """*datasets*: {subdivision name: load_trackdata() result}."""
        self.lines, self.cumlen, self.offsets, self.grade_ft, self.grade_m, self.source = [], [], [], [], [], []
        for name, data in datasets.items():
            for b in data['blocks']:
                x, y = _to_3857.transform(b['lonlat'][:, 0], b['lonlat'][:, 1])
                line = shapely.LineString(np.column_stack([x, y]))
                if line.length == 0:
                    continue
                self.lines.append(line)
                self.cumlen.append(np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]))
                self.offsets.append(b['offsets_ft'])
                self.grade_ft.append(b['grade_ft'])
                self.grade_m.append(b['grade_m'])
                self.source.append(name)
        self.tree = shapely.STRtree(self.lines) if self.lines else None

    def __len__(self):
        return len(self.lines)


SNAP_STEP_M = 50.0           # route sampled every ~50 m for snapping to the track
SNAP_MAX_SAMPLES = 20000
SNAP_RADIUS_M = 1000.0       # the ANTT lines are up to ~750 m off the real track in places
SNAP_K = 8                   # candidate blocks per sample
SNAP_SIGMA_M = 100.0         # emission: cost per this many metres between route and track
SNAP_BETA_M = 20.0           # transition: cost per this many metres of progress mismatch
SNAP_SWITCH_COST = 2.0       # changing block (keeps the match off a parallel track)
SNAP_OFF_COST = SNAP_RADIUS_M / SNAP_SIGMA_M  # a sample with no track nearby
SNAP_OFF_SWITCH_COST = 5.0   # entering / leaving the track


def snap_to_track(coords_ll, index):
    """Route of (lat, lon) points moved onto the subdivision track where it runs along it.

    The ANTT lines the router uses are coarse — hundreds of metres off the track in hilly
    stretches.  Hidden-Markov map matching: the route is sampled every SNAP_STEP_M, each sample
    may sit on one of the nearest blocks or 'off' the track, and Viterbi picks the sequence
    whose distance to the route and progress along the blocks best agree with the route.
    Stretches on a block are replaced by the block's own points (~30 m apart); stretches off
    the track keep the route.  Returns (coords_ll, fraction of the route on the track)."""
    if index.tree is None or len(coords_ll) < 2:
        return coords_ll, 0.0
    lat, lon = np.asarray(coords_ll, dtype=float).T
    x, y = _to_3857.transform(lon, lat)
    route = shapely.LineString(np.column_stack([x, y]))
    scale = 1 / np.cos(np.radians(np.mean(lat)))
    n = int(min(SNAP_MAX_SAMPLES, max(2, route.length / (SNAP_STEP_M * scale) + 1)))
    dist = np.linspace(0, route.length, n)
    samples = shapely.line_interpolate_point(route, dist)
    step = (dist[1] - dist[0]) / scale

    # Candidates: the SNAP_K nearest blocks within SNAP_RADIUS_M of each sample
    s_idx, b_idx = index.tree.query(samples, predicate='dwithin', distance=SNAP_RADIUS_M * scale)
    if not len(s_idx):
        return coords_ll, 0.0
    lines = np.asarray(index.lines, dtype=object)
    along = shapely.line_locate_point(lines[b_idx], samples[s_idx])
    pts = shapely.get_coordinates(shapely.line_interpolate_point(lines[b_idx], along))
    d = shapely.distance(samples[s_idx], shapely.points(pts)) / scale
    order = np.lexsort((d, s_idx))
    s_idx, b_idx, along, pts, d = s_idx[order], b_idx[order], along[order], pts[order], d[order]
    rank = np.arange(len(s_idx)) - np.searchsorted(s_idx, s_idx)
    keep = rank < SNAP_K
    s_idx, b_idx, along, pts, d, rank = s_idx[keep], b_idx[keep], along[keep], pts[keep], d[keep], rank[keep]

    # State 0 is 'off the track'; states 1..K are the candidates (missing ones cost inf)
    K = SNAP_K + 1
    emit = np.full((n, K), np.inf)
    emit[:, 0] = SNAP_OFF_COST
    emit[s_idx, rank + 1] = d / SNAP_SIGMA_M
    blk = np.full((n, K), -1)
    blk[s_idx, rank + 1] = b_idx
    alo = np.zeros((n, K))
    alo[s_idx, rank + 1] = along
    pxy = np.zeros((n, K, 2))
    pxy[s_idx, rank + 1] = pts

    cost = emit[0].copy()
    back = np.zeros((n, K), dtype=np.int8)
    off_trans = np.full((K, K), SNAP_OFF_SWITCH_COST)
    off_trans[0, 0] = 0.0
    for i in range(1, n):
        if not np.isfinite(emit[i, 1:]).any() and not np.isfinite(cost[1:]).any():
            back[i] = 0  # off the track on both sides: nothing to decide
            cost = cost[0] + emit[i]
            continue
        same = blk[i - 1][:, None] == blk[i][None, :]
        moved = np.where(same, np.abs(alo[i][None, :] - alo[i - 1][:, None]),
                         np.hypot(*(pxy[i][None, :, :] - pxy[i - 1][:, None, :]).transpose(2, 0, 1)))
        trans = np.abs(moved / scale - step) / SNAP_BETA_M + np.where(same, 0.0, SNAP_SWITCH_COST)
        trans[0, :] = off_trans[0, :]
        trans[:, 0] = off_trans[:, 0]
        total = cost[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        cost = total[back[i], np.arange(K)] + emit[i]
    state = np.zeros(n, dtype=int)
    state[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        state[i - 1] = back[i, state[i]]

    # Geometry: block points for runs on a block, the route itself for runs off the track
    sb = blk[np.arange(n), state]
    sa = alo[np.arange(n), state]
    out = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sb[j + 1] == sb[i]:
            j += 1
        if sb[i] < 0:
            piece = substring(route, dist[max(i - 1, 0)], dist[min(j + 1, n - 1)])
        else:
            line = index.lines[sb[i]]
            a0, a1 = sa[i], sa[j]
            # Run the block to its end when the next / previous sample is on another block
            if i > 0 and sb[i - 1] >= 0:
                a0 = _towards_end(line, a0, a1, step * scale)
            if j + 1 < n and sb[j + 1] >= 0:
                a1 = _towards_end(line, a1, a0, step * scale)
            piece = substring(line, a0, a1)
        xy = shapely.get_coordinates(piece)
        if len(xy):
            out.append(xy)
        i = j + 1
    xy = np.concatenate(out)
    keep = np.concatenate([[True], np.any(np.diff(xy, axis=0) != 0, axis=1)])
    xy = xy[keep]
    lons, lats = _to_3857.transform(xy[:, 0], xy[:, 1], direction='INVERSE')
    return list(zip(lats.tolist(), lons.tolist())), float((sb >= 0).mean())


def _towards_end(line, a, other, reach):
    """*a* moved to the block end away from *other*, if that end is within *reach*."""
    end = 0.0 if a < other or (a == other and a < line.length / 2) else line.length
    return end if abs(end - a) <= reach else a


def elevation_profile(coords_ll, index):
    """Elevation along a route of (lat, lon) points.

    Returns None when no part of the route has elevation data, else dict(km, elevation_m
    [NaN where no data], source [subdivision name or ''], coverage, min_m, max_m, ascent_m,
    descent_m, max_grade_pct, max_grade_km, breaks [dict(km, jump_m, from, to)]).
    km is true distance along the route, like the app's route km."""
    if index.tree is None or len(coords_ll) < 2:
        return None
    lat, lon = np.asarray(coords_ll, dtype=float).T
    x, y = _to_3857.transform(lon, lat)
    route = shapely.LineString(np.column_stack([x, y]))
    n = int(min(PROFILE_MAX_SAMPLES, max(2, route.length / PROFILE_STEP_M + 1)))
    dist = np.linspace(0, route.length, n)
    samples = shapely.line_interpolate_point(route, dist)
    # Mercator stretches distances by 1/cos(lat); match distances in true metres
    scale = 1 / np.cos(np.radians(np.mean(lat)))

    s_idx, b_idx = index.tree.query_nearest(samples, max_distance=MATCH_DISTANCE_M * scale, all_matches=False)
    elev = np.full(n, np.nan)
    source = np.full(n, '', dtype=object)
    for b in np.unique(b_idx):
        sel = s_idx[b_idx == b]
        along = shapely.line_locate_point(index.lines[b], samples[sel])
        offset = np.interp(along, index.cumlen[b], index.offsets[b])
        elev[sel] = np.interp(offset, index.grade_ft[b], index.grade_m[b])
        source[sel] = index.source[b]
    if np.isnan(elev).all():
        return None

    # A sample can catch a crossing line at another height for a moment: 5-sample median removes it
    pad = np.pad(elev, 2, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(pad, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN windows where there is no data
        smooth = np.where(np.isnan(elev), np.nan, np.nanmedian(windows, axis=1))

    # True metres along the route: each vertex-to-vertex step scaled by cos(latitude) there
    merc_cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    true_cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y))
                                              * np.cos(np.radians((lat[1:] + lat[:-1]) / 2)))])
    true_m = np.interp(dist, merc_cum, true_cum)
    km = true_m / 1000.0
    spacing = max(true_m[-1] / max(n - 1, 1), 1e-9)
    # Height jumps no track can climb: the route changes between lines at different levels (a
    # grade-separated crossing that the ANTT lines draw as a junction).  Shown as a break, and
    # left out of ascent / descent / grade.
    diffs = np.diff(smooth)
    jump = np.abs(diffs) > max(JUMP_MIN_M, JUMP_MAX_GRADE * spacing)
    breaks = [{'km': float(km[j]), 'jump_m': float(diffs[j]), 'from': source[j], 'to': source[j + 1]}
              for j in np.flatnonzero(jump)]
    smooth_plot = smooth.copy()
    smooth_plot[1:][jump] = np.nan  # gap in the chart at the jump

    have = ~np.isnan(smooth)
    both = have[1:] & have[:-1] & ~jump
    step = max(1, int(round(GRADE_WINDOW_M / spacing)))
    max_grade, max_grade_km = 0.0, None
    if n > step:
        jumps_before = np.concatenate([[0], np.cumsum(jump)])
        g = (smooth[step:] - smooth[:-step]) / (true_m[step:] - true_m[:-step]) * 100
        ok = have[step:] & have[:-step] & (jumps_before[step:] == jumps_before[:-step])
        if ok.any():
            k = np.nanargmax(np.where(ok, np.abs(g), -1))
            max_grade, max_grade_km = float(g[k]), float(km[k])
    return {
        'km': km,
        'elevation_m': smooth_plot,
        'source': source,
        'breaks': breaks,
        'coverage': float(have.mean()),
        'min_m': float(np.nanmin(smooth)),
        'max_m': float(np.nanmax(smooth)),
        'ascent_m': float(diffs[both & (diffs > 0)].sum()),
        'descent_m': float(-diffs[both & (diffs < 0)].sum()),
        'max_grade_pct': max_grade,
        'max_grade_km': max_grade_km,
    }


if __name__ == '__main__':
    # python trackdata.py <cache_dir> <xml> [<xml> ...]
    # Fills the disk cache; the app runs this in a background process at start-up.
    import sys
    for arg in sys.argv[2:]:
        try:
            load_trackdata_cached(arg, sys.argv[1])
        except Exception as e:  # one bad file must not stop the rest
            print(f"{arg}: {e}", file=sys.stderr)
