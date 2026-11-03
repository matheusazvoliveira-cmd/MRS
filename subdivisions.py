"""
Subdivision track data from KMZ exports (e.g. RUMO.00001.1016001.kmz).

These are the readable exports of the onboard track database (the same
subdivision/version naming as the encrypted .opk files): track blocks,
speed restrictions, switches, level crossings, bridges/tunnels, km markers
and non-communication areas.  load_subdivision() turns one KMZ into map-ready
GeoJSON FeatureCollections (EPSG:4326) whose feature properties carry their
own style (color / weight / dash / icon) and popup HTML.
"""

import hashlib
import html
import math
import pickle
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import shapely
from shapely.ops import linemerge, substring

KML_NS = '{http://earth.google.com/kml/2.1}'

# Speed limit colour bins (upper bound inclusive) — slower is warmer
SPEED_BINS = [
    (20, '#d93025', '≤ 20'),
    (30, '#f29900', '21–30'),
    (40, '#e8a200', '31–40'),
    (50, '#1e8e3e', '41–50'),
    (10_000, '#1a73e8', '> 50'),
]

# Layer id -> (label for the UI, zoom from which it is drawn)
SUBDIV_LAYERS = {
    'speed':      ('Speed limits', 0),
    'track':      ('Track blocks', 0),
    'nca':        ('No-comm areas', 0),
    'structures': ('Bridges & tunnels', 9),
    'crossings':  ('Level crossings', 11),
    'switches':   ('Switches', 12),
    'km':         ('KM markers', 12),
}

_SKIP_DIRS = {'.venv', 'venv', '.cache', 'static', 'build', 'dist', '__pycache__', '.git', '.claude'}

# Railroad by the RRID in the KML document name
RAILROADS = {'150': 'Rumo', '125': 'MRS'}

# Bump when load_subdivision() output changes so disk caches are rebuilt
PARSER_VERSION = 3

# Bridge / tunnel signs.  Rumo writes "#Bridge[#Ponte" / "#Tunnel]#Tunel" ([ = start, ] = end);
# MRS files are free text: "PONTE 12 BEGIN", "BRIDGE-86 END", "PONTE 4A BEGIN", "Bridge 3B" / "Bridge 3E".
_SIGN_BRACKET = re.compile(r'^#(Bridge|Tunnel)([\[\]])')
_SIGN_BEGIN_END = re.compile(r'^(PONTE|BRIDGE|TUNEL|TÚNEL|TUNNEL)[\s-]*(.*?)\s*(BEGIN|END)$', re.I)
_SIGN_B_E = re.compile(r'^(Bridge|Ponte|Tunnel|Tunel)\s+(\w*?\d\w*?)([BE])$', re.I)


def _structure_sign(text):
    """(kind 'Bridge'|'Tunnel', number, is_start) for a bridge/tunnel sign, else None."""
    text = text.strip()
    m = _SIGN_BRACKET.match(text)
    if m:
        return m.group(1), '', m.group(2) == '['
    m = _SIGN_BEGIN_END.match(text) or _SIGN_B_E.match(text)
    if not m:
        return None
    kind = 'Bridge' if m.group(1).upper() in ('PONTE', 'BRIDGE') else 'Tunnel'
    number = m.group(2).strip()
    return kind, ('' if number.lower() == 'unknown' else number), m.group(3).upper() in ('BEGIN', 'B')


def display_name(raw):
    """'FERROVIA DO ACO' -> 'Ferrovia do Aco' (Portuguese prepositions stay lower case)."""
    words = raw.title().split()
    return ' '.join(w.lower() if i and w.lower() in {'do', 'da', 'de', 'dos', 'das', 'e'} else w
                    for i, w in enumerate(words))


def _is_qa_input(path):
    """Survey / QA input files ship next to the exports ('QA Artifacts/Inputs/...') — not track databases."""
    return any(part.lower().startswith('qa ') or part.lower() == 'inputs' for part in path.parts)


def find_kmz_files(base_dir):
    """Every subdivision KMZ under *base_dir*, at any depth (tool folders and QA inputs skipped)."""
    out = []
    for f in Path(base_dir).rglob('*.kmz'):
        rel = f.relative_to(base_dir)
        if not (set(rel.parts) & _SKIP_DIRS) and not _is_qa_input(rel):
            out.append(f)
    return sorted(out)


def read_header(path):
    """Cheap catalogue entry from the first bytes of the KML: name, railroad, subdivision id,
    version and whether it is an engineering build — without parsing the whole file."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        kml = next(n for n in z.namelist() if n.lower().endswith('.kml'))
        with z.open(kml) as fh:
            head = fh.read(4000).decode('utf-8', 'replace')
    title = re.search(r'<name>(.*?)</name>', head, re.S)
    title = re.sub(r'<!\[CDATA\[|\]\]>|\s+', ' ', title.group(1)).strip() if title else path.stem
    m = re.match(r'(.*?)\s*RRID:\s*(\S+)$', title)
    name, rrid = (m.group(1), m.group(2)) if m else (title, '')
    desc = re.search(r'<description>(.*?)</description>', head, re.S)
    info = dict(re.findall(r'^\s*([^:\n]+?):\s*(.+?)\s*$', desc.group(1), re.M)) if desc else {}

    # Version from "<RR>.<subdiv>.<version>" in the file name, else in a parent folder name
    version = ''
    for part in [path.stem] + [p.name for p in path.parents]:
        vm = re.match(r'^[A-Z]+\.\d{5}\.(\d+)', part)
        if vm:
            version = vm.group(1)
            break
    path_text = str(path).lower()
    eng = (len(version) == 7 and version.startswith('9')) or '_eng' in path_text or 'test release' in path_text

    railroad = RAILROADS.get(rrid) or path.stem.split('.')[0].title()
    subdiv_id = int(info.get('Subdivision ID', '0') or 0)
    return {
        'key': f"{railroad}.{subdiv_id:05d}",
        'path': str(path),
        'mtime': path.stat().st_mtime_ns,
        'name': display_name(name),
        'railroad': railroad,
        'subdiv_id': subdiv_id,
        'version': version,
        'eng': eng,
        'info': info,
    }


def catalog(base_dir):
    """Headers of every subdivision KMZ under *base_dir* (unreadable files are listed with 'error')."""
    entries = []
    for f in find_kmz_files(base_dir):
        try:
            entries.append(read_header(f))
        except Exception as e:
            entries.append({'key': None, 'path': str(f), 'error': f"{f.name}: {e}"})
    return entries


def pick_versions(entries, prefer_eng=False):
    """One entry per subdivision: the newest production build (or engineering build when
    *prefer_eng* and one exists).  Returns {key: entry} sorted by railroad and subdivision."""
    by_key = {}
    for e in entries:
        if e.get('key'):
            by_key.setdefault(e['key'], []).append(e)
    picked = {}
    for key, cands in by_key.items():
        pool = [c for c in cands if c['eng'] == prefer_eng] or cands
        picked[key] = max(pool, key=lambda c: (int(c['version'] or 0), c['mtime']))
    return dict(sorted(picked.items(), key=lambda kv: (kv[1]['railroad'], kv[1]['subdiv_id'])))


def _cache_path(path, cache_dir):
    path = Path(path)
    st_ = path.stat()
    key = f"{path.resolve()}|{st_.st_size}|{st_.st_mtime_ns}|v{PARSER_VERSION}"
    return Path(cache_dir) / f"subdiv_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"


def is_cached(path, cache_dir):
    """True if load_subdivision_cached() would read *path* from the disk cache."""
    return _cache_path(path, cache_dir).exists()


def load_subdivision_cached(path, cache_dir):
    """load_subdivision() with a disk cache keyed by file path, size, mtime and PARSER_VERSION."""
    path = Path(path)
    cache_path = _cache_path(path, cache_dir)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupt cache — parse again
    data = load_subdivision(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except OSError:
        pass
    return data


def speed_color(limit):
    for upper, color, _ in SPEED_BINS:
        if limit <= upper:
            return color
    return SPEED_BINS[-1][1]


def _data(pm):
    return {d.get('name'): (d.findtext(KML_NS + 'value') or '').strip() for d in pm.iter(KML_NS + 'Data')}


def _coords(pm):
    """[[lon, lat], ...] of a placemark's (first) geometry, rounded to ~1 m."""
    text = pm.findtext('.//' + KML_NS + 'coordinates') or ''
    out = []
    for c in text.split():
        parts = c.split(',')
        out.append([round(float(parts[0]), 6), round(float(parts[1]), 6)])
    return out


def _geom_type(pm):
    for tag in ('Point', 'LineString', 'Polygon'):
        if pm.find('.//' + KML_NS + tag) is not None:
            return tag
    return None


def _popup(title, rows):
    body = ''.join(f"<br>{html.escape(k)}: {html.escape(str(v))}" for k, v in rows if v not in (None, ''))
    return f"<b>{html.escape(title)}</b>{body}"


def _feature(geom_type, coords, **props):
    geometry = {'type': geom_type, 'coordinates': coords[0] if geom_type == 'Point' else coords}
    return {'type': 'Feature', 'geometry': geometry, 'properties': props}


STRUCTURE_STYLE = {  # drawn as a wide casing under the speed-limit line
    'Bridge': {'color': '#202124', 'weight': 12, 'opacity': 0.85},
    'Tunnel': {'color': '#9aa0a6', 'weight': 12, 'opacity': 0.9, 'dash': '2 6'},
}
_STRUCTURE_MAX_PAIR_M = 10_000   # a start sign pairs with an end sign at most this far away
_ON_TRACK_DEG = 0.0003           # ~30 m: both signs must be this close to one track line


def _length_m(coords):
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin((p2 - p1) / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
        total += 2 * 6_371_000 * math.asin(math.sqrt(a))
    return total


def _structure_segments(signs, track_lines):
    """Bridge / tunnel features: each start sign paired with its end sign and drawn along the track.

    *signs* are (kind, number, is_start, [lon, lat], data).  Numbered signs pair by number;
    unnumbered ones (Rumo) pair with the nearest free end sign of the same kind.  The segment
    follows the track block (or merged track) both signs lie on, else a straight line.
    """
    seen, unique = set(), []
    for s in signs:  # the same sign is often exported once per direction
        key = (s[0], s[1], s[2], round(s[3][0], 5), round(s[3][1], 5))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    starts = [s for s in unique if s[2]]
    ends = [s for s in unique if not s[2]]
    candidates = sorted(  # same track block first (parallel tracks carry their own signs)
        (a[4].get('blockId') != b[4].get('blockId'), _length_m([a[3], b[3]]), i, j)
        for i, a in enumerate(starts) for j, b in enumerate(ends)
        if a[0] == b[0] and a[1] == b[1] and _length_m([a[3], b[3]]) <= _STRUCTURE_MAX_PAIR_M)
    used_s, used_e, pairs = set(), set(), []
    for _, _, i, j in candidates:  # closest pairs first
        if i not in used_s and j not in used_e:
            used_s.add(i)
            used_e.add(j)
            pairs.append((starts[i], ends[j]))

    lines = [shapely.LineString(c) for c in track_lines if len(c) >= 2]
    lines += list(getattr(linemerge(lines), 'geoms', [])) if lines else []  # blocks joined end to end
    tree = shapely.STRtree(lines) if lines else None

    features = []
    for start, end in pairs:
        kind, number, _, a, d = start
        p, q = shapely.Point(a), shapely.Point(end[3])
        coords = [a, end[3]]
        if tree is not None:
            near = [lines[k] for k in tree.query(shapely.MultiPoint([a, end[3]]).envelope.buffer(_ON_TRACK_DEG))]
            on_both = [ln for ln in near if ln.distance(p) <= _ON_TRACK_DEG and ln.distance(q) <= _ON_TRACK_DEG]
            if on_both:
                line = min(on_both, key=lambda ln: ln.distance(p) + ln.distance(q))
                da, db = sorted((line.project(p), line.project(q)))
                part = substring(line, da, db)
                if part.geom_type == 'LineString' and len(part.coords) >= 2:
                    coords = [[round(x, 6), round(y, 6)] for x, y in part.coords]
        length = _length_m(coords)
        title = f"{kind}{' ' + number if number else ''} · {length:,.0f} m"
        features.append(_feature('LineString', coords, **STRUCTURE_STYLE[kind], popup=_popup(title, [
            ('Start sign', d.get('text')), ('End sign', end[4].get('text')),
            ('Block', d.get('blockId')), ('Offset', d.get('offset')),
        ])))
    for s in [s for i, s in enumerate(starts) if i not in used_s] + [s for j, s in enumerate(ends) if j not in used_e]:
        kind, number, is_start, a, d = s  # a sign without its partner: keep it as an icon
        features.append(_feature('Point', [a], icon='🚇' if kind == 'Tunnel' else '🌉', color='#ffffff', popup=_popup(
            f"{kind}{' ' + number if number else ''} {'start' if is_start else 'end'} (no matching sign)",
            [('Sign', d.get('text')), ('Block', d.get('blockId')), ('Offset', d.get('offset'))])))
    return features


def _paired(folder):
    """(point_data, line_coords) pairs from a folder that alternates a data Point and its LineString."""
    pending = None
    for pm in folder.findall(KML_NS + 'Placemark'):
        kind = _geom_type(pm)
        if kind == 'Point':
            pending = _data(pm)
        elif kind == 'LineString' and pending is not None:
            yield pending, _coords(pm)
            pending = None


def load_subdivision(path):
    """Parse one subdivision KMZ.  Returns dict(name, info, bounds, layers={id: FeatureCollection})."""
    with zipfile.ZipFile(path) as z:
        kml_name = next(n for n in z.namelist() if n.lower().endswith('.kml'))
        raw = z.read(kml_name)
    if b'<!DOCTYPE' in raw[:2000] or b'<!ENTITY' in raw:
        raise ValueError(f"{Path(path).name}: KML with DOCTYPE/ENTITY declarations is not supported")
    doc = ET.fromstring(raw).find(KML_NS + 'Document')

    title = re.sub(r'\s+', ' ', ''.join(doc.find(KML_NS + 'name').itertext())).strip()  # "SERRA DE SANTOS RRID: 150"
    m = re.match(r'(.*?)\s*RRID:\s*(\S+)$', title)
    name = display_name(m.group(1) if m else title)
    description = doc.findtext(KML_NS + 'description') or ''
    info = dict(re.findall(r'^\s*([^:\n]+?):\s*(.+?)\s*$', description, flags=re.M))
    if m:
        info['RRID'] = m.group(2)
    folders = {(f.findtext(KML_NS + 'name') or '').strip(): f for f in doc.findall(KML_NS + 'Folder')}
    layers = {key: [] for key in SUBDIV_LAYERS}

    if 'Line Markings' in folders:
        for d, coords in _paired(folders['Line Markings']):
            km = f"{d.get('StartMilepost', '')} – {d.get('EndMilepost', '')}" if d.get('StartMilepost') else ''
            layers['track'].append(_feature(
                'LineString', coords, color='#3c4043', weight=3,
                popup=_popup(f"Block {d.get('blockId', '')}", [
                    ('Track', d.get('TGroupType')), ('Length', d.get('Length')), ('KM', km),
                    ('Previous', f"{d.get('PreviousType', '')} {d.get('PreviousElement', '')}".strip()),
                    ('Next', f"{d.get('NextType', '')} {d.get('NextElement', '')}".strip()),
                ])))

    if 'Speed Restrictions' in folders:
        speeds = []
        for d, coords in _paired(folders['Speed Restrictions']):
            try:
                limit = int(float(d.get('slimit', '')))
            except ValueError:
                continue
            speeds.append((limit, _feature(
                'LineString', coords, color=speed_color(limit), weight=6,
                popup=_popup(f"Speed limit {limit}", [
                    ('Train', d.get('ttype')), ('Direction', d.get('direction')), ('Block', d.get('blockId')),
                    ('Offset', f"{d.get('soffset', '')}–{d.get('eoffset', '')}"),
                ]))))
        # Lowest limit drawn last, so where directions differ the stricter one shows
        layers['speed'] = [f for _, f in sorted(speeds, key=lambda t: -t[0])]

    if 'NonCommArea' in folders:
        for d, coords in _paired(folders['NonCommArea']):
            layers['nca'].append(_feature(
                'LineString', coords, color='#d93025', weight=5, dash='6 6',
                popup=_popup('No-communication area', [
                    ('Block', d.get('blockId')), ('Offset', f"{d.get('soffset', '')}–{d.get('eoffset', '')}"),
                ])))

    for pm in (folders['Switches'].findall(KML_NS + 'Placemark') if 'Switches' in folders else []):
        d = _data(pm)
        layers['switches'].append(_feature(
            'Point', _coords(pm), color='#9334e6', radius=5,
            label=None,
            popup=_popup(f"Switch {d.get('Namen', '')}", [
                ('ID', d.get('SwitchID')), ('Orientation', d.get('SwitchOrient')), ('Elevation', d.get('Elevation')),
            ])))

    for pm in (folders['Crossings'].findall(KML_NS + 'Placemark') if 'Crossings' in folders else []):
        d = _data(pm)
        layers['crossings'].append(_feature(
            'Point', _coords(pm), icon='✕', color='#f29900',
            popup=_popup(d.get('crossingname') or 'Level crossing', [
                ('Type', d.get('crossingtype')), ('Width', d.get('width')),
                ('Block', d.get('blockId')), ('Offset', d.get('offset')),
            ])))

    signs = []
    for pm in (folders['Signages'].findall(KML_NS + 'Placemark') if 'Signages' in folders else []):
        d = _data(pm)
        parsed = _structure_sign(d.get('text') or '')
        if parsed:
            signs.append((*parsed, _coords(pm)[0], d))
    track_lines = [f['geometry']['coordinates'] for f in layers['track']]
    layers['structures'] = _structure_segments(signs, track_lines)

    for pm in (folders['Milepost Markers'].findall(KML_NS + 'Placemark') if 'Milepost Markers' in folders else []):
        d = _data(pm)
        layers['km'].append(_feature(
            'Point', _coords(pm), color='#5f6368', radius=3, label=f"km {d.get('text', '')}",
            popup=_popup(f"KM {d.get('text', '')}", [('Block', d.get('blockId')), ('Offset', d.get('offset'))])))

    # Bounds from the track (falls back to every coordinate)
    pts = [c for f in (layers['track'] or sum(layers.values(), [])) for c in
           (f['geometry']['coordinates'] if f['geometry']['type'] == 'LineString' else [f['geometry']['coordinates']])]
    bounds = ([[min(p[1] for p in pts), min(p[0] for p in pts)], [max(p[1] for p in pts), max(p[0] for p in pts)]]
              if pts else None)

    return {
        'name': name,
        'info': info,
        'bounds': bounds,
        'layers': {key: {'type': 'FeatureCollection', 'features': feats} for key, feats in layers.items()},
    }


if __name__ == '__main__':
    # python subdivisions.py <cache_dir> <kmz> [<kmz> ...]
    # Fills the disk cache; the app runs this in a background process at start-up.
    import sys

    cache_dir, *kmz_paths = sys.argv[1:]
    for kmz in kmz_paths:
        try:
            load_subdivision_cached(kmz, cache_dir)
        except Exception as e:
            print(f"skipped {kmz}: {e}", file=sys.stderr)
