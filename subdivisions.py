"""
Subdivision track data from KMZ exports (e.g. RUMO.00001.1016001.kmz).

These are the readable exports of the onboard track database (the same
subdivision/version naming as the encrypted .opk files): track blocks,
speed restrictions, switches, level crossings, bridges/tunnels, km markers
and non-communication areas.  load_subdivision() turns one KMZ into map-ready
GeoJSON FeatureCollections (EPSG:4326) whose feature properties carry their
own style (color / weight / dash / icon) and popup HTML.
"""

import html
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

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
    'structures': ('Bridges & tunnels', 11),
    'crossings':  ('Level crossings', 11),
    'switches':   ('Switches', 12),
    'km':         ('KM markers', 12),
}

_SKIP_DIRS = {'.venv', 'venv', '.cache', 'static', 'build', 'dist', '__pycache__', '.git'}


def find_kmz_files(base_dir):
    """KMZ files in *base_dir* and its direct subfolders (generated/tool folders skipped)."""
    base = Path(base_dir)
    files = list(base.glob('*.kmz'))
    for sub in base.iterdir():
        if sub.is_dir() and sub.name not in _SKIP_DIRS:
            files.extend(sub.glob('*.kmz'))
    return sorted(files)


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
    name = (m.group(1) if m else title).title()
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

    structures = {'Tunnel[': ('🚇', 'Tunnel start'), 'Tunnel]': ('🚇', 'Tunnel end'),
                  'Bridge[': ('🌉', 'Bridge start'), 'Bridge]': ('🌉', 'Bridge end')}
    for pm in (folders['Signages'].findall(KML_NS + 'Placemark') if 'Signages' in folders else []):
        d = _data(pm)
        tag = (d.get('text') or '').split('#')[1] if (d.get('text') or '').startswith('#') else ''
        if tag in structures:
            icon, title = structures[tag]
            layers['structures'].append(_feature(
                'Point', _coords(pm), icon=icon, color='#ffffff',
                popup=_popup(title, [('Block', d.get('blockId')), ('Offset', d.get('offset'))])))

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
