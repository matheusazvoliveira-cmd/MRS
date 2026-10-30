"""
MRS Distance Calculator — Standalone CLI
=========================================
Calculate rail distance between stations without launching the Streamlit app.

Usage examples
--------------
# Basic: distance from A to B
python distance_calc.py "Barreiro (idx=947) [FBO]" "Brisamar [FBA]"

# Multi-stop: A → via → B
python distance_calc.py "Barreiro (idx=947) [FBO]" "Barra do Pirai [FBP]" "Brisamar [FBA]"

# Restrict to MRS track only
python distance_calc.py --mode strict "Barreiro (idx=947) [FBO]" "Cel. João Carlos Guedes [FJC]"

# Prefer MRS + VLI track (other networks penalised)
python distance_calc.py --mode prefer --network MRS,VLI "Barreiro (idx=947) [FBO]" "Brisamar [FBA]"

# List all available station names
python distance_calc.py --list-stations

# Fuzzy search for station name
python distance_calc.py --search "belo"

Options
-------
--mode          Routing mode: allow_all (default) | prefer | strict
                (prefer_mrs / strict_mrs still work and mean --network MRS)
--network       Rail networks for prefer/strict, comma-separated: MRS (default), Rumo,
                Vale, VLI, Transnordestina, Ferroeste, FTC — e.g. MRS,VLI
--penalty       Penalty multiplier for other networks (prefer mode), default=10

Routing uses the national rail graph from routing.py, cached on disk in .cache/
(shared with the Streamlit app), so only the very first run builds it.
"""

import argparse
import sys
from pathlib import Path

import routing
from routing import ALLOW_ALL, OTHER_NETWORK_PENALTY_DEFAULT, RAIL_OPERATORS

_BASE = Path(__file__).parent.resolve()

RAILS_SHP  = _BASE / "LInhas"  / "Linhas_BR.shp"
STATIONS_SHP = _BASE / "Estacoes" / "Estacoes.shp"
GRAPH_CACHE_DIR = _BASE / ".cache"


def load_stations():
    """Station lookup (EPSG:3857, indexed by label) and sorted labels."""
    stations_lookup = routing.prepare_stations(routing.read_stations(STATIONS_SHP)).set_index('station_label')
    return stations_lookup, sorted(stations_lookup.index)


CODIGO_OPERATOR = {
    11: "MRS",
    1: "Rumo (Malha Sul)", 5: "Rumo (Paulista)", 7: "Rumo (Oeste)", 9: "Rumo (Norte)",
    3: "Vale (EFC)", 4: "Vale (EFVM)",
    8: "VLI (FCA)", 12: "VLI (FNS Norte)", 13: "VLI (FNS Central)",
    2: "Transnordestina (CFN)",
    6: "Ferroeste",
    10: "FTC",
}


# ---------------------------------------------------------------------------
# Main calculation routine
# ---------------------------------------------------------------------------
def calculate_distance(waypoints, routing_mode=ALLOW_ALL, penalty_mult=OTHER_NETWORK_PENALTY_DEFAULT,
                       networks=("MRS",), verbose=True):
    """
    Calculate the shortest rail distance along a sequence of waypoints.

    Parameters
    ----------
    waypoints : list[str]
        Ordered station labels, e.g. ["Barreiro (idx=947) [FBO]", "Brisamar [FBA]"]
    routing_mode : str
        "allow_all" | "prefer" | "strict" (legacy "prefer_mrs" / "strict_mrs" = MRS)
    penalty_mult : float
        Penalty multiplier for track outside *networks* (only used with prefer mode)
    networks : sequence of str
        Operator names from routing.RAIL_OPERATORS used by prefer / strict
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        total_km      : float — total distance
        segment_km    : list[float] — distance per waypoint segment
        breakdown     : dict[operator_label -> km]
        order         : list[str] — waypoints in order
    """
    if verbose:
        print("\nLoading stations...")
    stations_lookup, labels = load_stations()

    # Resolve station names
    resolved = []
    for wp in waypoints:
        if wp in stations_lookup.index:
            resolved.append(wp)
        else:
            # Case-insensitive fuzzy match
            matches = [l for l in labels if wp.lower() in l.lower()]
            if len(matches) == 1:
                print(f"  '{wp}' → matched as '{matches[0]}'")
                resolved.append(matches[0])
            elif len(matches) > 1:
                print(f"  Ambiguous station '{wp}'. Matches:")
                for m in matches[:10]:
                    print(f"    - {m}")
                if len(matches) > 10:
                    print(f"    ... and {len(matches) - 10} more")
                sys.exit(1)
            else:
                print(f"  Station not found: '{wp}'")
                print("  Use --search to look up station names.")
                sys.exit(1)

    if verbose:
        print(f"\nWaypoints: {' → '.join(resolved)}")
        print(f"Routing mode : {routing_mode}" + ("" if routing_mode == ALLOW_ALL else f" ({' + '.join(networks)})"))
        print("Loading routing graph...")

    graph = routing.load_or_build_graph(
        lambda: routing.read_rails(RAILS_SHP).to_crs(epsg=3857), RAILS_SHP, GRAPH_CACHE_DIR,
        progress_cb=(lambda msg: print(f"  {msg}")) if verbose else None,
    )

    points_xy = []
    for lab in resolved:
        row = stations_lookup.loc[lab]
        points_xy.append((float(row.geometry.x), float(row.geometry.y)))

    if verbose:
        print("\nComputing shortest paths...")
    try:
        result = routing.route(graph, points_xy, routing_mode, penalty_mult,
                               codes=routing.network_codes(networks))
    except routing.NoPathError as e:
        i = e.segment_index
        print(f"  ERROR: No path found for segment '{resolved[i]}' → '{resolved[i+1]}'")
        print("  Try: --mode prefer, --mode allow_all, or add networks with --network")
        sys.exit(1)

    segment_km = result['segment_km']
    total_km = result['total_km']
    if verbose:
        for i, km in enumerate(segment_km):
            print(f"  {resolved[i]} → {resolved[i+1]} : {km:.2f} km")

    breakdown_raw = {}
    for codes in result['segment_codes']:
        for codigo, km in codes.items():
            breakdown_raw[codigo] = breakdown_raw.get(codigo, 0.0) + km

    # Map raw codes to operator names
    breakdown = {}
    for codigo, km in breakdown_raw.items():
        try:
            code_int = int(codigo)
        except (TypeError, ValueError):
            code_int = None
        label = CODIGO_OPERATOR.get(code_int, f"Code {codigo}" if codigo is not None else "Unknown")
        breakdown[label] = breakdown.get(label, 0.0) + km

    return {
        'total_km': total_km,
        'segment_km': segment_km,
        'breakdown': breakdown,
        'order': resolved,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate MRS rail distance between stations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('waypoints', nargs='*', help='Station names in order (origin via... destination)')
    parser.add_argument('--mode', choices=['allow_all', 'prefer', 'strict', 'prefer_mrs', 'strict_mrs'],
                        default='allow_all', help='Routing mode (default: allow_all)')
    # Comma-separated, not nargs='+' (that would swallow the station names that follow)
    parser.add_argument('--network', default='MRS', metavar='NAMES',
                        help='Rail networks for prefer/strict, comma-separated (default: MRS). Choices: '
                             + ', '.join(RAIL_OPERATORS))
    parser.add_argument('--penalty', type=float, default=OTHER_NETWORK_PENALTY_DEFAULT,
                        help='Penalty multiplier for other networks in prefer mode (default: 10)')
    # Kept so existing commands keep working; the precomputed graph needs no corridor/noding options
    parser.add_argument('--no-corridor', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--buffer-km', type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument('--no-node', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--list-stations', action='store_true',
                        help='Print all available station names and exit')
    parser.add_argument('--search', metavar='TERM',
                        help='Search for stations matching TERM (case-insensitive)')

    args = parser.parse_args()

    if args.list_stations:
        print("Loading stations...")
        _, labels = load_stations()
        print(f"\n{len(labels)} stations available:\n")
        for lbl in labels:
            print(f"  {lbl}")
        return

    if args.search:
        print(f"Loading stations and searching for '{args.search}'...")
        _, labels = load_stations()
        term = args.search.lower()
        matches = [l for l in labels if term in l.lower()]
        if matches:
            print(f"\n{len(matches)} match(es) for '{args.search}':\n")
            for m in matches:
                print(f"  {m}")
        else:
            print(f"No stations found matching '{args.search}'.")
        return

    # Network names are case-insensitive
    by_lower = {name.lower(): name for name in RAIL_OPERATORS}
    requested = [n.strip() for n in args.network.split(',') if n.strip()]
    unknown = [n for n in requested if n.lower() not in by_lower]
    if unknown:
        parser.error(f"Unknown network(s): {', '.join(unknown)}. Choices: {', '.join(RAIL_OPERATORS)}")
    networks = [by_lower[n.lower()] for n in requested]

    if len(args.waypoints) < 2:
        parser.error("Provide at least two station names (origin and destination).")

    result = calculate_distance(
        waypoints=args.waypoints,
        routing_mode=args.mode,
        penalty_mult=args.penalty,
        networks=networks,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"  Route   : {' → '.join(result['order'])}")
    print(f"  Mode    : {args.mode}")
    print(f"  TOTAL   : {result['total_km']:.2f} km")
    if len(result['segment_km']) > 1:
        print("\n  Segments:")
        for i, km in enumerate(result['segment_km']):
            print(f"    {result['order'][i]} → {result['order'][i+1]} : {km:.2f} km")
    print("\n  Breakdown by operator:")
    for op, km in sorted(result['breakdown'].items(), key=lambda x: -x[1]):
        pct = km / result['total_km'] * 100 if result['total_km'] > 0 else 0
        print(f"    {op:<35} {km:>8.2f} km  ({pct:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
