# MRS Corridor Builder — how to use the map

Guide for an AI assistant (e.g. Claude chat) helping someone use this app. It describes
what every control does, how the common tasks work, what the data means and its known
limits.

**For the assistant:** the user usually writes in Portuguese — answer in Portuguese, but
keep control names in English exactly as written here, because that is how they appear
on screen. You cannot see the app; when unsure, ask the user what they see rather than
guessing. Do not invent features that are not listed here.

---

## 1. What the app is

A Streamlit web app (Python) for Brazilian rail corridors. It shows the national rail
network (ANTT data) by operator, computes shortest rail routes between stations with
rules about which railways to use, lets the user save routes as named "systems", and
overlays detailed onboard track data (speed limits, switches, crossings, km markers…)
from subdivision KMZ exports for MRS and Rumo.

## 2. Starting it

- **From the project folder:** `.venv\Scripts\python.exe -m streamlit run streamlit_app_full.py`,
  then open http://localhost:8501. The `.streamlit/config.toml` in the folder enables the
  required settings (static file serving, Google-style theme).
- **Packaged version:** `MRS_Map.exe` (built with `build_exe.ps1`) starts the same app.
- First opening after a start takes ~10–30 s (loading shapefiles); later clicks take
  1–2 s. Computing a route takes about 0.02 s.

## 3. Screen layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ » Origin (A) [⇄] Destination (B)  Via  [Only|Prefer|Any] [🚂 MRS] [⚙] Compute Commit │  row 1: route
│   🔍 Find a station   🗂️ Saved   🛤️ Subdivisions   (FTC)(Ferroeste)(MRS)(Rumo)…     │  row 2: explore
├──────────────────────────────────────────────────────────────────────────────┤
│                                                             [✕ Hide layers]   │
│                         full-screen map                      layer list        │
│                                                                                │
│              ┌ results card (after Compute) ┐            Zoom · ⛶ · + −       │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **» (top-left):** opens the side panel (management tools). It opens below the top bar;
  its « close button stays pinned at its top while scrolling.
- **Map controls (bottom-right):** zoom level display, fullscreen, zoom in/out.
- **Layer list (top-right):** check/uncheck map layers. **✕ Hide layers / ☰ Layers**
  above it hides or shows the list (remembered per browser).

## 4. Top bar — row 1: build a route

Row 1 is a form: choosing values does nothing until **Compute** or **Commit** is pressed.

| Control | What it does |
|---|---|
| **Origin (A)**, **Destination (B)** | Start and end stations. Type to search. Labels look like `Name [ABC]` (3-letter code); duplicated names get `(idx=N)`. |
| **⇄** | Swaps origin and destination. |
| **Via (optional)** | Intermediate stops, visited in the order chosen. |
| **Routing mode** | **Only** = use only the chosen rail networks. **Prefer** = any track, but other networks cost *penalty × length*. **Any** = shortest distance on any track. |
| **🚂 (network chip)** | Opens chips to choose the rail networks used by Only/Prefer: MRS, Rumo, Vale, VLI, Transnordestina, Ferroeste, FTC. One or several (e.g. MRS + VLI). Default MRS. The chip shows the choice (`🚂 VLI`, `🚂 MRS +1`). Ignored in Any mode. |
| **⚙** | **Penalty for other networks (Prefer mode)** (1–100, default 10), **Route color**, **System name** (default "Corridor 1"), **Path name (optional)**. |
| **Compute** | Calculates the route and draws it (blue preview line + start/end markers). The map zooms to the route. |
| **Commit** | Saves the current preview into the system named in ⚙ → *System name*. A short confirmation appears at the bottom. Saved to `committed_highlights.json` automatically. |

**Results card** (bottom of the map, after Compute): total km, number of stations, the
route, the rule used (e.g. "Only VLI"), timings, and a table per segment with
**Distance (km)** and **By operator (km)** (e.g. "MRS 524.4, VLI 47.2, Unknown 2.6").
Errors such as "No path found…" also appear here; the map stays visible.
**✕** closes the card; the route stays on the map.

**Drawn on the subdivision track:** the ANTT lines the router uses are coarse — in hilly
stretches (e.g. Ferrovia do Aço) they are up to ~750 m off the real track. Where the route
runs along a picked subdivision, the line drawn (and saved on Commit) follows the
subdivision's own track points (~30 m apart) instead; elsewhere it stays on the ANTT line.
The caption says how much ("drawn on the subdivision track for 99 %"). The km figures still
come from the ANTT network.

**⛰️ Elevation** (in the same card, where the route crosses a subdivision in `Subdivs/`):
a height profile along the route (hover for km, height and subdivision), lowest–highest
height, total climb ↑ and descent ↓, and the steepest grade over 1 km with where it is
(e.g. "max grade -2.1 % (km 518)"). "Data on 88 % of the route: …" says which part has
data — stretches without a subdivision are gaps in the chart. Heights come from the
TrackData XML next to each KMZ (the onboard track database); they are GPS (ellipsoid)
heights, about 5–20 m below sea-level altitude in Brazil, so use them for climbs and
grades rather than as exact altitudes.
"⚠️ Height jump of 43 m at km 128.8 (Linha do Centro → Ferrovia do Aco)" means the route
switches between two tracks at different levels: the ANTT lines meet there although the
real tracks may cross on different levels (this one is near Jeceaba, MG). The jump is left
out of the climb and grade figures.

## 5. Top bar — row 2: explore the map (applies immediately)

| Control | What it does |
|---|---|
| **🔍 Find a station** | Pick a station → a red star marks it and the map zooms to it. Clear the field (×) to remove the marker. |
| **🗂️ Saved** | **Saved highlights** switch (show/hide all saved systems in the layer list) and **Station groups** chips (which station-group systems to draw). |
| **🛤️ Subdivisions** | Detailed track data — see section 7. |
| **Operator chips** (FTC, Ferroeste, MRS, Rumo, Transnordestina, VLI, Vale) | Which operators' rail lines are drawn, each in its own colour. MRS is on by default. Hover the help for station counts per operator. |

## 6. Map layers (layer list, top-right)

- **openstreetmap** — base map. **Rail Gauge (OpenRailwayMap)** — optional overlay.
- **System: <name>** — each saved system; **off by default**, tick to show. Shows the
  route lines, white circles at path ends with the station name, and any station
  markers added to the paths.
- **All Brazil Rails** — the whole network in orange (off by default).
- **🚂 <Operator> Rails** — from the operator chips.
- **🛤️ MRS · <subdivision>**, **🛤️ Rumo · <subdivision>** — active subdivisions.
- **All Stations** — all 1,862 stations, with **level of detail by zoom** (like Google Maps):
  - zoom ≤ 7: only landmarks — ⚓ ports, ⇄ railway interchanges, large yards (151);
  - from zoom 8: + cities and terminals (532); names of landmarks from zoom 7;
  - from zoom 10: + small stations, km posts, switches (1,179); all names from zoom 11.
  - Overlapping names are hidden automatically; the more important station keeps its name.
- **Preview Route Track / Preview Stations** — the last computed route.
- **🎯 Selected Station** — the station from Find a station.
- **Station System: <name>** — active station groups (custom shapes/colours).

## 7. Subdivisions (detailed onboard track data)

Open **🛤️ Subdivisions** in row 2.

- Chips grouped by railroad: **MRS** (Pombal-Guaiba, Linha do Centro, Baixada Santista,
  Ferrovia do Aco) and **Rumo** (Serra de Santos, Embu-Guacu, Itu, Campinas, Araraquara,
  Rio Preto, Chapadao do Sul, Rondonopolis, Sao Simao, Rio Verde, Boa Vista, Panorama,
  Colombia, Pereque, Santa Helena, Rianapolis, Uruacu, Porto Nacional). Click to show;
  the map zooms to it. The first time a subdivision is opened it can take a few seconds.
- **Layers** chips:
  - **Speed limits** — track coloured by maximum speed: red ≤ 20, orange 21–30,
    amber 31–40, green 41–50, blue > 50 (legend in the popover). Where the limit differs
    by direction, the stricter one is drawn. Click a segment for limit, train type,
    direction and block.
  - **Track blocks** — the track blocks (VIA 1/2/3, yard tracks) with length and km
    (off by default; Speed limits already draws the track).
  - **No-comm areas** — red dashed segments without communication coverage.
  - **Bridges & tunnels** (from zoom 9) — each bridge or tunnel drawn along the track, from
    its start sign to its end sign, as a wide casing under the speed colour: dark solid =
    bridge, grey dotted = tunnel. Click for "Bridge 86 · 282 m" and its signs/block. A
    sign whose partner is missing shows as a 🌉 / 🚇 icon. Each track of a double line has
    its own segment. Only where the file has those signs: Ferrovia do Aço (116 bridges),
    Baixada Santista and the Rumo subdivisions; Linha do Centro and Pombal-Guaíba have none.
  - **Level crossings** ✕ (from zoom 11), **Switches** (from zoom 12), **KM markers** with
    labels (from zoom 12).
  - Nothing appears below the layer's zoom level; the Zoom box (bottom right) then says
    "Zoom in to 9+ to see Bridges & tunnels". With no subdivision picked the layers show nothing
    (the popover says so).
- **Engineering builds** — use the engineering/test export (version 9xxxxxx) where a
  subdivision has one, instead of the production build (1xxxxxx). The popover lists the
  version in use ("Versions: … Itu 9002003 (eng)").
- **↻ Rescan files** — look for new or changed KMZ files.

Where the files come from: any `.kmz` under the app folder (e.g. `Subdivs/<id>/…/RUMO.00005.1016001.kmz`).
Survey/QA input files inside "QA Artifacts/Inputs" folders are ignored. The file name
pattern is `<RAILROAD>.<subdivision>.<version>`. The `.opk` files are the encrypted
onboard versions of the same data and cannot be read by the app.

## 8. Side panel (») — management

- **🎨 Display → Simplify saved highlights** — draws saved routes with fewer points
  (faster with many paths, slight visual loss). Display only, does not change data.
- **🗂️ Station Group Systems** — mark sets of stations with a shape and colour:
  System name, Group name, Stations in group, Marker shape (circle/square/triangle),
  Marker color → **💾 Save/Update Group**. Also **📤 Export Station Groups**,
  import (upload JSON → **📥 Load Station Groups**), **🗑️ Delete Group**,
  **🧹 Delete System**. Show them on the map with 🗂️ Saved → Station groups.
- **📤 Export All** — writes all saved systems to `exported_highlights.json`.
  **Import JSON highlights** — upload one or more JSON files → **📥 Load into session**
  (paths already present are skipped).
- **🗑️ Clear All Systems** — deletes every saved system (no undo).
- **Manage system** (choose one saved system):
  - rename (text + **✏️**), **📄 Export JSON** (writes `<system>.json`),
    **🗑️ Clear System**;
  - **🔗 Merge** two paths that share a start/end station into one;
  - per path: rename (**✏️**), delete (**✕**), colour (**🎨**);
  - **Add specific stations along this path**: pick from stations within 150 m of the
    path → **➕ Add** → choose marker colour → **✅ Confirm Stations**; remove with
    **🗑️ Remove** / **🧹 Clear All**;
  - paths with > 500 points: **💾 Backup** then **⚡ Simplify** to N points —
    **permanent**, the original coordinates are lost (a backup file is written first).

## 9. Common tasks (step by step)

- **Distance between two stations:** Origin (A) → Destination (B) → Routing mode **Any**
  → **Compute** → read the results card.
- **Route only on MRS track:** mode **Only**, 🚂 = MRS → Compute. If "No path found",
  try **Prefer** (MRS favoured but other track allowed) or add networks in 🚂.
- **Route that may use VLI but prefers MRS:** mode **Prefer**, 🚂 = MRS, ⚙ penalty
  10 (higher = avoid other networks more).
- **Save a corridor:** ⚙ → System name + Path name → Compute → **Commit**. Repeat for
  more paths of the same system. Show it via the layer list ("System: …").
- **Find where a station is:** 🔍 Find a station.
- **See speed limits / switches on a line:** 🛤️ Subdivisions → choose subdivision →
  zoom in (switches and km markers need zoom ≥ 12).
- **Compare railways:** operator chips in row 2 (each operator in its own colour).

## 10. Data and known limits (important when interpreting results)

- **Rail network and stations:** ANTT shapefiles `LInhas/Linhas_BR.shp` and
  `Estacoes/Estacoes.shp`. Operator by ANTT code (CodigoFerr): MRS 11; Rumo 1, 5, 7, 9;
  Vale 3, 4; VLI 8, 12, 13; Transnordestina 2; Ferroeste 6; FTC 10.
- **Distances are 6–8 % too long.** They are measured in the Web Mercator projection
  (EPSG:3857), which stretches distances at Brazil's latitudes; this matches the
  distances in older saved routes. True distance ≈ shown km × cos(latitude), e.g.
  83.2 km shown ≈ 77 km at 22° S.
- **"Unknown" km** in results is track with no operator code in the ANTT data. "Only"
  routing cannot use it — e.g. ~2.6 km near Brisamar breaks "Only MRS" routes through
  Brisamar; use Prefer there.
- **Only the FTC (Tereza Cristina, SC) is not connected** to the rest of the network — it is
  a separate railway. A route between FTC and any other railway fails with "no path". If a
  distance looks too long, check the drawn route and the **By operator (km)** column, and
  use a nearby station on the expected railway or add a Via station.
- **"Only" snaps every station to the nearest track of the chosen networks**, however far.
  A station on another railway can land somewhere unexpected (e.g. "Araguari (idx=15) [LAY]"
  is in Santa Catarina on Rumo; with Only VLI it snaps next to Campinas). Check the station
  code and use Prefer/Any for stations off the chosen network.
- **Several stations share a name** (e.g. "Santos Dumont" exists on MRS and VLI); the
  3-letter code in brackets and "(idx=N)" tell them apart.
- **Subdivision speeds are km/h** (the track databases say so, and their values match the
  km/h in the railways' change orders). Offsets inside the files are in feet.
- Subdivision track lines up with the ANTT lines within ~3 m (median).
- Station "landmark" tiers come from ANTT fields (IndicadorP = port, IndicadorF =
  railway interchange, CodigoEsca = station class) — inferred, not officially documented.

## 11. Files the app writes (in the app folder)

| File / folder | Content |
|---|---|
| `committed_highlights.json` | Saved systems (auto-saved on every change). |
| `station_groups.json` | Station group systems. |
| `exported_highlights.json`, `exported_station_groups.json`, `<system>.json` | Manual exports. |
| `.cache/` | Routing graph and parsed subdivisions (safe to delete; rebuilt automatically). |
| `static/` | Map layer files served to the browser (safe to delete; regenerated). |

## 12. Command-line distance (no browser)

```
.venv\Scripts\python.exe distance_calc.py "Barreiro (idx=947) [FBO]" "Brisamar [FBA]"
.venv\Scripts\python.exe distance_calc.py "Barreiro (idx=947) [FBO]" "Barra do Pirai [FBP]" "Brisamar [FBA]"
.venv\Scripts\python.exe distance_calc.py --mode strict --network MRS "Barreiro (idx=947) [FBO]" "Cel. João Carlos Guedes [FJC]"
.venv\Scripts\python.exe distance_calc.py --mode prefer --network MRS,VLI --penalty 10 "Barreiro (idx=947) [FBO]" "Brisamar [FBA]"
.venv\Scripts\python.exe distance_calc.py --search barreiro    # find exact station names
```
Modes: `allow_all` (default), `prefer`, `strict` (`prefer_mrs` / `strict_mrs` = MRS).
`--network` takes comma-separated names (`MRS,VLI`). Station names must match exactly
(use `--search`); a partial name works only if it matches a single station.

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "No path found" | Only mode too strict, or uncoded track. Use Prefer, add networks, or Any. |
| Layer not visible | Check it in the layer list; switches/km markers need zoom ≥ 12; stations tiers appear by zoom. |
| Layer list covers the map | **✕ Hide layers** (top-right). |
| New KMZ not listed | 🛤️ Subdivisions → **↻ Rescan files**. |
| Map briefly blank after a click | Normal: the map redraws after each change (~1–2 s). |
| App very slow right after start | First load builds caches; wait ~30 s once. |
