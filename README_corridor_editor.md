# MRS Corridor JSON Editor

A visual editor for manipulating saved corridor JSON files without route recalculation.

## What It Does

Load existing corridor JSON files (like "Carrossel do Minerio.json") and:
- **Change colors** of paths
- **Rename** systems and paths
- **Move paths** between systems
- **Delete** paths or entire systems
- **Reorganize** corridor structure
- **Preview** all changes on an interactive map

**No route calculation** — pure visualization and editing of existing saved corridors.

## How to Run

```powershell
# Activate the virtual environment
& .venv\Scripts\Activate.ps1

# Run the corridor editor
python -m streamlit run corridor_editor.py
```

Or from the network:
- Point your browser to `http://10.242.137.200:8502` (if running on server)

## How to Use

### 1. Load a File
- **Sidebar** → Select a corridor JSON file from dropdown
- Click **"📥 Load File"**
- The app will load all systems and paths

### 2. Edit Systems & Paths
- **Select a system** from the dropdown
- **Rename the system** using the expander
- **Edit individual paths**:
  - Change color using the color picker
  - Rename the path
  - Move to a different system
  - Delete the path
- **Delete entire systems** if needed

### 3. Preview on Map
- The map shows all visible systems/paths in real-time
- Toggle station markers on/off
- Filter which systems are visible
- Changes appear immediately on the map

### 4. Save Your Work
- Choose **"Overwrite original"** or **"Save as new file"**
- Click **"💾 Save File"**
- The app tracks unsaved changes with a warning indicator

## Features

✅ **Load** any corridor JSON from workspace  
✅ **Visual editing** with instant map preview  
✅ **Color picker** for each path  
✅ **Rename** systems and paths  
✅ **Reorganize** paths between systems  
✅ **Statistics** - total systems, paths, distance  
✅ **Safe saving** - option to save as new file  

## File Structure

The editor works with JSON files in this format:
```json
{
  "System Name": [
    {
      "coords_ll": [[lat, lon], ...],
      "route_color": "#1f77b4",
      "path_name": "Station A → Station B",
      "total_km": 195.82,
      "preview_station_points": [...],
      "selected_station_markers": [...]
    }
  ]
}
```

## Differences from Main App

| Feature | Main App (`streamlit_app_full.py`) | Corridor Editor (`corridor_editor.py`) |
|---------|-------------------------------------|----------------------------------------|
| Route calculation | ✅ Yes (A* pathfinding) | ❌ No - loads existing |
| Shapefile loading | ✅ Yes (rails, stations) | ❌ No - JSON only |
| Graph building | ✅ Yes | ❌ No |
| Edit corridors | ❌ No - view only | ✅ Yes - full editing |
| Change colors | Limited (preview only) | ✅ Yes - persistent |
| Reorganize structure | ❌ No | ✅ Yes |

## Use Cases

- **Refine presentations**: Adjust colors for better visibility
- **Reorganize corridors**: Group related routes into systems
- **Clean up data**: Remove duplicate or test paths
- **Merge projects**: Combine paths from multiple JSON files
- **Rename for clarity**: Update system/path names for reports

## Technical Notes

- Uses the same Folium map rendering as the main app
- No dependency on shapefiles - pure JSON manipulation
- Changes are stored in memory until you explicitly save
- Original files are preserved unless you choose to overwrite
