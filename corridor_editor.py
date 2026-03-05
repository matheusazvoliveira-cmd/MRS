"""
MRS Corridor JSON Editor

Load existing corridor JSON files and edit:
- System names
- Path names & colors
- Move paths between systems
- Join/separate paths
- Reorganize structure

No route calculation - pure visualization and editing of saved corridors.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from pathlib import Path
import copy

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
LABEL_FONT_SIZE = 17

# Brazil bounds for map
_BR_BOUNDS = [[-33.75, -73.99], [5.27, -28.84]]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_corridor_json_files():
    """Find all corridor JSON files in workspace (exclude config files)."""
    exclude = {'station_groups.json', 'exported_station_groups.json', 'committed_highlights.json'}
    files = []
    
    # Root level JSONs
    for f in BASE_DIR.glob('*.json'):
        if f.name not in exclude:
            files.append(f)
    
    # Subdirectory JSONs
    for f in BASE_DIR.glob('*/*.json'):
        if f.name not in exclude:
            files.append(f)
    
    return sorted(files, key=lambda x: x.name)


def load_corridor_json(filepath):
    """Load corridor JSON file and return parsed data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_corridor_json(filepath, data):
    """Save corridor data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def render_path_on_map(m, path_data, path_color, path_label, show_stations=True):
    """Render a single path on the folium map."""
    if not path_data.get('coords_ll'):
        return
    
    coords = path_data['coords_ll']
    
    # Draw the route polyline
    folium.PolyLine(
        locations=coords,
        color=path_color,
        weight=4,
        opacity=0.8,
        popup=f"<b>{path_label}</b><br>{path_data.get('total_km', 0):.2f} km",
        tooltip=path_label,
    ).add_to(m)
    
    # Add station markers if requested
    if show_stations:
        # Preview stations (endpoints)
        for station in path_data.get('preview_station_points', []):
            folium.CircleMarker(
                location=[station['lat'], station['lon']],
                radius=6,
                color='#ffffff',
                fill=True,
                fill_color=path_color,
                fill_opacity=1.0,
                weight=2,
                popup=station['label'],
                tooltip=station['label'],
            ).add_to(m)
        
        # Selected station markers (waypoints/highlights)
        for marker in path_data.get('selected_station_markers', []):
            folium.CircleMarker(
                location=[marker['lat'], marker['lon']],
                radius=5,
                color='#ffffff',
                fill=True,
                fill_color=marker.get('color', path_color),
                fill_opacity=0.9,
                weight=2,
                popup=marker['label'],
                tooltip=marker['label'],
            ).add_to(m)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="MRS Corridor Editor",
        page_icon="🛤️",
        layout="wide",
    )
    
    st.title("🛤️ MRS Corridor JSON Editor")
    st.markdown("Load and edit existing corridor JSON files — change colors, rename systems/paths, reorganize structure.")
    
    # Initialize session state
    if 'corridor_data' not in st.session_state:
        st.session_state.corridor_data = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'modified' not in st.session_state:
        st.session_state.modified = False
    
    # Sidebar - File selection and editing
    with st.sidebar:
        st.header("📂 File Manager")
        
        # Find all corridor JSON files
        json_files = find_corridor_json_files()
        if not json_files:
            st.error("No corridor JSON files found in workspace")
            return
        
        file_options = {str(f.relative_to(BASE_DIR)): f for f in json_files}
        
        selected_file_rel = st.selectbox(
            "Select corridor JSON file:",
            options=list(file_options.keys()),
            key="file_selector"
        )
        
        selected_file = file_options[selected_file_rel]
        
        # Load button
        if st.button("📥 Load File", type="primary"):
            try:
                data = load_corridor_json(selected_file)
                st.session_state.corridor_data = data
                st.session_state.current_file = selected_file
                st.session_state.modified = False
                st.success(f"Loaded {len(data)} system(s)")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        
        st.divider()
        
        # Only show editing options if data is loaded
        if st.session_state.corridor_data:
            st.header("✏️ Edit Systems & Paths")
            
            data = st.session_state.corridor_data
            
            # System selector
            system_names = list(data.keys())
            if not system_names:
                st.warning("No systems in this file")
            else:
                selected_system = st.selectbox(
                    "Select system to edit:",
                    options=system_names,
                    key="system_selector"
                )
                
                if selected_system:
                    st.subheader(f"📍 {selected_system}")
                    
                    paths = data[selected_system]
                    st.caption(f"{len(paths)} path(s) in this system")
                    
                    # Rename system
                    with st.expander("🏷️ Rename System"):
                        new_system_name = st.text_input(
                            "New system name:",
                            value=selected_system,
                            key=f"rename_system_{selected_system}"
                        )
                        if st.button("Apply Rename", key=f"btn_rename_{selected_system}"):
                            if new_system_name != selected_system and new_system_name.strip():
                                data[new_system_name] = data.pop(selected_system)
                                st.session_state.modified = True
                                st.rerun()
                    
                    # Edit individual paths
                    st.divider()
                    st.subheader("🛤️ Edit Paths")
                    
                    for idx, path in enumerate(paths):
                        path_name = path.get('path_name', f'Path {idx+1}')
                        current_color = path.get('route_color', '#1f77b4')
                        
                        with st.expander(f"Path {idx+1}: {path_name}"):
                            # Color picker
                            new_color = st.color_picker(
                                "Path color:",
                                value=current_color,
                                key=f"color_{selected_system}_{idx}"
                            )
                            if new_color != current_color:
                                path['route_color'] = new_color
                                st.session_state.modified = True
                            
                            # Rename path
                            new_path_name = st.text_input(
                                "Path name:",
                                value=path_name,
                                key=f"pathname_{selected_system}_{idx}"
                            )
                            if new_path_name != path_name:
                                path['path_name'] = new_path_name
                                st.session_state.modified = True
                            
                            # Move to different system
                            other_systems = [s for s in system_names if s != selected_system]
                            if other_systems:
                                move_to = st.selectbox(
                                    "Move to system:",
                                    options=["(keep here)"] + other_systems,
                                    key=f"move_{selected_system}_{idx}"
                                )
                                if st.button(f"Move path", key=f"btn_move_{selected_system}_{idx}"):
                                    if move_to != "(keep here)":
                                        data[move_to].append(path)
                                        paths.pop(idx)
                                        st.session_state.modified = True
                                        st.rerun()
                            
                            # Delete path
                            if st.button(f"🗑️ Delete this path", key=f"del_{selected_system}_{idx}", type="secondary"):
                                paths.pop(idx)
                                st.session_state.modified = True
                                st.rerun()
                    
                    # Delete entire system
                    st.divider()
                    if st.button(f"🗑️ Delete entire system: {selected_system}", type="secondary"):
                        del data[selected_system]
                        st.session_state.modified = True
                        st.rerun()
            
            st.divider()
            
            # Save section
            st.header("💾 Save Changes")
            
            if st.session_state.modified:
                st.warning("⚠️ Unsaved changes")
            else:
                st.success("✅ No unsaved changes")
            
            save_option = st.radio(
                "Save as:",
                options=["Overwrite original", "Save as new file"],
                key="save_option"
            )
            
            if save_option == "Save as new file":
                new_filename = st.text_input(
                    "New filename:",
                    value=f"{st.session_state.current_file.stem}_edited.json",
                    key="new_filename"
                )
                save_path = BASE_DIR / new_filename
            else:
                save_path = st.session_state.current_file
            
            if st.button("💾 Save File", type="primary", disabled=not st.session_state.modified):
                try:
                    save_corridor_json(save_path, st.session_state.corridor_data)
                    st.session_state.modified = False
                    st.success(f"Saved to {save_path.name}")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
    
    # Main area - Map visualization
    if st.session_state.corridor_data:
        data = st.session_state.corridor_data
        
        st.subheader("🗺️ Map Preview")
        
        # Map controls
        col1, col2 = st.columns(2)
        with col1:
            show_stations = st.checkbox("Show station markers", value=True)
        with col2:
            # System filter
            all_systems = list(data.keys())
            visible_systems = st.multiselect(
                "Visible systems:",
                options=all_systems,
                default=all_systems,
                key="visible_systems"
            )
        
        # Create map
        # Calculate center from first path
        center_lat, center_lon = -22.0, -43.5  # Default to Rio area
        if data and visible_systems:
            first_system = visible_systems[0]
            if data[first_system] and data[first_system][0].get('coords_ll'):
                first_coord = data[first_system][0]['coords_ll'][0]
                center_lat, center_lon = first_coord[0], first_coord[1]
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap',
            max_bounds=True,
            bounds=_BR_BOUNDS,
        )
        
        # Render all paths
        for system_name in visible_systems:
            if system_name not in data:
                continue
            
            paths = data[system_name]
            for path in paths:
                path_color = path.get('route_color', '#1f77b4')
                path_label = f"{system_name}: {path.get('path_name', 'Unnamed')}"
                render_path_on_map(m, path, path_color, path_label, show_stations)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Display map
        st_folium(m, width=1400, height=700, returned_objects=[])
        
        # Statistics
        st.divider()
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        
        total_paths = sum(len(paths) for paths in data.values())
        total_km = sum(
            path.get('total_km', 0)
            for paths in data.values()
            for path in paths
        )
        
        with col1:
            st.metric("Systems", len(data))
        with col2:
            st.metric("Total Paths", total_paths)
        with col3:
            st.metric("Total Distance", f"{total_km:.1f} km")
    
    else:
        st.info("👈 Select and load a corridor JSON file from the sidebar to begin editing")


if __name__ == "__main__":
    main()
