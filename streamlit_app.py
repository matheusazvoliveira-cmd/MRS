import threading
import time
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# Paths (relative to this notebook folder)
STATIONS_SHP = "Estacoes/Estacoes.shp"
RAILS_SHP = "LInhas/Linhas_BR.shp"


@st.cache_data
def load_stations(path=STATIONS_SHP):
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        st.error(f"Failed to read stations shapefile: {e}")
        return None

    # Prefer three-letter code column if present
    code_col = None
    for candidate in ["CodigoTres", "CODIGO_TRES", "code3", "code"]:
        if candidate in gdf.columns:
            code_col = candidate
            break

    if code_col is None:
        gdf["station_code"] = ""
    else:
        gdf["station_code"] = gdf[code_col].astype(str)

    # Build a human label
    name_col = next((c for c in ["NomeEstaca", "name", "nome"] if c in gdf.columns), None)
    if name_col:
        gdf["station_label"] = gdf[name_col].astype(str)
    else:
        gdf["station_label"] = gdf.index.astype(str)

    gdf["label_with_code"] = gdf.apply(lambda r: f"{r['station_label']} [{r['station_code']}]" if r['station_code'] else r['station_label'], axis=1)
    return gdf


def compute_preview_placeholder(origin, destination, vias, progress_callback=None):
    """Placeholder for long-running routing computation.

    Replace this with the notebook's `compute_preview` logic ported into functions.
    The `progress_callback` (callable(percent:int)) can be used for progress updates.
    """
    steps = [5, 30, 60, 85, 100]
    for p in steps:
        time.sleep(0.6)
        if progress_callback:
            progress_callback(p)

    # Fake result
    return {
        "route": [origin, *vias, destination],
        "message": f"Preview completed: {origin} → {', '.join(vias) if vias else 'no vias'} → {destination}",
    }


def main():
    st.set_page_config(page_title="MRS Corridor — Streamlit Prototype", layout="wide")
    st.title("MRS Corridor — Streamlit Prototype")

    gdf = load_stations()
    if gdf is None:
        return

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.header("Route builder")
        origin = st.selectbox("Origin", options=gdf["label_with_code"].tolist())
        destination = st.selectbox("Destination", options=gdf["label_with_code"].tolist(), index=min(1, len(gdf)-1))
        vias = st.multiselect("VIA stations (optional)", options=gdf["label_with_code"].tolist())

        st.markdown("**Selected codes:**")
        st.write(origin)
        st.write("→")
        for v in vias:
            st.write(v)
        st.write("→")
        st.write(destination)

        run_btn = st.button("Run preview")
        progress_bar = st.progress(0)
        status_txt = st.empty()

    with right_col:
        # Center map on stations centroid
        centroid = gdf.geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=6)

        # Add station markers (lightweight)
        for _, r in gdf.iterrows():
            folium.CircleMarker(location=[r.geometry.y, r.geometry.x], radius=2, color="#0077cc", fill=True,
                                popup=r["label_with_code"]).add_to(m)

        map_result = st_folium(m, width=900, height=700)

    if run_btn:
        # Run compute in a thread so Streamlit UI stays responsive for progress updates
        result_box = st.empty()

        def progress_cb(p):
            progress_bar.progress(int(p))
            status_txt.text(f"Progress: {p}%")

        def worker():
            origin_code = origin
            dest_code = destination
            res = compute_preview_placeholder(origin_code, dest_code, vias, progress_callback=progress_cb)
            result_box.write(res["message"])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


if __name__ == "__main__":
    main()
