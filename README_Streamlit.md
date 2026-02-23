# Streamlit App — MRS Corridor Builder

Full-featured Streamlit port of the notebook routing app with all utilities.

## Versions

### `streamlit_app_full.py` (Recommended)
Complete port of the notebook with:
- ✅ Full graph building (node noding, snapping, corridor constraint, topology repair)
- ✅ Station dropdown with 3-letter codes
- ✅ Route computation (shortest path for each segment)
- ✅ Interactive Folium map with rail base layer and station markers
- ✅ Progress feedback and route info display
- ✅ Rail code breakdown by segment

### `streamlit_app.py` (Scaffold)
Lightweight scaffold (basic UI + placeholder compute logic) for quick prototyping.

## Quick Start

1. Create and activate a Python environment (Python 3.9+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the full app:

```powershell
streamlit run streamlit_app_full.py
```

3. Open browser to `http://localhost:8501`

## Usage

1. **Select Route:** Choose Station A, Station B, and optional VIA stations from the left sidebar.
2. **Configure:** Pick routing mode (Strict MRS, Prefer MRS, Allow all), toggle noding/snapping/corridor constraint.
3. **Compute:** Click **Compute Preview** to run the routing algorithm.
4. **View:** Route displays on the map with distance label and segment breakdown.

## File Paths

Update these in the app code if needed:

```python
RAILS_SHP_PATH = Path(r"...\Estacoes Code\LInhas\Linhas_BR.shp")
STATIONS_SHP_PATH = Path(r"...\Estacoes Code\Estacoes\Estacoes.shp")
```

## Notes

- Shapefiles must use EPSG:4674 or the code will auto-set it.
- First run loads data into cache; subsequent runs are faster.
- For production, consider background workers (Celery, RQ) for heavy computations and WebSocket-based progress updates.
