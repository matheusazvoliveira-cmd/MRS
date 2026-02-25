$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PyInstallerCmd = 'pyinstaller'
if (Get-Command pyinstaller -ErrorAction SilentlyContinue) {
    $PyInstallerCmd = 'pyinstaller'
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PyInstallerCmd = 'python -m PyInstaller'
} else {
    throw 'Python/PyInstaller not found. Install Python and run: pip install pyinstaller'
}

# Preflight: verify runtime imports used by streamlit_app_full.py
$PythonCmd = 'python'
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = 'python'
}

$importCheck = @"
import importlib
mods = [
    'streamlit', 'streamlit_folium', 'geopandas', 'shapely', 'pyproj',
    'fiona', 'pandas', 'networkx', 'folium', 'numpy', 'scipy'
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit('Missing packages: ' + ', '.join(missing))
print('Dependency check OK')
"@

& $PythonCmd -c $importCheck
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency preflight failed. Install missing packages before building.'
}

$Args = @(
    '--noconfirm'
    '--clean'
    '--name', 'MRS_Map'
    '--onefile'
    '--collect-all', 'streamlit'
    '--collect-all', 'streamlit_folium'
    '--hidden-import', 'streamlit_folium'
    '--add-data', 'streamlit_app_full.py;.'
    '--add-data', 'Estacoes;Estacoes'
    '--add-data', 'LInhas;LInhas'
    'launcher.py'
)

if ($PyInstallerCmd -eq 'pyinstaller') {
    & pyinstaller @Args
} else {
    & python -m PyInstaller @Args
}

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "Build complete: dist\MRS_Map.exe"
