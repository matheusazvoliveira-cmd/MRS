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

$Args = @(
    '--noconfirm'
    '--clean'
    '--name', 'MRS_Map'
    '--onefile'
    '--collect-all', 'streamlit'
    '--collect-all', 'streamlit_folium'
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

Write-Host "Build complete: dist\MRS_Map.exe"
