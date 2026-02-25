param(
    [string]$ExePath = ".\dist\MRS_Map.exe",
    [ValidateSet('error','warning','info','debug')]
    [string]$LogLevel = 'debug',
    [int]$TimeoutMinutes = 0
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$resolvedExe = Resolve-Path $ExePath -ErrorAction Stop
$exeFullPath = $resolvedExe.Path
$exeDir = Split-Path -Parent $exeFullPath

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $root ("logs\\run_" + $timestamp)
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$stdoutLog = Join-Path $runDir 'stdout.log'
$stderrLog = Join-Path $runDir 'stderr.log'
$metaLog = Join-Path $runDir 'meta.txt'
$combinedLog = Join-Path $runDir 'combined.log'
$zipPath = Join-Path $root ("logs\\mrs_debug_" + $timestamp + ".zip")

$env:STREAMLIT_LOG_LEVEL = $LogLevel
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'

$pyiTemp = Join-Path $runDir '_pyi_temp'
New-Item -ItemType Directory -Path $pyiTemp -Force | Out-Null
$env:PYINSTALLER_TEMP = $pyiTemp

$meta = @()
$meta += "Timestamp: $(Get-Date -Format o)"
$meta += "ExePath: $exeFullPath"
$meta += "WorkingDir: $exeDir"
$meta += "Host: $env:COMPUTERNAME"
$meta += "User: $env:USERNAME"
$meta += "OS: $([System.Environment]::OSVersion.VersionString)"
$meta += "PSVersion: $($PSVersionTable.PSVersion)"
$meta += "StreamlitLogLevel: $LogLevel"
$meta += "TimeoutMinutes: $TimeoutMinutes"
$meta += ""
$meta += "Executable file info:"
$meta += (Get-Item $exeFullPath | Format-List Name,Length,CreationTime,LastWriteTime | Out-String)
$meta += "File hash (SHA256):"
$meta += (Get-FileHash -Algorithm SHA256 -Path $exeFullPath | Format-List | Out-String)
Set-Content -Path $metaLog -Value ($meta -join [Environment]::NewLine)

Write-Host "Starting executable with logging..."
Write-Host "- Exe: $exeFullPath"
Write-Host "- Logs: $runDir"
Write-Host "- Streamlit log level: $LogLevel"
Write-Host "- Expected URL: http://localhost:8501"

$proc = Start-Process -FilePath $exeFullPath `
    -WorkingDirectory $exeDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Start-Sleep -Seconds 8

$portChecks = @()
foreach ($port in @(8501, 3000)) {
    $ok = $false
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:" + $port + "/_stcore/health") -TimeoutSec 3
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
            $ok = $true
        }
    } catch {
        $ok = $false
    }
    $status = if ($ok) { 'OPEN' } else { 'CLOSED' }
    $portChecks += ("Port " + $port + ": " + $status)
}

Add-Content -Path $metaLog -Value ""
Add-Content -Path $metaLog -Value "Port checks after startup:"
foreach ($line in $portChecks) {
    Add-Content -Path $metaLog -Value $line
}

if ($TimeoutMinutes -gt 0) {
    $timeoutMs = $TimeoutMinutes * 60 * 1000
    $finished = $proc.WaitForExit($timeoutMs)
    if (-not $finished) {
        Write-Warning "Timeout reached ($TimeoutMinutes min). Stopping process..."
        Stop-Process -Id $proc.Id -Force
    }
} else {
    Write-Host "App is running. Reproduce the issue, then close the app window/terminal to finish logging."
    Wait-Process -Id $proc.Id
}

$stdoutText = if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Raw } else { '' }
$stderrText = if (Test-Path $stderrLog) { Get-Content $stderrLog -Raw } else { '' }

$combined = @()
$combined += "==== META ===="
$combined += (Get-Content $metaLog -Raw)
$combined += ""
$combined += "==== STDOUT ===="
$combined += $stdoutText
$combined += ""
$combined += "==== STDERR ===="
$combined += $stderrText
Set-Content -Path $combinedLog -Value ($combined -join [Environment]::NewLine)

$possibleAppFiles = @(
    (Join-Path $exeDir 'committed_highlights.json'),
    (Join-Path $exeDir 'exported_highlights.json')
)
foreach ($f in $possibleAppFiles) {
    if (Test-Path $f) {
        Copy-Item -Path $f -Destination $runDir -Force
    }
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $runDir '*') -DestinationPath $zipPath

Write-Host ""
Write-Host "Debug bundle ready:"
Write-Host $zipPath
Write-Host ""
Write-Host "Share this .zip file for troubleshooting."
