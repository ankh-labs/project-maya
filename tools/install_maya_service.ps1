#Requires -Version 5
<#
    tools/install_maya_service.ps1 - installs or refreshes Maya's NSSM service (port 8080).

    Why this exists rather than "just run maya.py": a Windows service gets a minimal
    environment, and this engine needs two things a user shell provides for free.

      1. cublas64_13.dll.  The engine links cublas dynamically and the DLL lives in
         CUDA\<ver>\bin\x64, which is NOT on a LocalSystem service's PATH.  Without it the
         engine dies before serving, and nothing in the server's log says why.
      2. The user's profile dirs (calibration and caches under ~/.config/project-maya).

    Run it elevated, e.g. from the agent's non-elevated shell:
        ssh agips@localhost "powershell -NoProfile -ExecutionPolicy Bypass -File <this file>"

    It stops a hand-started instance first, so it is safe to run over a running server.
#>
[CmdletBinding()]
param(
    [string] $ServiceName = 'Maya-8080',
    [int]    $Port        = 8080,
    [string] $Repo        = 'G:\AI\maya-eval\project-maya',
    [string] $Config      = 'maya-v3q-q4-par4.json',
    # The server-side fairness knob (seconds an answer may hold the engine while others
    # wait).  The config's "env" block reaches only the ENGINE child, so this one belongs
    # to the service environment.
    [string] $FairSliceS  = '20',
    [string] $ProfileDir  = 'C:\Users\agips',
    [switch] $NoStart
)

$ErrorActionPreference = 'Stop'

$nssm = (Get-ChildItem 'C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_*\nssm-*\win64\nssm.exe' `
         -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if (-not $nssm) { throw 'nssm.exe not found (winget package NSSM.NSSM)' }

$py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py))      { throw "venv python missing: $py" }
if (-not (Test-Path $Repo))    { throw "repo missing: $Repo" }
$logs = Join-Path $Repo 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# ---------------------------------------------------------------- the CUDA bin holding cublas
$cudaBin = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\x64\cublas64_13.dll' `
           -ErrorAction SilentlyContinue |
           Sort-Object { [version]($_.Directory.Parent.Parent.Name -replace '^v', '') } |
           Select-Object -Last 1 | ForEach-Object { $_.DirectoryName }
if (-not $cudaBin) { throw 'cublas64_13.dll not found under the CUDA toolkit - the engine cannot start without it' }
"cuda bin   : $cudaBin"

# ---------------------------------------------------------------- 1. stop a hand-started instance
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'maya\.py|serve\\server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process maya -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

# ---------------------------------------------------------------- 2. refresh an earlier install
& sc.exe query $ServiceName *> $null
if ($LASTEXITCODE -eq 0) {
    & $nssm stop $ServiceName confirm *> $null
    Start-Sleep -Seconds 4
    & $nssm remove $ServiceName confirm *> $null
    Start-Sleep -Seconds 2
    "removed the previous $ServiceName"
}

# ---------------------------------------------------------------- 3. the service
& $nssm install $ServiceName $py "maya.py --config $Config --port $Port --host 0.0.0.0"
& $nssm set $ServiceName AppDirectory $Repo
& $nssm set $ServiceName DisplayName "Maya $Port - GLM-5.3-Flash (project-maya)"
& $nssm set $ServiceName Description "Maya's OpenAI-compatible server on port $Port ($Config)."
& $nssm set $ServiceName Start SERVICE_AUTO_START
& $nssm set $ServiceName AppStdout  (Join-Path $logs "maya-$Port.out.log")
& $nssm set $ServiceName AppStderr  (Join-Path $logs "maya-$Port.err.log")
& $nssm set $ServiceName AppRotateFiles 1
& $nssm set $ServiceName AppRotateOnline 1
& $nssm set $ServiceName AppRotateBytes 10485760
# maya.py treats Ctrl+C/SIGTERM/SIGBREAK as a graceful stop; give it room before a kill.
& $nssm set $ServiceName AppStopMethodConsole 25000
& $nssm set $ServiceName AppStopMethodWindow 25000
& $nssm set $ServiceName AppKillProcessTree 1
& $nssm set $ServiceName AppRestartDelay 5000
& $nssm set $ServiceName AppThrottle 15000

# ---------------------------------------------------------------- 4. environment
$path = @($cudaBin, 'C:\Windows\system32', 'C:\Windows', 'C:\Windows\System32\Wbem') -join ';'
$envPairs = @(
    "PATH=$path",
    "USERPROFILE=$ProfileDir",
    "APPDATA=$ProfileDir\AppData\Roaming",
    "LOCALAPPDATA=$ProfileDir\AppData\Local",
    "HOME=$ProfileDir",
    "TEMP=$ProfileDir\AppData\Local\Temp",
    "TMP=$ProfileDir\AppData\Local\Temp",
    "MAYA_FAIR_SLICE_S=$FairSliceS"
)
& $nssm set $ServiceName AppEnvironmentExtra @envPairs | Out-Null

# nssm joins the pairs; verify what actually landed and write it directly if it did not
$key = "HKLM:\SYSTEM\CurrentControlSet\Services\$ServiceName\Parameters"
$got = (Get-ItemProperty -Path $key -Name Environment -ErrorAction SilentlyContinue).Environment
if (-not $got -or @($got).Count -lt $envPairs.Count) {
    New-ItemProperty -Path $key -Name Environment -PropertyType MultiString -Value $envPairs -Force | Out-Null
    $got = (Get-ItemProperty -Path $key -Name Environment).Environment
}
"environment: " + @($got).Count + " entries"
@($got) | ForEach-Object { "   $_" }

# ---------------------------------------------------------------- 5. start and wait for the port
if (-not $NoStart) {
    & sc.exe start $ServiceName | Out-Null
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 5
        if (@(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count) {
            "listening on $Port after $(($i + 1) * 5)s"
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        "NOT listening after 300 s - check $(Join-Path $logs "maya-$Port.err.log")"
        exit 1
    }
    & sc.exe query $ServiceName | Select-String 'STATE' | ForEach-Object { "  " + $_.Line.Trim() }
    "engine: " + @(Get-Process maya -ErrorAction SilentlyContinue).Count + " process(es)"
}
