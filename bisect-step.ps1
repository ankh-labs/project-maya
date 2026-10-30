# One bisect step: build, run the arm, decide good/bad/skip.
#   0   = 3/3 clean
#   1   = any failure
#   125 = SKIP (unbuildable, or the rotor never engaged - a boot without rotor lines is not evidence)
$ErrorActionPreference = 'Continue'
Set-Location 'G:\AI\maya-eval\project-maya'

$sha = (& git rev-parse --short HEAD 2>$null)
$log = "G:\AI\maya-eval\project-maya\bisect-step.log"
"[$(Get-Date -Format o)] testing $sha" | Add-Content $log

# ---- 1. build; a commit that cannot compile is SKIPPED, not marked bad
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'maya\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
Get-Process maya -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 4

& cmd /c "G:\AI\maya-eval\project-maya\build-maya.bat" *> "$env:TEMP\bisect-build.log"
if ($LASTEXITCODE -ne 0) {
    "    $sha BUILD FAILED -> skip" | Add-Content $log
    exit 125
}
"    $sha built" | Add-Content $log

# ---- 2. the arm: config-only, n=2, 512 tokens, three boots
Remove-Item Env:\STRATA_GLM_STREAM_FETCH -ErrorAction SilentlyContinue
Remove-Item Env:\STRATA_GLM_TOK_ID_LOG -ErrorAction SilentlyContinue
Remove-Item Env:\STRATA_GLM_ROW_DRAIN_SEC -ErrorAction SilentlyContinue
Remove-Item Env:\STRATA_GLM_ROUTE_FAST -ErrorAction SilentlyContinue

$LOGP = 'G:\AI\maya-eval\project-maya\maya-v3q-q4.log'
$P = 'Explain ZFS RAIDZ1 parity in about thirty words.'
$Q = 'In two sentences, what does a checksum catch that a mirror does not?'
function Body([string]$p) {
    @{ model = 'glm'; messages = @(@{ role = 'user'; content = $p }); temperature = 0; max_tokens = 512; stream = $false } |
        ConvertTo-Json -Depth 6
}
function Boot {
    Remove-Item $LOGP -ErrorAction SilentlyContinue
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', '"G:\AI\maya-eval\project-maya\run-maya-cfgonly.bat"' `
        -WorkingDirectory 'G:\AI\maya-eval\project-maya' -WindowStyle Hidden | Out-Null
    $deadline = (Get-Date).AddMinutes(20)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try { $null = Invoke-RestMethod 'http://127.0.0.1:8080/metrics' -TimeoutSec 4; return $true } catch {}
    }
    return $false
}
function Pair([int]$budget) {
    $a = Start-ThreadJob -ScriptBlock {
        param($b, $t)
        try { $r = Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $b -TimeoutSec $t
              [pscustomobject]@{ t = $r.usage.completion_tokens; f = $r.choices[0].finish_reason } } catch { [pscustomobject]@{ t = -1; f = 'error' } }
    } -ArgumentList (Body $P), $budget
    $b2 = Start-ThreadJob -ScriptBlock {
        param($b, $t)
        try { $r = Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $b -TimeoutSec $t
              [pscustomobject]@{ t = $r.usage.completion_tokens; f = $r.choices[0].finish_reason } } catch { [pscustomobject]@{ t = -1; f = 'error' } }
    } -ArgumentList (Body $Q), $budget
    $null = Wait-Job $a, $b2 -Timeout ($budget + 60)
    $ra = Receive-Job $a; $rb = Receive-Job $b2
    Remove-Job $a, $b2 -Force -ErrorAction SilentlyContinue
    return @($ra, $rb)
}

$clean = 0
$rotorTotal = 0
for ($i = 1; $i -le 3; $i++) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'maya\.py' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    Get-Process maya -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 6
    if (-not (Boot)) { "    $sha boot $i did not come up" | Add-Content $log; continue }
    $r = Pair 900
    $rotor = @(Select-String -Path $LOGP -Pattern 'rotor entered' -ErrorAction SilentlyContinue).Count
    $rotorTotal += $rotor
    $ok = ($r[0].t -gt 0) -and ($r[1].t -gt 0)
    if ($ok) { $clean++ }
    "    $sha boot $i : $($r[0].t)+$($r[1].t) tok, rotor $rotor" | Add-Content $log
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'maya\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
Get-Process maya -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# ---- 3. the rotor guard: no rotation engaged anywhere = not evidence
if ($rotorTotal -eq 0) {
    "    $sha ROTOR NEVER ENGAGED -> skip (a clean result without it would corrupt the bisect)" | Add-Content $log
    exit 125
}

if ($clean -eq 3) { "    $sha GOOD (3/3 clean, rotor $rotorTotal)" | Add-Content $log; exit 0 }
"    $sha BAD ($clean/3 clean, rotor $rotorTotal)" | Add-Content $log
exit 1
