# Unattended bisect: cd52717 (verified good) .. branch tip (verified bad).
# git bisect run does the looping; the discipline lives in bisect-step.ps1 (skip on unbuildable, skip when the rotor
# never engaged, good only on 3/3 clean).
$ErrorActionPreference = 'Continue'
Set-Location 'G:\AI\maya-eval\project-maya'

$stepLog = 'G:\AI\maya-eval\project-maya\bisect-step.log'
$runLog  = 'G:\AI\maya-eval\project-maya\bisect-run.log'
$bisLog  = 'G:\AI\maya-eval\project-maya\docs\bisect-log.txt'
Remove-Item $stepLog, $runLog, $bisLog -ErrorAction SilentlyContinue

& git bisect reset 2>&1 | Out-Null
$bad = (& git rev-parse --short HEAD).Trim()
& git bisect start 2>&1 | Out-Null
& git bisect bad $bad 2>&1 | Out-Null
& git bisect good cd52717 2>&1 | Out-Null
"[$(Get-Date -Format o)] range: $bad (bad) .. cd52717 (good)" | Add-Content $stepLog

# the loop itself
& cmd /c 'git bisect run cmd /c "G:\AI\maya-eval\project-maya\bisect-step.bat"' *> $runLog
$rc = $LASTEXITCODE

& git bisect log 2>&1 | Set-Content $bisLog -Encoding utf8
"exit=$rc" | Add-Content $bisLog
$firstBad = (& cmd /c 'git bisect log' 2>&1) | Select-String -Pattern 'first bad commit' | Select-Object -Last 1

& git bisect reset 2>&1 | Out-Null

Write-Output ("  bisect finished, exit=" + $rc)
if ($firstBad) { Write-Output ("  " + $firstBad.Line.Trim()) } else { Write-Output "  (no 'first bad commit' in the log)" }
Write-Output "  --- step log ---"
Get-Content $stepLog -ErrorAction SilentlyContinue | Select-Object -Last 40 | ForEach-Object { "    " + $_ }
Write-Output "  --- run log (tail) ---"
Get-Content $runLog -ErrorAction SilentlyContinue | Select-Object -Last 12 | ForEach-Object { "    " + $_.Trim() }
Write-Output ("  tree restored to: " + (& git rev-parse --abbrev-ref HEAD) + " @ " + (& git rev-parse --short HEAD))
