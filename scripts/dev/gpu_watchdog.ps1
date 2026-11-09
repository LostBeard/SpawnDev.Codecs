# Watchdog for browser test runs: kills the GPU process of the PLAYWRIGHT-launched Chrome (never TJ's own browser) when
# its working set passes -LimitGB, so a giant shader compile (Tint) cannot starve the PC. Logs the peak it saw.
# Playwright's browser process is the chrome.exe whose command line has --remote-debugging-pipe; its GPU process is the
# child with --type=gpu-process (the child's own command line carries none of Playwright's flags).
param([double]$LimitGB = 3, [int]$IntervalMs = 1000, [string]$Log = (Join-Path $env:TEMP 'gpu_watchdog.log'))

$limit = [int64]($LimitGB * 1GB)
$peak = 0
$lastReport = Get-Date
"$(Get-Date -Format o) watchdog start, limit $LimitGB GB" | Out-File -FilePath $Log -Append
while ($true) {
    $all = Get-CimInstance Win32_Process -Filter "Name = 'chrome.exe'"
    $pwBrowsers = @($all | Where-Object { $_.CommandLine -match '--remote-debugging-pipe' } | ForEach-Object { $_.ProcessId })
    $gpu = $all | Where-Object { $_.CommandLine -match '--type=gpu-process' -and $pwBrowsers -contains $_.ParentProcessId }
    foreach ($p in $gpu) {
        $ws = (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue).WorkingSet64
        if ($ws -gt $peak) { $peak = $ws }
        if ($ws -gt $limit) {
            "$(Get-Date -Format o) KILL playwright gpu-process $($p.ProcessId) at $([math]::Round($ws / 1GB, 2)) GB" | Out-File -FilePath $Log -Append
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    if (((Get-Date) - $lastReport).TotalSeconds -ge 30) {
        "$(Get-Date -Format o) playwright browsers $($pwBrowsers.Count), gpu peak $([math]::Round($peak / 1GB, 2)) GB" | Out-File -FilePath $Log -Append
        $lastReport = Get-Date
    }
    Start-Sleep -Milliseconds $IntervalMs
}
