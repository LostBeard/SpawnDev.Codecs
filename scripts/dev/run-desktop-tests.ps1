# Runs desktop tests of THIS worktree one process each (exactly how PlaywrightMultiTest's console lane does),
# at BelowNormal priority, printing one line per test. Build first with scripts/dev/build-console.ps1.
#   pwsh scripts/dev/run-desktop-tests.ps1 -Pattern 'PngDecoderGpu'
# -Class selects the backend class: CpuCodecsTests (default, ILGPU CPU backend), CudaCodecsTests, OpenCLCodecsTests,
# VelocityCodecsTests (the vectorized CPU backend applications get by default).
# GPU classes share the machine's one GPU with other agents: do NOT run Cuda/OpenCL unless the coordinator
# (Geordi) has said the GPU is yours.
# -OutDir keeps each test's full console output as <OutDir>\<Class.Method>.log (diagnostics such as VP9_STAGE_TIMES=1
# print there); without it the output is deleted once the result line is read.
# -Batch runs the tests in ONE process (DemoConsole --batch) instead of one each: starting a process costs ~0.75 s per
# test, ~90 min of a full desktop sweep (2026-11-16 measurement). A test that hangs past -TimeoutSec or crashes the
# process is charged to that test (TIMEOUT / NO-RESULT) and the rest continue in a fresh process. Same output lines.
param(
    [Parameter(Mandatory)][string]$Pattern,          # regex over "ClassName.Method"
    [string]$Class = 'CpuCodecsTests',
    [int]$TimeoutSec = 600,
    [string]$OutDir = '',
    [switch]$Batch
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $root 'SpawnDev.Codecs.DemoConsole\bin\Release\net10.0\SpawnDev.Codecs.DemoConsole.exe'
if (-not (Test-Path $exe)) { throw "Not built: $exe (run scripts/dev/build-console.ps1)" }
$list = @(& $exe | Where-Object { $_ -like "$Class.*" -and $_ -match $Pattern })
$pass = 0; $fail = 0; $skip = 0
if ($OutDir) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
# keeps (with -OutDir) or deletes a test's output file
function Complete-Output([string]$file, [string]$test) {
    if ($OutDir) { Move-Item $file (Join-Path $OutDir "$test.log") -Force } else { Remove-Item $file -ErrorAction SilentlyContinue }
}
# one result line from a "TEST: {...}" json (ResultText, Error) and the elapsed ms
function Write-Result($j, [string]$t, [long]$ms) {
    if ($j.ResultText -eq 'Success') { $script:pass++; "PASS ${ms}ms $t" }
    elseif (-not $j.Error) { $script:skip++; "SKIP $($j.ResultText) :: $t" }   # UnsupportedTestException = skipped on this backend
    else { $script:fail++; "FAIL $($j.ResultText) $t`n     $($j.Error)" }
}

if (-not $Batch) {
    foreach ($t in $list) {
        $out = [IO.Path]::GetTempFileName()
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process $exe -ArgumentList $t -RedirectStandardOutput $out -NoNewWindow -PassThru
        try { $p.PriorityClass = 'BelowNormal' } catch {}
        if (-not $p.WaitForExit($TimeoutSec * 1000)) { $p.Kill($true); "TIMEOUT  $t"; $fail++; Complete-Output $out $t; continue }
        $sw.Stop()
        $line = (Get-Content $out | Where-Object { $_ -like 'TEST: *' } | Select-Object -Last 1)
        Complete-Output $out $t
        if (-not $line) { "NO-RESULT $t (exit $($p.ExitCode))"; $fail++; continue }
        Write-Result ($line.Substring(6) | ConvertFrom-Json) $t ([long]$sw.Elapsed.TotalMilliseconds)
    }
}
else {
    $remaining = [System.Collections.Generic.List[string]]::new([string[]]$list)
    while ($remaining.Count -gt 0) {
        $listFile = [IO.Path]::GetTempFileName()
        Set-Content -Path $listFile -Value $remaining
        $out = [IO.Path]::GetTempFileName()
        $p = Start-Process $exe -ArgumentList '--batch', "`"$listFile`"" -RedirectStandardOutput $out -NoNewWindow -PassThru
        try { $p.PriorityClass = 'BelowNormal' } catch {}
        $fs = [IO.FileStream]::new($out, 'Open', 'Read', 'ReadWrite, Delete')
        $reader = [IO.StreamReader]::new($fs)
        $pending = ''                 # a line not yet ended by '\n'
        $current = $null              # the test between its BEGIN and TEST lines
        $began = [Diagnostics.Stopwatch]::new()
        $log = [Text.StringBuilder]::new()
        $progress = $false            # this process finished at least one test
        $exited = $false
        while ($true) {
            $exited = $p.WaitForExit(200)
            $pending += $reader.ReadToEnd()
            $parts = $pending -split "`n"
            $pending = $parts[-1]     # the last piece has no '\n' yet (empty when the chunk ended on one)
            # (no complete line: $parts[0..-1] would count DOWN and yield two items in PowerShell)
            $complete = if ($parts.Count -gt 1) { $parts[0..($parts.Count - 2)] } else { @() }
            foreach ($raw in $complete) {
                $line = $raw.TrimEnd("`r")
                if ($line.StartsWith('BEGIN: ')) {
                    $current = $line.Substring(7); $began.Restart(); [void]$log.Clear()
                }
                elseif ($line.StartsWith('TEST: ') -and $current) {
                    $j = $line.Substring(6) | ConvertFrom-Json
                    if ($OutDir) { [IO.File]::WriteAllText((Join-Path $OutDir "$current.log"), $log.ToString()) }
                    Write-Result $j $current ([long]$j.Ms)
                    [void]$remaining.Remove($current); $current = $null; $progress = $true
                }
                elseif ($line.StartsWith('ERROR: Test not found: ')) {
                    $t = $line.Substring(23); "NO-RESULT $t (not found)"; $fail++
                    [void]$remaining.Remove($t); $current = $null; $progress = $true
                }
                elseif ($current) { [void]$log.AppendLine($line) }
            }
            if ($exited) { break }
            if ($current -and $began.Elapsed.TotalSeconds -gt $TimeoutSec) {
                $p.Kill($true); [void]$p.WaitForExit(10000)
                "TIMEOUT  $current"; $fail++
                if ($OutDir) { [IO.File]::WriteAllText((Join-Path $OutDir "$current.log"), $log.ToString()) }
                [void]$remaining.Remove($current); $current = $null; $progress = $true
                break
            }
        }
        $reader.Dispose()
        Remove-Item $out, $listFile -ErrorAction SilentlyContinue
        if ($current) {
            # the process ended inside a test (crash): charge it, the rest run in a fresh process
            "NO-RESULT $current (exit $($p.ExitCode))"; $fail++
            if ($OutDir) { [IO.File]::WriteAllText((Join-Path $OutDir "$current.log"), $log.ToString()) }
            [void]$remaining.Remove($current); $progress = $true
        }
        if (-not $progress -and $remaining.Count -gt 0) {
            # the process ended before starting any test: charge the next one so the loop always advances
            $t = $remaining[0]; "NO-RESULT $t (batch process exit $($p.ExitCode) before the test began)"; $fail++
            [void]$remaining.RemoveAt(0)
        }
    }
}
"--- $pass passed, $fail failed, $skip skipped of $($list.Count)"
if ($fail -gt 0) { exit 1 }
