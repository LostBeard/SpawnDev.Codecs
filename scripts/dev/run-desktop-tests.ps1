# Runs desktop tests of THIS worktree one process each (exactly how PlaywrightMultiTest's console lane does),
# at BelowNormal priority, printing one line per test. Build first with scripts/dev/build-console.ps1.
#   pwsh scripts/dev/run-desktop-tests.ps1 -Pattern 'PngDecoderGpu'
# -Class selects the backend class: CpuCodecsTests (default, ILGPU CPU backend), CudaCodecsTests, OpenCLCodecsTests.
# GPU classes share the machine's one GPU with other agents: do NOT run Cuda/OpenCL unless the coordinator
# (Geordi) has said the GPU is yours.
param(
    [Parameter(Mandatory)][string]$Pattern,          # regex over "ClassName.Method"
    [string]$Class = 'CpuCodecsTests',
    [int]$TimeoutSec = 600
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe = Join-Path $root 'SpawnDev.Codecs.DemoConsole\bin\Release\net10.0\SpawnDev.Codecs.DemoConsole.exe'
if (-not (Test-Path $exe)) { throw "Not built: $exe (run scripts/dev/build-console.ps1)" }
$list = & $exe | Where-Object { $_ -like "$Class.*" -and $_ -match $Pattern }
$pass = 0; $fail = 0; $skip = 0
foreach ($t in $list) {
    $out = [IO.Path]::GetTempFileName()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process $exe -ArgumentList $t -RedirectStandardOutput $out -NoNewWindow -PassThru
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { $p.Kill($true); "TIMEOUT  $t"; $fail++; continue }
    $sw.Stop()
    $line = (Get-Content $out | Where-Object { $_ -like 'TEST: *' } | Select-Object -Last 1)
    Remove-Item $out -ErrorAction SilentlyContinue
    if (-not $line) { "NO-RESULT $t (exit $($p.ExitCode))"; $fail++; continue }
    $j = $line.Substring(6) | ConvertFrom-Json
    if ($j.ResultText -eq 'Success') { $pass++; "PASS $([int]$sw.Elapsed.TotalMilliseconds)ms $t" }
    elseif (-not $j.Error) { $skip++; "SKIP $($j.ResultText) :: $t" }   # UnsupportedTestException = skipped on this backend
    else { $fail++; "FAIL $($j.ResultText) $t`n     $($j.Error)" }
}
"--- $pass passed, $fail failed, $skip skipped of $($list.Count)"
