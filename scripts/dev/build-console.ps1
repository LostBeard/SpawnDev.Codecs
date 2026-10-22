# Builds the desktop test runner (SpawnDev.Codecs.DemoConsole + everything it references) at BelowNormal
# priority, single-threaded, so it does not disturb other agents' GPU/CPU runs on the shared PC.
# Usage: pwsh scripts/dev/build-console.ps1            (from any worktree; builds THAT worktree)
param([string]$Log = "")
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$proj = Join-Path $root 'SpawnDev.Codecs.DemoConsole\SpawnDev.Codecs.DemoConsole.csproj'
if (-not $Log) { $Log = Join-Path ([IO.Path]::GetTempPath()) ("codecs-build-" + [Guid]::NewGuid().ToString('N') + ".log") }
$p = Start-Process dotnet -ArgumentList 'build', $proj, '-c', 'Release', '-m:1', '-nologo', '-v:q', '-p:UseSharedCompilation=false' `
    -RedirectStandardOutput $Log -NoNewWindow -PassThru
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.WaitForExit()
"build exit $($p.ExitCode)   (log: $Log)"
Select-String -Path $Log -Pattern ': error' | Select-Object -First 25 | ForEach-Object { $_.Line }
Select-String -Path $Log -Pattern 'Warning\(s\)|Error\(s\)' | ForEach-Object { $_.Line }
exit $p.ExitCode
