@echo off
@REM The release sweep: every class on every lane. Desktop classes in batch mode (one process per class, see
@REM run-desktop-tests.ps1 -Batch), then the browser classes through PlaywrightMultiTest scoped with PMT_FILTER (dotnet
@REM test --filter with several `|` terms selects every case, the desktop classes included).
@REM   scripts\dev\run-release-sweep.cmd <outdir>
@REM Holds the GPU, the CPU and the Codecs PMT ports for hours: mark the board and tell the crew first. Launch it outside
@REM the agent shell (WMI Win32_Process Create, hidden window) so it outlives the session's job object.
@REM The oracle tests need Pillow 13 / libwebp 1.6.0: CODECS_PILLOW_PYTHON defaults to _oracle-venv in the folder that
@REM holds this repo (a venv outside every session scratchpad: py -3.13 -m venv + pip install pillow==13.0.0 numpy soundfile).
setlocal
if "%~1" == "" ( echo usage: run-release-sweep.cmd ^<outdir^> & exit /b 2 )
set OUT=%~f1
if not exist "%OUT%" mkdir "%OUT%"
cd /d "%~dp0..\.."
if "%CODECS_PILLOW_PYTHON%" == "" set CODECS_PILLOW_PYTHON=%CD%\..\_oracle-venv\Scripts\python.exe
set SPAWNDEV_IMG_PYTHON=%CODECS_PILLOW_PYTHON%

call :stamp START
pwsh -NoProfile -File scripts\dev\build-console.ps1 > "%OUT%\build.log" 2>&1
if errorlevel 1 ( call :stamp BUILD_FAILED & exit /b 1 )
call :stamp BUILD_OK

for %%C in (CudaCodecsTests OpenCLCodecsTests VelocityCodecsTests CpuCodecsTests) do (
    call :stamp %%C start
    pwsh -NoProfile -File scripts\dev\run-desktop-tests.ps1 -Class %%C -Pattern "." -TimeoutSec 3600 -OutDir "%OUT%\out_%%C" -Batch > "%OUT%\%%C.log" 2>&1
    call :stamp %%C end
)

call :stamp PMT_START
set PMT_FILTER=WebGPUCodecsTests.,WasmCodecsTests.,WebGLCodecsTests.
dotnet test PlaywrightMultiTest\PlaywrightMultiTest.csproj --logger "trx;LogFileName=release-sweep.trx" > "%OUT%\pmt.log" 2> "%OUT%\pmt.err.log"
call :stamp PMT_END
call :stamp SWEEP_DONE
exit /b 0

:stamp
echo %* %DATE% %TIME% >> "%OUT%\status.log"
exit /b 0
