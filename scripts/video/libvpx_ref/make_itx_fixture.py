"""Build itx_ref.exe against libvpx's compiled objects, run it, gzip its output to the VP9 inverse transform fixture.

Usage: py -3 make_itx_fixture.py <libvpx source dir> <libvpx build dir>

The build dir is the one build_libvpx_ref.py filled (generated headers + every libvpx .obj). Every .obj there except
libvpx_ref.obj (it has a main) goes into a static library, so the linker takes only the objects itx_ref.c needs
(vp9_idct, inv_txfm, vp9_scan, ...); lib.exe's LNK4006 duplicate warnings (an older vp9_decodeframe.obj next to
patched_vp9_decodeframe.obj) are harmless, neither object is pulled. itx_ref.exe and the library land in <build dir>/itx_ref; the fixture is
SpawnDev.Codecs.Demo.Shared/TestData/Vp9Dsp/itx.gz (single extension: x.bin.gz would be a satellite resource).
"""
import glob, gzip, os, subprocess, sys

src, build = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
out_dir = os.path.join(build, "itx_ref")
os.makedirs(out_dir, exist_ok=True)

vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
assert os.path.exists(vcvars), vcvars
objs = sorted(p for p in glob.glob(os.path.join(build, "*.obj")) if os.path.basename(p).lower() != "libvpx_ref.obj")
assert objs, f"no .obj files in {build} (run build_libvpx_ref.py first)"

lib = os.path.join(out_dir, "libvpx_objs.lib")
exe = os.path.join(out_dir, "itx_ref.exe")
with open(os.path.join(out_dir, "lib.rsp"), "w") as f:
    f.write(f'/nologo /OUT:"{lib}"\n' + "\n".join(f'"{p}"' for p in objs) + "\n")
with open(os.path.join(out_dir, "build.bat"), "w") as f:
    f.write(f'@call "{vcvars}" >nul\n')
    f.write(f'lib @"{os.path.join(out_dir, "lib.rsp")}" || exit /b 1\n')
    # the flags libvpx's objects were compiled with (build_libvpx_ref.py)
    f.write(f'cl /nologo /O2 /W0 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /I"{build}" /I"{src}" '
            f'/Fo"{os.path.join(out_dir, "itx_ref.obj")}" /Fe"{exe}" "{os.path.join(here, "itx_ref.c")}" "{lib}" || exit /b 1\n')
if os.path.exists(exe):
    os.remove(exe)
b = subprocess.run(["cmd", "/c", os.path.join(out_dir, "build.bat")], capture_output=True, text=True)
print(b.stdout[-3000:], b.stderr[-3000:])
assert b.returncode == 0 and os.path.exists(exe), f"build failed ({b.returncode})"

raw = os.path.join(out_dir, "itx.raw")
r = subprocess.run([exe, raw], capture_output=True, text=True, check=True)
print(r.stdout.strip())
data = open(raw, "rb").read()
dst = os.path.join(repo, "SpawnDev.Codecs.Demo.Shared", "TestData", "Vp9Dsp", "itx.gz")
# mtime 0: the same cases give the same bytes
with open(dst, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as g:
    g.write(data)
print(f"{len(data)} bytes -> {dst} ({os.path.getsize(dst)} bytes)")
