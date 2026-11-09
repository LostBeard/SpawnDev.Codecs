"""Build intra_ref.exe against the libvpx objects build_libvpx_ref.py compiled, run it, and write the fixture
SpawnDev.Codecs.Demo.Shared/TestData/Vp9Dsp/intra.gz (libvpx v1.17.0 vp9_predict_intra_block on every mode x transform
size x plane; format in intra_ref.c).

Usage: py -3 make_intra_fixture.py <libvpx source dir> <libvpx build dir> [work dir]

The build dir holds the .obj files of build_libvpx_ref.py (generic C, 8-bit). They are archived into a static library
(every .obj but libvpx_ref.obj, which has main) so the linker takes only the members intra_ref.c needs: the build dir
can also hold a stale vp9_decodeframe.obj next to patched_vp9_decodeframe.obj (which needs libvpx_ref.c's hook), and
neither is needed here.
"""
import glob, gzip, os, subprocess, sys

src, build = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(build, "intra_ref")
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
os.makedirs(work, exist_ok=True)

objs = sorted(p for p in glob.glob(os.path.join(build, "*.obj")) if os.path.basename(p) != "libvpx_ref.obj")
assert any(os.path.basename(p) == "vp9_reconintra.obj" for p in objs), "vp9_reconintra.obj missing: run build_libvpx_ref.py"
assert any(os.path.basename(p) == "intrapred.obj" for p in objs), "intrapred.obj missing: run build_libvpx_ref.py"
vcvars = sorted(glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"))[-1]
lib = os.path.join(work, "libvpx_objs.lib")
exe = os.path.join(work, "intra_ref.exe")
lib_rsp = os.path.join(work, "lib.rsp")
with open(lib_rsp, "w") as f:
    f.write(f'/nologo /OUT:"{lib}"\n')
    for p in objs:
        f.write(f'"{p}"\n')
bat = os.path.join(work, "build_intra_ref.bat")
with open(bat, "w") as f:
    f.write(f'@call "{vcvars}" >nul\r\n')
    f.write(f'lib @"{lib_rsp}" || exit /b 1\r\n')
    f.write(f'cl /nologo /O2 /W3 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /I"{build}" /I"{src}" /Fo"{work}\\\\" '
            f'/Fe"{exe}" "{os.path.join(here, "intra_ref.c")}" "{lib}" || exit /b 1\r\n')
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-4000:], r.stderr[-2000:])
    sys.exit("build FAILED")
out = os.path.join(work, "intra.bin")
r = subprocess.run([exe, out], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
if r.returncode != 0:
    sys.exit(f"intra_ref FAILED ({r.returncode})")
data = open(out, "rb").read()
fixture = os.path.join(repo, "SpawnDev.Codecs.Demo.Shared", "TestData", "Vp9Dsp", "intra.gz")
with open(fixture, "wb") as f:
    with gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=9, mtime=0) as g:
        g.write(data)
print(f"{len(data)} bytes -> {fixture} ({os.path.getsize(fixture)} bytes)")
