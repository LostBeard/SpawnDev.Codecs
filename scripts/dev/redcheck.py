"""Red-check a desktop test: apply each mutation to a source file, rebuild, run the test, expect it to FAIL; restore the
file byte for byte (and touch it, so MSBuild rebuilds) after every mutation, then rebuild and expect a pass.

Usage: py -3 scripts/dev/redcheck.py <mutations.json>
mutations.json: {"test": "CpuCodecsTests.Name", "mutations": [{"name": "...", "file": "repo-relative path",
                 "old": "exact text (must occur once)", "new": "replacement"}, ...]}
Exit code 0 only when every mutation failed the test and the restored source passes it.
"""
import json, os, re, subprocess, sys

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = json.load(open(sys.argv[1], encoding="utf-8"))
exe = os.path.join(repo, "SpawnDev.Codecs.DemoConsole", "bin", "Release", "net10.0", "SpawnDev.Codecs.DemoConsole.exe")
build = ["pwsh", "-NoProfile", "-File", os.path.join(repo, "scripts", "dev", "build-console.ps1")]


def run():
    b = subprocess.run(build, capture_output=True, text=True)
    if b.returncode != 0:
        return "BUILD-FAILED", b.stdout[-300:]
    r = subprocess.run([exe, spec["test"]], capture_output=True, text=True, timeout=3600)
    res = re.search(r'"ResultText":"([^"]*)"', r.stdout)
    err = re.search(r'"Error":"([^"]{0,300})', r.stdout)
    return (res.group(1) if res else "NO-RESULT"), (err.group(1) if err else "")


originals = {}
ok = True
try:
    for m in spec["mutations"]:
        path = os.path.join(repo, m["file"])
        if path not in originals:
            originals[path] = open(path, "rb").read()
        text = originals[path].decode("utf-8")
        assert text.count(m["old"]) == 1, f"mutation '{m['name']}': old text occurs {text.count(m['old'])} times"
        open(path, "wb").write(text.replace(m["old"], m["new"]).encode("utf-8"))
        result, err = run()
        failed = result not in ("Success",) and result != "BUILD-FAILED"
        ok &= failed
        print(f"{'RED ' if failed else 'MISS'} {m['name']}: {result} :: {err[:200]}", flush=True)
        open(path, "wb").write(originals[path])
        os.utime(path, None)
finally:
    for path, data in originals.items():
        open(path, "wb").write(data)
        os.utime(path, None)
for path, data in originals.items():
    assert open(path, "rb").read() == data, f"{path} not restored"
result, err = run()
print(f"restored: {result} {err[:200]}")
ok &= result == "Success"
sys.exit(0 if ok else 1)
