"""Build the DaTC Studio release for the current platform (Windows or Linux).

    python packaging/build_release.py [--work DIR] [--out DIR]

Runs PyInstaller on packaging/datc.spec, copies the data folders the program
reads next to the executables (experiments/, reference/, splits/, app/README.md
and the top-level documents), checks that `datc --version` runs, and writes
DaTC-Studio-<version>-<platform>-x64.zip (Windows) or .tar.gz (Linux) to --out.
The datasets and the computed results are not included: the release computes
them from the user's copy of the data, as the source tree does.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COPY_DIRS = ("experiments", "reference", "splits")
COPY_FILES = ("README.md", "LICENSE", "CITATION.cff", os.path.join("app", "README.md"))


def version():
    src = open(os.path.join(ROOT, "app", "datc_studio.py"), encoding="utf-8").read()
    return re.search(r'__version__ = "([^"]+)"', src).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "build"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()
    ver = version()
    plat = {"Windows": "windows", "Linux": "linux"}.get(platform.system())
    if plat is None:
        raise SystemExit(f"unsupported platform {platform.system()}")
    dist = os.path.join(a.work, "dist")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--workpath", os.path.join(a.work, "pyi"), "--distpath", dist, os.path.join(HERE, "datc.spec")]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    app = os.path.join(dist, "DaTC-Studio")
    for d in COPY_DIRS:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(app, d), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in COPY_FILES:
        os.makedirs(os.path.dirname(os.path.join(app, f)) or app, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), os.path.join(app, f))
    exe = os.path.join(app, "datc.exe" if plat == "windows" else "datc")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=300)
    print(out.stdout.strip(), out.stderr.strip()[-500:], flush=True)
    if out.returncode != 0 or ver not in out.stdout:
        raise SystemExit(f"the built program does not start: exit {out.returncode}")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"DaTC-Studio-{ver}-{plat}-x64")
    if plat == "windows":
        path = shutil.make_archive(stem, "zip", dist, "DaTC-Studio")
    else:
        path = stem + ".tar.gz"
        with tarfile.open(path, "w:gz") as t:   # tar keeps the executable bits
            t.add(app, arcname="DaTC-Studio")
    print(f"RELEASE BUILT {path} ({os.path.getsize(path) / 2**20:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
