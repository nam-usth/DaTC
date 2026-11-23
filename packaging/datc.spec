# PyInstaller spec of the DaTC Studio release (one folder, two executables sharing it).
#   DaTCStudio  the window (no console on Windows)
#   datc        the same program with a console: --selftest, --gif, --run-script, --version
# Built by packaging/build_release.py, which also copies the data folders next to the executables.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = SPECPATH + "/.."
sys.path.insert(0, ROOT + "/src")   # collect_submodules imports the package; pathex below is not enough

datc_modules = collect_submodules("datc")
assert "datc.scoring" in datc_modules, datc_modules   # used by the experiment scripts only
hidden = (datc_modules
          + collect_submodules("vtkmodules")
          + ["sklearn.metrics", "sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting",
             "sklearn.utils._vector_sentinel", "sklearn.neighbors._partition_nodes",
             # pyvista imports imageio only when a GIF is written, so PyInstaller does not see it
             "imageio", "imageio.v2", "imageio.plugins.pillow", "imageio.plugins.pillow_legacy",
             "concurrent.futures", "argparse", "glob", "runpy", "scipy.sparse.csgraph", "scipy.spatial.transform"])

a = Analysis(
    [ROOT + "/app/datc_studio.py"],
    pathex=[ROOT + "/src"],
    hiddenimports=hidden,
    datas=collect_data_files("pyvista") + collect_data_files("pyvistaqt") + copy_metadata("imageio"),
    # the app uses PyQt5; other Qt bindings in the env (vtk/matplotlib pull PySide6) abort the build
    excludes=["tkinter", "IPython", "jupyter", "notebook", "pytest", "tensorflow", "torch", "open3d",
              "PySide6", "PySide2", "PyQt6"],
    noarchive=False,
)
if sys.platform.startswith("linux"):
    # PyInstaller may pick the distribution's libssl/libcrypto over the environment's newer ones,
    # which libcurl and hdf5 need; where the environment has the library, ship that copy
    fixed = []
    for dest, src, kind in a.binaries:
        own = os.path.join(sys.prefix, "lib", os.path.basename(dest))
        if not os.path.realpath(src).startswith(os.path.realpath(sys.prefix) + os.sep) and os.path.exists(own):
            print("library from the environment:", dest, src, "->", own)
            src = own
        fixed.append((dest, src, kind))
    a.binaries = fixed

pyz = PYZ(a.pure)

common = dict(debug=False, bootloader_ignore_signals=False, strip=False, upx=False)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DaTCStudio", console=False, **common)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="datc", console=True, **common)
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="DaTC-Studio")
