"""DaTC Studio - match two fracture rims with the Darboux turn code, and browse or reproduce Table II.

    python app/datc_studio.py [fragment_a.ply fragment_b.ply | fantastic_breaks_object_folder]
    python app/datc_studio.py --selftest <object folder> out.png   (headless check, saves screenshots)
    python app/datc_studio.py --gif <object folder> out.gif [--icp]  (headless: the pairing GIF)
    datc --run-script experiments/<script>.py [args]               (release only: run a script)

Pair tab: two fragments in their assembled pose -> contact regions -> rims ->
(alpha, beta) -> 9 x 5 symbols -> cyclic Smith-Waterman, drawn live (src/datc/pipeline.py).
Retrieval tab: rankings stored by experiments/table1_reviewed.py, and buttons that run the
experiment scripts of the paper as separate processes, with their output streamed to the log.
The layout follows MS-DBSCAN Studio: a fixed-width control panel whose top-level regions have
a red border and whose parameter groups have a blue border, regions fold to their title.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

__version__ = "1.0.0"
SELFTEST = len(sys.argv) >= 4 and sys.argv[1] == "--selftest"
FROZEN = getattr(sys, "frozen", False)   # a PyInstaller release: the data folders sit next to the executable
ROOT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def selftest_failed(text: str):
    msg = f"SELFTEST FAILED ({__version__})\n{text}"
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.stem + "_selftest.txt").write_text(msg + "\n", encoding="utf-8")
    print(msg, flush=True)
    sys.exit(1)


try:
    import pyvista as pv  # noqa: E402
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
    from matplotlib.figure import Figure  # noqa: E402
    from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
    from pyvistaqt import QtInteractor  # noqa: E402

    from datc import datasets as DS  # noqa: E402
    from datc import geometry as G  # noqa: E402
    from datc.alphabet import Alphabet  # noqa: E402
    from datc.pipeline import Settings, frame_at, icp_refine, load_mask, match_rims, run_pair  # noqa: E402
    from scipy.spatial.transform import Rotation, Slerp  # noqa: E402
except Exception:
    if SELFTEST:
        selftest_failed(traceback.format_exc())
    raise

TITLE = "Matching Thin-Shell Fragments with a Darboux Turn Code"
MESH_EXT = (".ply", ".obj", ".stl", ".vtk", ".off")
POOL_CODE = ROOT / "results" / "table1_reviewed" / "fb_alphabet.json"
DEFAULTS = Settings()
PANEL_STYLE = ("QGroupBox { border: none; padding-top: 16px; margin-top: 11px; } "
               "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
RED = "QGroupBox { border: 1px solid red; }"
BLUE = "QGroupBox { border: 1px solid blue; }"
COL = dict(A="#d7263d", B="#2e86de", regA="#f6b26b", regB="#76c7c0", arc="#f1c40f",
           meshA="#e8e2da", meshB="#dde6ee", t="#d62728", g="#2ca02c", n="#1f77b4")
HELP = (ROOT / "app" / "README.md")


class Region(QtWidgets.QGroupBox):
    """A bordered region of the control panel that folds to its title (as in MS-DBSCAN Studio)."""

    def __init__(self, title: str, style: str, layout=QtWidgets.QVBoxLayout, expanded: bool = True):
        super().__init__(f"      {title} ")
        self.setStyleSheet(style)
        outer = QtWidgets.QVBoxLayout(self)
        self.margins = outer.contentsMargins()
        self.body = QtWidgets.QWidget()
        self.content = layout(self.body)
        self.content.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.body)
        self.arrow = QtWidgets.QToolButton(self)
        self.arrow.setAutoRaise(True)
        self.arrow.setFixedSize(16, 16)
        self.arrow.setCursor(QtCore.Qt.PointingHandCursor)
        self.arrow.clicked.connect(lambda: self.set_expanded(not self.expanded))
        self.set_expanded(expanded)

    def set_expanded(self, on: bool):
        self.expanded = bool(on)
        self.body.setVisible(self.expanded)
        self.arrow.setArrowType(QtCore.Qt.DownArrow if self.expanded else QtCore.Qt.RightArrow)
        m = self.margins
        self.layout().setContentsMargins(m.left(), m.top() if self.expanded else 0, m.right(),
                                         m.bottom() if self.expanded else 0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred if self.expanded else QtWidgets.QSizePolicy.Maximum)
        self.updateGeometry()

    def _folded(self, size):
        if not self.expanded:
            size.setHeight(self._title_rect().bottom() + 9)
        return size

    def sizeHint(self):
        return self._folded(super().sizeHint())

    def minimumSizeHint(self):
        return self._folded(super().minimumSizeHint())

    def _title_rect(self):
        opt = QtWidgets.QStyleOptionGroupBox()
        self.initStyleOption(opt)
        return self.style().subControlRect(QtWidgets.QStyle.CC_GroupBox, opt, QtWidgets.QStyle.SC_GroupBoxLabel, self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        r = self._title_rect()
        self.arrow.move(r.left() + 5, r.center().y() - self.arrow.height() // 2 + 1)

    def showEvent(self, e):
        super().showEvent(e)
        r = self._title_rect()
        self.arrow.move(r.left() + 5, r.center().y() - self.arrow.height() // 2 + 1)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and e.pos().y() <= self._title_rect().bottom():
            self.set_expanded(not self.expanded)
            return
        super().mousePressEvent(e)


def verdict_html(v, f, pose, pose_icp=None, icp_note=None):
    """The verdict in colour, the two inputs, the measured rates of the rule, and the ICP refinement."""
    if v["match"] is None:
        head = "<b>no verdict</b>"
    else:
        head = (f"<b style='color:{'#1a7f37' if v['match'] else '#c62828'}; font-size:14pt'>"
                f"{'MATCH' if v['match'] else 'NO MATCH'}</b>")
    lines = [head, f"score {f['score']:.4f} &nbsp; inlier share {100 * f['inliers']:.1f}%"]
    if pose.get("R") is not None:
        lines.append(f"RANSAC: {int(pose['inliers'].sum())} of {len(pose['path'])} matched samples within 3 ds; "
                     f"B turned by {pose['rotation_deg']:.1f}&deg;")
    else:
        lines.append("RANSAC: no pose (no consensus of 3 samples)")
    if pose_icp is not None:
        r = pose_icp["icp"]
        lines.append(f"ICP: RMS {r['rms0']:.3g} &rarr; {r['rms']:.3g} over {r['kept']} region points, "
                     f"{r['iters']} iterations; B turned a further {pose_icp['change_deg']:.2f}&deg; "
                     "(placement only, the verdict uses RANSAC)")
    elif icp_note:
        lines.append(f"ICP: {icp_note}")
    lines.append(f"<span style='color:#555'>{v['text'].split(': ', 1)[-1]}</span>")
    return "<br>".join(lines)


def spin(lo, hi, val, step, dec=3, tip=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setToolTip(tip)
    return s


def ispin(lo, hi, val, tip=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setToolTip(tip)
    return s


def form(rows):
    w = QtWidgets.QFormLayout()
    w.setLabelAlignment(QtCore.Qt.AlignRight)
    for k, v in rows:
        w.addRow(k, v)
    return w


class PairWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ma, mb, st, mask_a, mask_b, alph):
        super().__init__()
        self.args = (ma, mb, st, mask_a, mask_b, alph)

    def run(self):
        try:
            ma, mb, st, a, b, al = self.args
            self.done.emit(run_pair(ma, mb, st, a, b, al, progress=lambda p, s: self.progress.emit(p, s)))
        except Exception:  # shown to the user, never swallowed
            self.failed.emit(traceback.format_exc())


class LoadWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pa, pb):
        super().__init__()
        self.pa, self.pb = pa, pb

    def run(self):
        try:
            self.done.emit(DS.read_mesh(self.pa), DS.read_mesh(self.pb))
        except Exception:
            self.failed.emit(traceback.format_exc())


class IcpWorker(QtCore.QThread):
    """ICP refinement of a finished pair, off the main thread."""
    done = QtCore.pyqtSignal(object, object, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ma, fa, mb, fb, pose, ds, tau_ds):
        super().__init__()
        self.args = (ma, fa, mb, fb, pose, ds, tau_ds)

    def run(self):
        try:
            log = []
            out = icp_refine(*self.args, log=log)
            self.done.emit(out, log, self.args[-1])
        except Exception:
            self.failed.emit(traceback.format_exc())


class RetrievalWorker(QtCore.QThread):
    """Load the two fragments of a retrieval pair and run match_rims on their pool rims.

    The rims come from the pool cache unchanged; the meshes are read only to draw
    them and to orient the fracture planes (vertex centroid). Loaded meshes are
    returned so that the window can keep them for the next query."""
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pool, items, cache, roots, key, icp_tau=None):
        super().__init__()
        self.pool, self.items, self.cache, self.roots, self.key = pool, items, cache, roots, key
        self.icp_tau = icp_tau      # ICP reject distance in ds, or None for no ICP

    def _region(self, x, new):
        """Fracture region of a Fantastic Breaks rim: contact with its own mate, the rule that built the pool."""
        k = f"fbmask:{x['object']}"
        if k not in self.cache and k not in new:
            m = (self.cache.get(f"fb:{x['object']}") or new[f"fb:{x['object']}"])
            fb_, fr_, _, _ = DS.fb_masks(m["b"], m["r"])
            new[k] = {"b": fb_, "r": fr_}
        return (self.cache.get(k) or new[k])[x["side"]]

    def _mesh(self, x, new):
        if self.pool == "fb":
            k = f"fb:{x['object']}"
            if k not in self.cache and k not in new:
                if not self.roots["fb"]:
                    raise RuntimeError("set the Fantastic Breaks folder under Datasets to load the meshes")
                mb, mr = DS.fb_pair(self.roots["fb"], x["object"])
                new[k] = {"b": mb, "r": mr}
            return (self.cache.get(k) or new[k])[x["side"]]
        k = f"pr:{x['object']}"
        if k not in self.cache and k not in new:
            if not self.roots["presious"]:
                raise RuntimeError("set the PRESIOUS folder under Datasets to load the meshes")
            new[k] = DS.presious_object(self.roots["presious"], x["object"])[0]
        return (self.cache.get(k) or new[k])[x["frag"]]

    def run(self):
        try:
            new = {}
            meshes = [self._mesh(x, new) for x in self.items]
            rims = []
            for x in self.items:
                n = len(x["alpha"])
                rims.append(dict(alpha=np.asarray(x["alpha"]), beta=np.asarray(x["beta"]),
                                 pts=np.asarray(x["pts"], float)[:n]))
            log = []
            r = match_rims(rims[0], rims[1], float(self.items[0]["ds"]),
                           np.asarray(meshes[0].points).mean(0), np.asarray(meshes[1].points).mean(0), log=log)
            r["pose_icp"], r["icp_note"] = None, None
            if self.icp_tau is not None:
                if self.pool == "fb":
                    f1, f2 = self._region(self.items[0], new), self._region(self.items[1], new)
                    r["pose_icp"] = icp_refine(meshes[0], f1, meshes[1], f2, r["pose"], float(self.items[0]["ds"]),
                                               self.icp_tau, log)
                    if r["pose_icp"] is None:
                        r["icp_note"] = log[-1].split(": ", 1)[-1]
                else:
                    r["icp_note"] = ("not available for PRESIOUS pool rims (a fragment has one region per contact "
                                     "component, and the pool does not store which faces belong to which rim)")
                    log.append("ICP: " + r["icp_note"])
            r.update(rims=rims, meshes=meshes, new=new, log=log, key=self.key)
            self.done.emit(r)
        except Exception:
            self.failed.emit(traceback.format_exc())


def flat_axes(P):
    """Centre and principal axes of a rim: projecting on them shows its widest extent."""
    c = P.mean(0)
    return c, np.linalg.svd(P - c, full_matrices=False)[2]


def code_lines(alphabet, sym, per=24, rows=3):
    """The symbol string in rows of `per` symbols, cut after `rows` rows."""
    s = [alphabet.to_str(sym[i:i + per]) for i in range(0, len(sym), per)]
    more = f"\n... ({len(sym)} symbols)" if len(s) > rows else f"\n({len(sym)} symbols)"
    return "\n".join(s[:rows]) + more


class Studio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{TITLE} - DaTC Studio {__version__}")
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, int(0.92 * scr.width())), min(1000, int(0.9 * scr.height())))
        self.headless = False
        self.folder = os.environ.get("DATC_FB_ROOT", str(Path.cwd()))
        self.paths = {"A": None, "B": None}
        self.mask_paths = {"A": None, "B": None}
        self.meshes = None
        self.result = None
        self.worker = None
        self.loader = None
        self.proc = None
        self.last_error = None
        self.ret = None
        self.rworker = None
        self.r_pending = False
        self.rres = None
        self.rmesh_cache = {}
        self.icp_worker = None
        self.icp_again = False
        self.icp_note = None
        self.icp_tau_done = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _panel(self):
        panel = QtWidgets.QGroupBox()
        panel.setStyleSheet(PANEL_STYLE)
        panel.setFixedWidth(400)
        side = QtWidgets.QVBoxLayout(panel)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400 + scroll.verticalScrollBar().sizeHint().width() + 4)
        return scroll, side

    def _build(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.status = QtWidgets.QLabel("Choose two fragments, or a Fantastic Breaks object folder.")
        self.statusBar().addWidget(self.status, 1)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setFixedWidth(220)
        self.statusBar().addPermanentWidget(self.bar)
        self._build_pair_tab()
        self._build_retrieval_tab()
        help_view = QtWidgets.QTextBrowser()
        help_view.setOpenExternalLinks(True)
        if HELP.exists():
            help_view.setMarkdown(HELP.read_text(encoding="utf-8"))
        self.tabs.addTab(help_view, "Help")

    def _build_pair_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Pair")
        row = QtWidgets.QHBoxLayout(tab)
        scroll, side = self._panel()
        row.addWidget(scroll, 0)

        # ---- fragments
        frag = Region("Fragments", RED)
        fl = frag.content
        b = QtWidgets.QPushButton("Choose a Folder")
        b.clicked.connect(self.choose_folder)
        fl.addWidget(b)
        self.folder_label = QtWidgets.QLabel(self.folder)
        self.folder_label.setWordWrap(True)
        fl.addWidget(self.folder_label)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMinimumHeight(120)
        fl.addWidget(self.file_list)
        r = QtWidgets.QHBoxLayout()
        for tag in ("A", "B"):
            bb = QtWidgets.QPushButton(f"Set as {tag}")
            bb.clicked.connect(lambda _, t=tag: self.set_fragment(t))
            r.addWidget(bb)
        fl.addLayout(r)
        self.frag_label = {t: QtWidgets.QLabel(f"{t}: -") for t in ("A", "B")}
        for t in ("A", "B"):
            self.frag_label[t].setWordWrap(True)
            fl.addWidget(self.frag_label[t])
        masks = Region("Fracture region", BLUE)
        self.region_mode = QtWidgets.QComboBox()
        self.region_mode.addItems(["Contact in the assembled pose", "Vertex mask files"])
        self.region_mode.setToolTip("Contact: faces whose three vertices lie within eps of the other fragment "
                                    "(the paper's rule). Masks: one 0/1 per vertex (.txt, .npy, or .npz with 'mask').")
        masks.content.addWidget(self.region_mode)
        r = QtWidgets.QHBoxLayout()
        for tag in ("A", "B"):
            bb = QtWidgets.QPushButton(f"Mask {tag}...")
            bb.clicked.connect(lambda _, t=tag: self.choose_mask(t))
            r.addWidget(bb)
        masks.content.addLayout(r)
        self.mask_label = QtWidgets.QLabel("masks: none")
        self.mask_label.setWordWrap(True)
        masks.content.addWidget(self.mask_label)
        fl.addWidget(masks)
        side.addWidget(frag)

        # ---- method
        meth = Region("Darboux turn code", RED)
        rims = Region("Rims", BLUE, layout=QtWidgets.QFormLayout)
        self.ds = spin(1e-5, 1e4, DEFAULTS.ds, 0.001, 4, "Sampling step along the rim, in mesh units. "
                       "Fantastic Breaks: 0.0142; PRESIOUS: 4.85.")
        self.ds_auto = QtWidgets.QCheckBox("0.01 x joint diagonal")
        self.ds_auto.setToolTip("Use 0.01 x the bounding-box diagonal of both fragments instead of the step above")
        self.eps = spin(0.1, 10, DEFAULTS.eps, 0.1, 2, "Contact threshold, in units of 0.01 x joint diagonal")
        self.sm = spin(0, 20, DEFAULTS.sm, 0.5, 1, "Gaussian smoothing of the rim points, in samples")
        self.normals = QtWidgets.QComboBox()
        self.normals.addItems(["intact", "all", "fracture"])
        self.normals.setToolTip("Faces whose normals give the frame: intact side (paper), all, or fracture side")
        rims.content.addRow("step ds", self.ds)
        rims.content.addRow("", self.ds_auto)
        rims.content.addRow("contact eps", self.eps)
        rims.content.addRow("smoothing sm", self.sm)
        rims.content.addRow("normals", self.normals)
        meth.content.addWidget(rims)
        code = Region("Code", BLUE, layout=QtWidgets.QFormLayout)
        self.na = ispin(2, 16, DEFAULTS.na, "alpha bins")
        self.nb = ispin(1, 10, DEFAULTS.nb, "beta bins")
        self.code_src = QtWidgets.QComboBox()
        self.code_src.addItems(["Fit on the two rims", "Fantastic Breaks pool code"])
        self.code_src.setToolTip("Pool code: the 9 x 5 bin edges fitted on the 120 rims of the retrieval pool, "
                                 f"written by experiments/table1_reviewed.py to {POOL_CODE.relative_to(ROOT)}")
        code.content.addRow("alpha bins", self.na)
        code.content.addRow("beta bins", self.nb)
        code.content.addRow("bin edges", self.code_src)
        meth.content.addWidget(code)
        plc = Region("Placement", BLUE, layout=QtWidgets.QFormLayout)
        self.icp_chk = QtWidgets.QCheckBox("Refine by ICP on the fracture regions")
        self.icp_chk.setToolTip("Post-processing: trimmed point-to-point ICP of B's fracture-region vertices onto "
                                "A's, started from the RANSAC pose (datc.pose.icp). It changes the placement only.")
        self.icp_tau = spin(0.5, 50, DEFAULTS.icp_tau_ds, 0.5, 1,
                            "ICP drops point pairs farther apart than this, in sampling steps ds")
        self.icp_chk.toggled.connect(self.update_icp)
        self.icp_tau.valueChanged.connect(self.update_icp)
        plc.content.addRow(self.icp_chk)
        plc.content.addRow("ICP reject distance (ds)", self.icp_tau)
        meth.content.addWidget(plc)
        r = QtWidgets.QHBoxLayout()
        bd = QtWidgets.QPushButton("Load default")
        bd.clicked.connect(self.load_default)
        self.run_btn = QtWidgets.QPushButton("Run DaTC")
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self.run)
        r.addWidget(bd)
        r.addWidget(self.run_btn)
        meth.content.addLayout(r)
        side.addWidget(meth)

        # ---- result: the verdict, the numbers behind it, and the placement
        resr = Region("Result", RED)
        self.verdict_label = QtWidgets.QLabel("no result yet")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setTextFormat(QtCore.Qt.RichText)
        resr.content.addWidget(self.verdict_label)
        side.addWidget(resr)

        # ---- view
        view = Region("View", RED)
        self.chk = {}
        for key, text, on in (("mesh", "Fragments", True), ("region", "Fracture regions", True),
                              ("rim", "Rims", True), ("arc", "Matched samples", True),
                              ("frame", "Darboux frame (t red, g green, n blue)", True),
                              ("placed", "Place B by the RANSAC fit (+ ICP if on)", False),
                              ("apart", "Pull the fragments apart", False)):
            c = QtWidgets.QCheckBox(text)
            c.setChecked(on)
            c.toggled.connect(self.redraw)
            if key == "apart":
                c.toggled.connect(lambda _: self.plotter.reset_camera())
            self.chk[key] = c
            view.content.addWidget(c)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(QtWidgets.QLabel("frame at sample"))
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.redraw)
        r.addWidget(self.frame_slider)
        view.content.addLayout(r)
        self.frame_info = QtWidgets.QLabel("")
        self.frame_info.setWordWrap(True)
        view.content.addWidget(self.frame_info)
        side.addWidget(view)

        exp = Region("Export", RED, expanded=False)
        b1 = QtWidgets.QPushButton("Save the 3D view (PNG)")
        b1.clicked.connect(lambda: self.save_png())
        b2 = QtWidgets.QPushButton("Save the result (JSON)")
        b2.clicked.connect(lambda: self.save_json())
        b3 = QtWidgets.QPushButton("Save the pairing (GIF)")
        b3.setToolTip("B moves from apart to the pose found by RANSAC")
        b3.clicked.connect(lambda: self.save_gif())
        exp.content.addWidget(b1)
        exp.content.addWidget(b2)
        exp.content.addWidget(b3)
        side.addWidget(exp)
        self.export_region = exp
        side.addStretch(1)

        # ---- right: 3D on top, codes and log below
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plotter = QtInteractor(right)
        self.plotter.set_background("white")
        right.addWidget(self.plotter.interactor)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.fig = Figure(figsize=(7, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        bottom.addWidget(self.canvas)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.log_box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        bottom.addWidget(self.log_box)
        bottom.setSizes([700, 400])
        right.addWidget(bottom)
        right.setSizes([620, 300])
        row.addWidget(right, 1)
        self._refresh_files()

    def _build_retrieval_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Retrieval")
        row = QtWidgets.QHBoxLayout(tab)
        scroll, side = self._panel()
        row.addWidget(scroll, 0)

        rep = Region("Reproduce Table II", RED)
        roots = Region("Datasets", BLUE, layout=QtWidgets.QFormLayout)
        self.fb_root = QtWidgets.QLineEdit(os.environ.get("DATC_FB_ROOT", ""))
        self.fb_root.setPlaceholderText("folder with 00/, 01/, ... of Fantastic Breaks")
        self.pr_root = QtWidgets.QLineEdit(os.environ.get("DATC_PRESIOUS_ROOT", ""))
        self.pr_root.setPlaceholderText("folder with DoraBlock/, DoraColumn/, ...")
        roots.content.addRow("Fantastic Breaks", self.fb_root)
        roots.content.addRow("PRESIOUS", self.pr_root)
        rep.content.addWidget(roots)
        self.procs = ispin(1, 64, max(1, (os.cpu_count() or 2) - 1), "worker processes for scoring")
        f = form([("processes", self.procs)])
        rep.content.addLayout(f)
        self.steps = []
        for text, script, args in (
                ("1. Build the rim pools", "experiments/build_pools.py", "roots"),
                ("2. Upper block (reviewed settings)", "experiments/table1_reviewed.py", "procs"),
                ("3. Lower block (tuned alike)", "experiments/table1_tuned.py", "procs"),
                ("4. Check against the paper", "experiments/make_table1.py", "")):
            bb = QtWidgets.QPushButton(text)
            bb.clicked.connect(lambda _, s=script, a=args: self.run_script(s, a))
            rep.content.addWidget(bb)
            self.steps.append(bb)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_script)
        rep.content.addWidget(self.stop_btn)
        side.addWidget(rep)

        brow = Region("Rankings", RED)
        r = QtWidgets.QHBoxLayout()
        self.pool_box = QtWidgets.QComboBox()
        self.pool_box.addItems(["fb", "presious"])
        self.arm_box = QtWidgets.QComboBox()
        self.arm_box.addItems(["S_sw", "S_dtw_ab_cyc", "S_ddtw_kt_cyc", "S_rim_length", "S_rim_planarity"])
        r.addWidget(self.pool_box)
        r.addWidget(self.arm_box)
        brow.content.addLayout(r)
        bl = QtWidgets.QPushButton("Load rankings")
        bl.clicked.connect(self.load_rankings)
        brow.content.addWidget(bl)
        self.rank_summary = QtWidgets.QLabel("")
        self.rank_summary.setWordWrap(True)
        brow.content.addWidget(self.rank_summary)
        self.query_table = QtWidgets.QTableWidget(0, 4)
        self.query_table.setHorizontalHeaderLabels(["query", "mate rank", "first candidate", ""])
        self.query_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.query_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.query_table.verticalHeader().setVisible(False)
        self.query_table.setMinimumHeight(300)
        self.query_table.itemSelectionChanged.connect(self.show_query)
        brow.content.addWidget(self.query_table)
        cr = QtWidgets.QHBoxLayout()
        cr.addWidget(QtWidgets.QLabel("Fragment 2 is the"))
        self.cand_box = QtWidgets.QComboBox()
        self.cand_box.addItems(["first candidate", "mate"])
        self.cand_box.setToolTip("Which rim of the ranking is drawn against the query")
        self.cand_box.currentIndexChanged.connect(self.show_query)
        cr.addWidget(self.cand_box, 1)
        brow.content.addLayout(cr)
        self.ret_icp = QtWidgets.QCheckBox("Refine the placement by ICP (Fantastic Breaks)")
        self.ret_icp.setToolTip("ICP of fragment 2's fracture region onto fragment 1's, from the RANSAC pose; "
                                "reject distance as set in the Pair tab. Regions: contact with each fragment's own "
                                "mate, the rule that built the pool rims.")
        self.ret_icp.toggled.connect(self.show_query)
        brow.content.addWidget(self.ret_icp)
        self.ret_verdict = QtWidgets.QLabel("")
        self.ret_verdict.setWordWrap(True)
        self.ret_verdict.setTextFormat(QtCore.Qt.RichText)
        brow.content.addWidget(self.ret_verdict)
        side.addWidget(brow)
        side.addStretch(1)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.rplot = QtInteractor(right, shape=(2, 3))
        self.rplot.set_background("white")
        right.addWidget(self.rplot.interactor)
        self.script_log = QtWidgets.QPlainTextEdit()
        self.script_log.setReadOnly(True)
        self.script_log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        right.addWidget(self.script_log)
        right.setSizes([600, 320])
        row.addWidget(right, 1)

    # ------------------------------------------------------------------ messages
    def log(self, s):
        self.log_box.appendPlainText(s)

    def error(self, text):
        self.last_error = text
        self.log("ERROR\n" + text)
        self.status.setText(text.strip().splitlines()[-1])
        if not self.headless:
            QtWidgets.QMessageBox.critical(self, "DaTC Studio", text.strip().splitlines()[-1])

    # ------------------------------------------------------------------ fragments
    def choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with fragment meshes", self.folder)
        if d:
            self.open_folder(d)

    def open_folder(self, d):
        self.folder = d
        self.folder_label.setText(d)
        self._refresh_files()
        b, r = os.path.join(d, "model_b_0.ply"), os.path.join(d, "model_r_0.ply")
        if os.path.exists(b) and os.path.exists(r):   # a Fantastic Breaks object folder
            self.paths = {"A": b, "B": r}
            for t in ("A", "B"):
                self.frag_label[t].setText(f"{t}: {os.path.basename(self.paths[t])}")
            self.log(f"Fantastic Breaks object: A = model_b_0.ply, B = model_r_0.ply in {d}")
            self.load_meshes()

    def _refresh_files(self):
        self.file_list.clear()
        if os.path.isdir(self.folder):
            for f in sorted(os.listdir(self.folder)):
                if f.lower().endswith(MESH_EXT):
                    self.file_list.addItem(f)

    def set_fragment(self, tag):
        it = self.file_list.currentItem()
        if it is None:
            return
        self.paths[tag] = os.path.join(self.folder, it.text())
        self.frag_label[tag].setText(f"{tag}: {it.text()}")
        if all(self.paths.values()):
            self.load_meshes()

    def set_pair(self, pa, pb):
        self.paths = {"A": pa, "B": pb}
        for t in ("A", "B"):
            self.frag_label[t].setText(f"{t}: {os.path.basename(self.paths[t])}")
        self.load_meshes()

    def choose_mask(self, tag):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Vertex mask of fragment {tag}", self.folder,
                                                     "Masks (*.txt *.npy *.npz)")
        if p:
            self.mask_paths[tag] = p
            self.region_mode.setCurrentIndex(1)
            self.mask_label.setText("masks: " + ", ".join(f"{t} {os.path.basename(v)}"
                                                          for t, v in self.mask_paths.items() if v))

    def load_meshes(self):
        self.meshes = None
        self.result = None
        self.status.setText("loading the fragments")
        self.bar.setRange(0, 0)
        self.loader = LoadWorker(self.paths["A"], self.paths["B"])
        self.loader.done.connect(self._loaded)
        self.loader.failed.connect(lambda t: (self.bar.setRange(0, 100), self.error(t)))
        self.loader.start()

    def _loaded(self, ma, mb):
        self.meshes = (ma, mb)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.log(f"A: {ma.n_points} vertices, {ma.n_cells} faces;  B: {mb.n_points} vertices, {mb.n_cells} faces")
        self.status.setText("fragments loaded; press Run DaTC")
        self.redraw()
        self.plotter.reset_camera()

    # ------------------------------------------------------------------ run
    def load_default(self):
        self.ds.setValue(DEFAULTS.ds)
        self.ds_auto.setChecked(False)
        self.eps.setValue(DEFAULTS.eps)
        self.sm.setValue(DEFAULTS.sm)
        self.normals.setCurrentText(DEFAULTS.normals)
        self.na.setValue(DEFAULTS.na)
        self.nb.setValue(DEFAULTS.nb)
        self.icp_chk.setChecked(DEFAULTS.icp)
        self.icp_tau.setValue(DEFAULTS.icp_tau_ds)

    def settings(self):
        return Settings(ds=self.ds.value(), ds_rel=0.01 if self.ds_auto.isChecked() else None,
                        eps=self.eps.value(), sm=self.sm.value(), normals=self.normals.currentText(),
                        na=self.na.value(), nb=self.nb.value())

    def run(self):
        if self.meshes is None:
            self.error("Load two fragments first.")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        ma, mb = self.meshes
        mask_a = mask_b = None
        try:
            if self.region_mode.currentIndex() == 1:
                if not all(self.mask_paths.values()):
                    raise ValueError("Vertex mask files: choose a mask for A and for B")
                mask_a = load_mask(self.mask_paths["A"], ma.n_points)
                mask_b = load_mask(self.mask_paths["B"], mb.n_points)
            alph = None
            if self.code_src.currentIndex() == 1:
                if not POOL_CODE.exists():
                    raise ValueError(f"{POOL_CODE} not found: run step 2 of the Retrieval tab first")
                alph = Alphabet.from_json(str(POOL_CODE))
        except Exception as e:
            self.error(str(e))
            return
        self.run_btn.setEnabled(False)
        self.t0 = time.time()
        self.worker = PairWorker(ma, mb, self.settings(), mask_a, mask_b, alph)
        self.worker.progress.connect(lambda p, s: (self.bar.setValue(p), self.status.setText(s)))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda t: (self.run_btn.setEnabled(True), self.error(t)))
        self.worker.start()

    def _done(self, res):
        self.result = res
        for m, mask in zip(self.meshes, res.masks):   # region colouring, set once per run
            m.cell_data["region"] = mask.astype(np.uint8)
        self.run_btn.setEnabled(True)
        self.log("")
        for s in res.log:
            self.log(s)
        n = len(res.rims[0]["alpha"])
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, n - 1)
        self.frame_slider.setValue(int(res.path[len(res.path) // 2, 0]) if len(res.path) else 0)
        self.frame_slider.blockSignals(False)
        self.icp_note = None
        self.show_verdict()
        self.status.setText(("MATCH" if res.verdict["match"] else "NO MATCH" if res.verdict["match"] is False
                             else "no verdict") + f": score {res.score:.4f} (SW / (n_A + n_B)), "
                            f"{len(res.path)} matched samples, {time.time() - self.t0:.1f} s")
        self.export_region.set_expanded(True)
        self.redraw()
        self.draw_codes()
        self.update_icp()

    def show_verdict(self):
        res = self.result
        if res is not None:
            self.verdict_label.setText(verdict_html(res.verdict, res.features, res.pose,
                                                    res.pose_icp if self.icp_chk.isChecked() else None,
                                                    self.icp_note if self.icp_chk.isChecked() else None))

    def update_icp(self, *_):
        """Run ICP when it is switched on (or its distance changes) and the current result has none for it."""
        res = self.result
        if res is None:
            return
        if self.icp_chk.isChecked():
            tau = self.icp_tau.value()
            if getattr(self, "icp_tau_done", None) != tau or (res.pose_icp is None and self.icp_note is None):
                if self.icp_worker is not None and self.icp_worker.isRunning():
                    self.icp_again = True
                    return
                self.icp_again = False
                self.status.setText("ICP on the fracture regions")
                ma, mb = self.meshes
                self.icp_worker = IcpWorker(ma, res.masks[0], mb, res.masks[1], res.pose, res.ds, tau)
                self.icp_worker.done.connect(self._icp_done)
                self.icp_worker.failed.connect(self.error)
                self.icp_worker.start()
                return
        self.show_verdict()
        self.redraw()

    def _icp_done(self, pose_icp, log, tau):
        res = self.result
        res.pose_icp = pose_icp
        self.icp_tau_done = tau
        self.icp_note = None if pose_icp is not None else log[-1].split(": ", 1)[-1]
        for s in log:
            self.log(s)
        self.status.setText(log[-1] if log else "ICP done")
        if self.icp_again:
            self.icp_again = False
            self.update_icp()
            return
        self.show_verdict()
        self.redraw()

    def active_pose(self):
        """The placement shown: the ICP refinement when it is on and ran, else the RANSAC pose."""
        res = self.result
        if res is None:
            return None
        if self.icp_chk.isChecked() and res.pose_icp is not None:
            return res.pose_icp
        return res.pose

    # ------------------------------------------------------------------ drawing
    def _offset(self):
        if not (self.chk["apart"].isChecked() and self.meshes):
            return np.zeros(3)
        ca = np.asarray(self.meshes[0].points).mean(0)
        cb = np.asarray(self.meshes[1].points).mean(0)
        d = cb - ca
        L = float(np.linalg.norm(np.ptp(np.asarray(self.meshes[0].points), 0)))
        return d / (np.linalg.norm(d) + 1e-12) * 0.6 * L

    def _placement(self):
        """4 x 4 transform applied to B: the RANSAC pose when 'Place B' is ticked, then the pull-apart offset."""
        T = np.eye(4)
        res = self.result
        pose = self.active_pose()
        if res is not None and self.chk["placed"].isChecked() and pose["R"] is not None:
            T[:3, :3], T[:3, 3] = pose["R"], pose["t"]
        T[:3, 3] += self._offset()
        return T

    def redraw(self, *_):
        p = self.plotter
        cam = p.camera_position if self.meshes is not None and p.renderer.actors else None
        p.clear()
        if self.meshes is None:
            p.render()
            return
        TB = self._placement()
        moved = not np.allclose(TB, np.eye(4))
        res = self.result
        for k, (m, tag) in enumerate(zip(self.meshes, ("A", "B"))):
            if not self.chk["mesh"].isChecked():
                continue
            mm = m if (k == 0 or not moved) else m.transform(TB, inplace=False)
            if res is not None and self.chk["region"].isChecked():
                p.add_mesh(mm, scalars="region", cmap=[COL["mesh" + tag], COL["reg" + tag]],
                           clim=[0, 1], show_scalar_bar=False, smooth_shading=True)
            else:
                p.add_mesh(mm, color=COL["mesh" + tag], smooth_shading=True)
        if res is not None:
            r_ = 0.35 * res.settings.ds if res.settings.ds_rel is None else 0.0035 * self._diag()
            for k, (rim, tag) in enumerate(zip(res.rims, ("A", "B"))):
                if not self.chk["rim"].isChecked():
                    continue
                P = rim["pts"] if k == 0 else rim["pts"] @ TB[:3, :3].T + TB[:3, 3]
                line = pv.lines_from_points(P, close=True)
                hit = np.zeros(len(P))
                if self.chk["arc"].isChecked() and len(res.path):
                    hit[np.unique(res.path[:, k])] = 1.0
                line.point_data["matched"] = hit
                p.add_mesh(line.tube(radius=r_), scalars="matched", cmap=[COL[tag], COL["arc"]],
                           clim=[0, 1], show_scalar_bar=False)
            if self.chk["frame"].isChecked():
                i = self.frame_slider.value()
                x, t, g, n = frame_at(res.rims[0], i, res.settings.sm)
                L = 12 * r_ / 0.35
                for v, c in ((t, COL["t"]), (g, COL["g"]), (n, COL["n"])):
                    p.add_mesh(pv.Arrow(start=x, direction=v, scale=L), color=c)
                a, b = res.rims[0]["alpha"][i], res.rims[0]["beta"][i]
                s = res.alphabet.to_str(res.symbols[0][i:i + 1])
                self.frame_info.setText(f"sample {i} of A: alpha = {a:+.4f}, beta = {b:+.4f}, symbol {s}")
        if cam is not None:
            p.camera_position = cam
        p.render()

    def _diag(self):
        P = np.vstack([np.asarray(m.points) for m in self.meshes])
        return float(np.linalg.norm(np.ptp(P, 0)))

    def draw_codes(self):
        res = self.result
        self.fig.clear()
        if res is None:
            self.canvas.draw()
            return
        A, B = res.rims
        mate = res.direction == "mate"
        ax1, ax2 = self.fig.add_subplot(211), self.fig.add_subplot(212, sharex=None)
        ia, ib = res.path[:, 0], res.path[:, 1]
        sgn = -1.0 if mate else 1.0
        ax1.plot(A["alpha"], color=COL["A"], lw=1.0, label="A")
        ax1.plot(ia, sgn * B["alpha"][ib], ".", color=COL["B"], ms=2.5,
                 label="B at its aligned sample" + (" (alpha negated, reversed walk)" if mate else ""))
        ax2.plot(A["beta"], color=COL["A"], lw=1.0)
        ax2.plot(ia, B["beta"][ib], ".", color=COL["B"], ms=2.5)
        for ax, lab in ((ax1, "alpha"), (ax2, "beta")):
            if len(ia):
                ax.axvspan(ia.min(), ia.max(), color=COL["arc"], alpha=0.15, lw=0)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        ax2.set_xlabel("sample along rim A")
        ax1.legend(fontsize=7, loc="upper right")
        sa = res.alphabet.to_str(res.symbols[0][:40])
        ax1.set_title(f"score {res.score:.4f}   A: {sa}...", fontsize=8, loc="left")
        self.canvas.draw()

    # ------------------------------------------------------------------ export
    def save_png(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the 3D view", "datc_pair.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)
            self.log(f"saved {path}")
        return path

    def save_json(self, path=None):
        res = self.result
        if res is None:
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the result", "datc_pair.json", "JSON (*.json)")
        if not path:
            return None
        out = dict(fragments=self.paths, settings=vars(res.settings), score=res.score, raw_score=res.raw_score,
                   direction=res.direction, alphabet=res.alphabet_source,
                   n=[len(r["alpha"]) for r in res.rims], symbols=[res.alphabet.to_str(s) for s in res.symbols],
                   path=res.path.tolist(), log=res.log, version=__version__,
                   verdict=dict(match=res.verdict["match"], text=res.verdict["text"]), verdict_inputs=res.features,
                   pose=dict(direction=res.pose["direction"],
                             R=None if res.pose["R"] is None else res.pose["R"].tolist(),
                             t=None if res.pose["t"] is None else res.pose["t"].tolist(),
                             inliers=int(res.pose["inliers"].sum()), matched=len(res.pose["path"]),
                             rotation_deg=res.pose["rotation_deg"]))
        pi = res.pose_icp if self.icp_chk.isChecked() else None
        out["pose_icp"] = None if pi is None else dict(
            R=pi["R"].tolist(), t=pi["t"].tolist(), rotation_deg=pi["rotation_deg"], change_deg=pi["change_deg"],
            reject_distance=pi["tau"], region_points=[pi["n_src"], pi["n_dst"]],
            **{k: pi["icp"][k] for k in ("rms0", "rms", "kept0", "kept", "iters", "converged")})
        Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")
        self.log(f"saved {path}")
        return path

    def save_gif(self, path=None, n_frames=40):
        """B travels from 'pulled apart' to the placement (RANSAC, + ICP when on); the last frames hold it."""
        res = self.result
        if res is None or res.pose["R"] is None:
            self.error("No placement to animate: run DaTC first (RANSAC must have found a pose).")
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the pairing", "datc_pairing.gif", "GIF (*.gif)")
        if not path:
            return None
        self.status.setText("writing the GIF")
        QtWidgets.QApplication.processEvents()
        ma, mb = self.meshes
        pose = self.active_pose()
        R, t = pose["R"], pose["t"]
        cB = np.asarray(mb.points).mean(0)
        L = float(np.linalg.norm(np.ptp(np.asarray(ma.points), 0)))
        away = cB - np.asarray(ma.points).mean(0)
        away = away / (np.linalg.norm(away) + 1e-12) * 0.8 * L
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([np.eye(3), R])))
        rA = res.rims[0]["pts"]
        rB = res.rims[1]["pts"]
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.set_background("white")
        pl.add_mesh(ma, color=COL["meshA"], smooth_shading=True)
        r_ = 0.35 * res.ds
        pl.add_mesh(pv.lines_from_points(rA, close=True).tube(radius=r_), color=COL["A"])
        actorB = pl.add_mesh(mb, color=COL["meshB"], smooth_shading=True)
        actorRB = pl.add_mesh(pv.lines_from_points(rB, close=True).tube(radius=r_), color=COL["B"])
        word = "MATCH" if res.verdict["match"] else ("NO MATCH" if res.verdict["match"] is False else "")
        how = "RANSAC + ICP" if pose is not res.pose else "RANSAC"
        pl.add_text(f"DaTC + {how}   {word}   score {res.features['score']:.3f}, "
                    f"inliers {100 * res.features['inliers']:.0f}%", font_size=10, color="black")
        pl.camera_position = "iso"
        # frame everything that will be on screen: A, B apart, B placed
        pts = np.vstack([np.asarray(ma.points)[::50], np.asarray(mb.points)[::50] + away,
                         np.asarray(mb.points)[::50] @ R.T + t])
        pl.reset_camera(bounds=pv.PolyData(pts).bounds)
        pl.open_gif(path, fps=12)
        s_all = list(np.linspace(0, 1, n_frames) ** 0.7) + [1.0] * 12
        for s in s_all:
            Rs = slerp([s]).as_matrix()[0]
            T = np.eye(4)
            T[:3, :3] = Rs
            T[:3, 3] = cB - Rs @ cB + s * (R @ cB + t - cB) + (1 - s) * away
            for a in (actorB, actorRB):
                a.user_matrix = T
            pl.write_frame()
        pl.close()
        self.log(f"saved {path} ({len(s_all)} frames)")
        self.status.setText(f"saved {os.path.basename(path)}")
        return path

    # ------------------------------------------------------------------ retrieval tab
    def run_script(self, script, args):
        if self.proc is not None and self.proc.state() != QtCore.QProcess.NotRunning:
            self.script_log.appendPlainText("a script is already running")
            return
        argv = ["--run-script", str(ROOT / script)] if FROZEN else ["-u", str(ROOT / script)]
        if args == "roots":
            if self.fb_root.text().strip():
                argv += ["--fb-root", self.fb_root.text().strip()]
            if self.pr_root.text().strip():
                argv += ["--presious-root", self.pr_root.text().strip()]
        elif args == "procs":
            argv += ["--procs", str(self.procs.value())]
        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: self.script_log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        self.proc.finished.connect(self._script_done)
        # a release runs the scripts with its console executable `datc`, which carries the bundled Python
        exe = str(ROOT / ("datc.exe" if os.name == "nt" else "datc")) if FROZEN else sys.executable
        shown = ["datc"] + argv if FROZEN else ["python"] + argv[1:]
        self.script_log.appendPlainText(f"\n$ {' '.join(shown)}")
        for b in self.steps:
            b.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.proc.start(exe, argv)

    def _script_done(self, code, _status):
        self.script_log.appendPlainText(f"[exit code {code}]")
        for b in self.steps:
            b.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_script(self):
        if self.proc is not None:
            self.proc.kill()

    def load_rankings(self):
        pool = self.pool_box.currentText()
        npz = ROOT / "results" / "table1_reviewed" / f"{pool}_scores.npz"
        pkl = ROOT / "cache" / ("fb_pool60.pkl" if pool == "fb" else "presious.pkl")
        if not npz.exists() or not pkl.exists():
            self.rank_summary.setText(f"missing {npz.name} or {pkl.name}: run steps 1 and 2 first")
            return
        z = np.load(npz)
        with open(pkl, "rb") as f:
            items = pickle.load(f)["items"]
        arm = self.arm_box.currentText()
        S = z[arm].astype(float)
        mate = z["mate"]
        rows = []
        for i, j in enumerate(mate):
            if j < 0:
                continue
            s = S[i].copy()
            s[i] = -np.inf
            better = int((s > s[j]).sum())
            tied = int((s == s[j]).sum()) - 1
            rows.append((i, int(j), better + tied / 2 + 1, int(np.argmax(s))))
        self.ret = dict(pool=pool, items=items, S=S, rows=rows, names=[str(x) for x in z["names"]])
        top1 = np.mean([r[2] == 1 for r in rows])
        self.rank_summary.setText(f"{pool}, {arm}: {len(rows)} queries, mate ranked first (no tie) for "
                                  f"{100 * top1:.1f}%. Select a query to see it, its mate and its first candidate.")
        self.query_table.setRowCount(len(rows))
        for k, (i, j, mid, best) in enumerate(rows):
            for c, v in enumerate((self.ret["names"][i], f"{mid:g}", self.ret["names"][best],
                                   "mate" if best == j else "")):
                self.query_table.setItem(k, c, QtWidgets.QTableWidgetItem(v))
        self.query_table.resizeColumnsToContents()

    def show_query(self):
        if self.ret is None:
            return
        sel = self.query_table.selectionModel().selectedRows()
        if not sel:
            return
        i, j, mid, best = self.ret["rows"][sel[0].row()]
        other = best if self.cand_box.currentIndex() == 0 else j
        key = (self.ret["pool"], i, other)
        if self.rworker is not None and self.rworker.isRunning():
            self.r_pending = True          # drawn when the running pair is done
            return
        self.r_pending = False
        items = self.ret["items"]
        roots = dict(fb=self.fb_root.text().strip(), presious=self.pr_root.text().strip())
        role = "mate" if other == j else "first candidate (not the mate)"
        self.ret_verdict.setText(f"matching query {self.ret['names'][i]} with its {role} "
                                 f"{self.ret['names'][other]} ...")
        self.rworker = RetrievalWorker(self.ret["pool"], [items[i], items[other]], self.rmesh_cache, roots, key,
                                       self.icp_tau.value() if self.ret_icp.isChecked() else None)
        self.rworker.done.connect(self._ret_done)
        self.rworker.failed.connect(self._ret_failed)
        self.rworker.start()

    def _ret_failed(self, text):
        self.last_error = text
        self.ret_verdict.setText(f"<span style='color:#c62828'>{text.strip().splitlines()[-1]}</span>")
        self.script_log.appendPlainText(text)
        if self.r_pending:
            self.show_query()

    def _ret_done(self, r):
        self.rmesh_cache.update(r.pop("new"))
        self.rres = r
        if self.r_pending:          # the selection moved on while this pair ran
            self.show_query()
            return
        pool, i, other = r["key"]
        names = self.ret["names"]
        mate = self.ret["rows"][[q[0] for q in self.ret["rows"]].index(i)][1]
        role = "mate" if other == mate else "first candidate, not the mate"
        self.ret_verdict.setText(f"{names[i]} vs {names[other]} ({role})<br>"
                                 + verdict_html(r["verdict"], r["features"], r["pose"], r["pose_icp"], r["icp_note"]))
        self.script_log.appendPlainText("\n".join(r["log"]))
        self.draw_retrieval(r, names[i], names[other])

    def draw_retrieval(self, r, name1, name2):
        """Six panels: the two fragments, the placed pair; the two coded rims, the Smith-Waterman matches."""
        p = self.rplot
        p.clear()
        rims, meshes, al, sy, pose = r["rims"], r["meshes"], r["alphabet"], r["symbols"], r["pose"]
        P1, P2 = rims[0]["pts"], rims[1]["pts"]
        diag = float(np.linalg.norm(np.ptp(np.vstack([P1, P2]), 0)))
        rad = 0.006 * diag
        T = np.eye(4)
        if pose["R"] is not None:
            T[:3, :3], T[:3, 3] = pose["R"], pose["t"]

        def tube(P, **kw):
            return pv.lines_from_points(P, close=True).tube(radius=rad, **kw)

        def title(txt, color="black"):
            p.add_text(txt, position="upper_left", font_size=11, color=color)

        # row 0: the fragments and the placed pair
        for c, (m, P, tag, nm) in enumerate(((meshes[0], P1, "A", name1), (meshes[1], P2, "B", name2))):
            p.subplot(0, c)
            p.add_mesh(m, color=COL["mesh" + tag], smooth_shading=True)
            p.add_mesh(tube(P), color=COL[tag])
            title(f"fragment {c + 1}: {nm}", COL[tag])
            p.reset_camera()
        p.subplot(0, 2)
        p.add_mesh(meshes[0], color=COL["meshA"], smooth_shading=True)
        p.add_mesh(tube(P1), color=COL["A"])
        if pose["R"] is not None:
            pz = r["pose_icp"] if r["pose_icp"] is not None else pose
            Tz = np.eye(4)
            Tz[:3, :3], Tz[:3, 3] = pz["R"], pz["t"]
            p.add_mesh(meshes[1].transform(Tz, inplace=False), color=COL["meshB"], smooth_shading=True, opacity=0.85)
            p.add_mesh(tube((pz["R"] @ P2.T).T + pz["t"]), color=COL["B"])
            v = r["verdict"]
            head = "no verdict" if v["match"] is None else ("MATCH" if v["match"] else "NO MATCH")
            how = f"RANSAC + ICP ({pz['change_deg']:.1f} deg more)" if pz is not pose else "RANSAC"
            title(f"{head}: fragment 2 placed by {how}\n{int(pose['inliers'].sum())} of {len(pose['path'])} "
                  f"matched samples within 3 ds, turned {pz['rotation_deg']:.1f} deg",
                  "#1a7f37" if v["match"] else "#c62828")
        else:
            title("NO MATCH: RANSAC found no pose\n(no consensus of 3 matched samples)", "#c62828")
        p.reset_camera()

        # row 1: the rims in the plane of rim 1, coloured by alpha bin, with the code
        c1, ax1 = flat_axes(P1)
        Q2 = (pose["R"] @ P2.T).T + pose["t"] if pose["R"] is not None else P2
        c2, ax2 = (c1, ax1) if pose["R"] is not None else flat_axes(P2)
        F1 = (P1 - c1) @ ax1.T
        F2 = (Q2 - c2) @ ax2.T
        bar = dict(title="alpha bin", n_labels=min(al.na, 5), fmt="%.0f", vertical=True, height=0.45,
                   width=0.05, position_x=0.9, position_y=0.3, label_font_size=10, title_font_size=10)
        for c, (F, s, nm, tag) in enumerate(((F1, sy[0], name1, "A"), (F2, sy[1], name2, "B"))):
            p.subplot(1, c)
            line = pv.lines_from_points(F, close=True)
            line.point_data["alpha bin"] = s[:, 0].astype(float)
            p.add_mesh(line.tube(radius=rad), scalars="alpha bin", cmap="coolwarm", clim=[0, al.na - 1],
                       n_colors=al.na, scalar_bar_args=dict(bar), show_scalar_bar=(c == 0))
            p.add_points(F[:1], color="black", point_size=10, render_points_as_spheres=True)
            title(f"rim {c + 1} and its DaTC code ({al.na} x {al.nb}, {r['alphabet_source']})", COL[tag])
            p.add_text("code from the black dot: " + code_lines(al, s), position="lower_left", font_size=8,
                       color="black", font="courier")
            p.view_xy()
            p.reset_camera()

        # (1, 2): the Smith-Waterman correspondence, rim 2 moved clear of rim 1 across its narrow axis
        p.subplot(1, 2)
        h = float(np.ptp(F1[:, 1]))
        shift = np.array([0.0, F1[:, 1].max() - F2[:, 1].min() + 0.6 * h + 0.05 * diag, 0.0])
        G2 = F2 + shift
        p.add_mesh(tube(F1), color=COL["A"])
        p.add_mesh(tube(G2), color=COL["B"])
        path = pose["path"] if pose["R"] is not None else r["path"]
        inl = pose["inliers"] if pose["R"] is not None else np.zeros(len(path), bool)
        step = max(1, len(path) // 60)
        for keep, col in ((inl, "#2ca02c"), (~inl, "#9e9e9e")):
            k = np.flatnonzero(keep)[::step]
            if len(k):
                a, b = F1[path[k, 0]], G2[path[k, 1]]
                pts = np.vstack([a, b])
                cells = np.hstack([[2, q, q + len(k)] for q in range(len(k))])
                p.add_mesh(pv.PolyData(pts, lines=cells), color=col, line_width=1.5)
        d = pose["direction"] if pose["R"] is not None else r["direction"]
        title(f"Smith-Waterman ({d}): {len(path)} matched samples, score {r['features']['score']:.4f}\n"
              f"green: within 3 ds after RANSAC; grey: outside; 1 in {step} drawn")
        p.view_xy()
        p.reset_camera()
        p.render()


# ---------------------------------------------------------------------- self-test
def window_capture(w, plotter, path):
    """Grab the window and paint the 3D view into it (a widget grab leaves OpenGL views black)."""
    img = w.grab().toImage()
    arr = np.ascontiguousarray(plotter.screenshot(return_img=True, transparent_background=False)[:, :, :3])
    h, wd = arr.shape[:2]
    q = QtGui.QImage(arr.data, wd, h, 3 * wd, QtGui.QImage.Format_RGB888)
    widget = plotter.interactor
    pos = widget.mapTo(w, QtCore.QPoint(0, 0))
    painter = QtGui.QPainter(img)
    painter.drawImage(QtCore.QRect(pos, widget.size()), q)
    painter.end()
    img.save(path)


def wait_icp(app, w, limit=600):
    t0 = time.time()
    app.processEvents()
    while time.time() - t0 < limit and w.icp_worker is not None and (w.icp_worker.isRunning() or w.icp_again):
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()


def make_gif(folder, out, icp):
    """Headless: run DaTC on one Fantastic Breaks object folder and write the pairing GIF."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Studio()
    w.headless = True
    w.show()
    w.open_folder(folder)
    w.loader.wait()
    app.processEvents()
    if w.meshes is None:
        raise SystemExit(f"could not load {folder}: {w.last_error}")
    w.icp_chk.setChecked(icp)
    w.run()
    w.worker.wait()
    app.processEvents()
    if w.result is None:
        raise SystemExit(f"DaTC failed: {w.last_error}")
    if icp:
        wait_icp(app, w)
    path = w.save_gif(out)
    if not path:
        raise SystemExit(f"no GIF: {w.last_error}")
    print(w.result.verdict["text"])
    print(chr(10).join(s for s in w.log_box.toPlainText().splitlines() if s.startswith(("placement", "ICP"))))
    print(f"GIF written: {path}")
    w.close()


def selftest(folder, png):
    """Headless: load a Fantastic Breaks object, run DaTC through the worker, check, save screenshots."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Studio()
    w.headless = True
    w.show()
    w.open_folder(folder)
    w.loader.wait()
    app.processEvents()
    if w.meshes is None:
        selftest_failed(str(w.last_error))
    w.run()
    w.worker.wait()
    app.processEvents()
    if w.result is None:
        selftest_failed(str(w.last_error))
    res = w.result
    # the pair run must give the rims of the pool builder exactly (same code path, same settings)
    name = "_".join(Path(folder).parts[-2:])
    ref = DS.fb_object_rims(str(Path(folder).parents[1]), name)
    same = all(np.array_equal(r["alpha"], q["alpha"]) and np.array_equal(r["beta"], q["beta"])
               for r, q in zip(res.rims, ref))
    if not same:
        selftest_failed("rims of the app differ from datasets.fb_object_rims")
    w.save_png(png)
    window_capture(w, w.plotter, png.replace(".png", "_window.png"))
    w.chk["apart"].setChecked(True)
    app.processEvents()
    w.save_png(png.replace(".png", "_apart.png"))
    w.save_json(png.replace(".png", ".json"))
    if res.verdict["match"] is None:
        selftest_failed("pair tab: no verdict (reference/verdict.json missing?)")
    w.icp_chk.setChecked(True)
    wait_icp(app, w)
    if res.pose_icp is None:
        selftest_failed(f"pair tab: ICP did not run ({w.icp_note or w.last_error})")
    w.chk["placed"].setChecked(True)
    app.processEvents()
    w.save_png(png.replace(".png", "_icp.png"))
    w.chk["placed"].setChecked(False)
    gif = w.save_gif(png.replace(".png", ".gif"), n_frames=16)
    if not gif or not Path(gif).exists():
        selftest_failed(f"pair tab: GIF not written ({w.last_error})")
    lines = [f"SELFTEST OK {name}: n_A {len(res.rims[0]['alpha'])} n_B {len(res.rims[1]['alpha'])} "
             f"score {res.score:.4f} direction {res.direction} matched {len(res.path)}; "
             f"rims identical to datasets.fb_object_rims; -> {png}",
             f"  verdict: {res.verdict['text']}",
             "  " + next((s for s in w.log_box.toPlainText().splitlines() if s.startswith("ICP")), "ICP: no log line"),
             f"  GIF: {gif}"]
    if POOL_CODE.exists():
        w.code_src.setCurrentIndex(1)
        w.run()
        w.worker.wait()
        app.processEvents()
        lines.append(f"  with the pool code: score {w.result.score:.4f}")
    if (ROOT / "results" / "table1_reviewed" / "fb_scores.npz").exists() and (ROOT / "cache" / "fb_pool60.pkl").exists():
        w.tabs.setCurrentIndex(1)
        w.pool_box.setCurrentText("fb")
        w.arm_box.setCurrentText("S_sw")
        w.fb_root.setText(str(Path(folder).parents[1]))
        w.load_rankings()
        if w.query_table.rowCount() != 120:
            selftest_failed(f"retrieval tab: {w.query_table.rowCount()} rows, expected 120")
        lines.append(f"  retrieval tab: {w.rank_summary.text().split('. Select')[0]}")
        rows = w.ret["rows"]
        hit = next(k for k, q in enumerate(rows) if q[3] == q[1])     # first candidate is the mate
        miss = next(k for k, q in enumerate(rows) if q[3] != q[1])    # first candidate is another rim
        for tag, k, cand, icp in (("mate", hit, 0, False), ("mate_icp", hit, 0, True), ("other", miss, 0, False),
                                  ("other_mate", miss, 1, False)):
            w.ret_icp.blockSignals(True)
            w.ret_icp.setChecked(icp)
            w.ret_icp.blockSignals(False)
            w.cand_box.setCurrentIndex(cand)
            w.query_table.clearSelection()
            w.rres, err0, t0 = None, w.last_error, time.time()
            w.query_table.selectRow(k)
            while time.time() - t0 < 600:
                app.processEvents()
                busy = w.rworker is not None and w.rworker.isRunning()
                if not busy and not w.r_pending and (w.rres is not None or w.last_error != err0):
                    break
                time.sleep(0.05)
            app.processEvents()
            if w.rres is None:
                selftest_failed(f"retrieval pair {tag}: {w.last_error}")
            out = png.replace(".png", f"_retrieval_{tag}.png")
            window_capture(w, w.rplot, out)
            i, other = w.rres["key"][1:]
            if icp and w.rres["pose_icp"] is None:
                selftest_failed(f"retrieval ICP did not run: {w.rres['icp_note']}")
            lines.append(f"  retrieval {tag}: {w.ret['names'][i]} vs {w.ret['names'][other]}: "
                         f"{w.rres['verdict']['text'].split(':')[0]}, score {w.rres['features']['score']:.4f}, "
                         f"inliers {100 * w.rres['features']['inliers']:.1f}% -> {out}")
    print("\n".join(lines), flush=True)
    Path(png).with_name(Path(png).stem + "_selftest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    w.close()


def run_script_main(script, args):
    """Release only: run an experiment script with the bundled Python (`datc --run-script <script> ...`).

    The script's path is kept in DATC_SCRIPT so that the worker processes of its
    process pool (Windows starts them fresh from this executable) can load the
    script's functions before unpickling a task."""
    import runpy
    script = os.path.abspath(script)
    os.environ["DATC_SCRIPT"] = script
    for s in (sys.stdout, sys.stderr):
        if s is not None:
            s.reconfigure(line_buffering=True)
    sys.argv = [script] + list(args)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    import multiprocessing
    if FROZEN and os.environ.get("DATC_SCRIPT") and "--multiprocessing-fork" in sys.argv:
        import runpy   # a pool worker of a script run by --run-script: give __main__ the script's functions
        sys.modules["__main__"].__dict__.update(
            {k: v for k, v in runpy.run_path(os.environ["DATC_SCRIPT"], run_name="__mp_main__").items()
             if not k.startswith("__")})
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        run_script_main(sys.argv[2], sys.argv[3:])
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "--gif":
        make_gif(sys.argv[2], sys.argv[3], "--icp" in sys.argv[4:])
        sys.exit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"DaTC Studio {__version__}")
        sys.exit(0)
    if SELFTEST:
        try:
            selftest(sys.argv[2], sys.argv[3])
        except Exception:
            selftest_failed(traceback.format_exc())
        sys.exit(0)
    app = QtWidgets.QApplication(sys.argv)
    win = Studio()
    win.show()
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        win.open_folder(sys.argv[1])
    elif len(sys.argv) == 3:
        win.set_pair(sys.argv[1], sys.argv[2])
    sys.exit(app.exec_())
