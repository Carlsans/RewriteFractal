"""RewriteFractal GUI. Run from the repository root:  python -m gui.app"""
import copy
import sys
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from gui import bridge  # noqa: E402


class Job(QThread):
    log = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn, self._stop = fn, threading.Event()

    def should_stop(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self))
        except Exception:
            self.failed.emit(traceback.format_exc())


def to_pixmap(rgb, width=None):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
    return pix.scaledToWidth(width, Qt.SmoothTransformation) if width else pix


class ImageView(QLabel):
    def __init__(self, size=420):
        super().__init__("no image")
        self.setMinimumSize(size, size // 2)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888")
        self.rgb = None

    def show_rgb(self, rgb):
        self.rgb = rgb
        self.setPixmap(to_pixmap(rgb, max(min(self.width(), 900) - 4, 100)))


class LogBox(QPlainTextEdit):
    def __init__(self, height=120):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setMaximumHeight(height)

    def add(self, text):
        self.appendPlainText(text)


class Runner:
    def __init__(self, log, buttons=()):
        self.log, self.buttons, self.job = log, buttons, None

    def start(self, fn, on_done=None):
        if self.job is not None and self.job.isRunning():
            return
        self.job = Job(fn)
        self.job.log.connect(self.log.add)
        self.job.failed.connect(lambda tb: self.log.add(tb.strip().splitlines()[-1]))
        if on_done:
            self.job.finished_ok.connect(on_done)
        self.job.finished.connect(lambda: [b.setEnabled(True) for b in self.buttons])
        for b in self.buttons:
            b.setEnabled(False)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.stop()


class ExploreTab(QWidget):
    genome_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.genome, self.rendered = None, None
        self.start = QLineEdit(); self.start.setToolTip("Start string. Capitals are constants (Z and C are the iterate and the pixel), "
                                                         "lower case letters are functions applied to the next symbol, + - * / combine.")
        self.rules = QTableWidget(0, 2); self.rules.setHorizontalHeaderLabels(["replace", "with"])
        self.rules.horizontalHeader().setStretchLastSection(True)
        self.consts = QTableWidget(0, 2); self.consts.setHorizontalHeaderLabels(["constant", "value"])
        self.consts.horizontalHeader().setStretchLastSection(True)
        self.res = QSpinBox(); self.res.setRange(32, 4000); self.res.setValue(512); self.res.setSingleStep(128)
        self.cmap = QComboBox(); self.cmap.addItems(bridge.COLORMAPS)
        self.source = QComboBox(); self.source.addItems(["exit values (ranked)", "bailout iteration"])
        self.maxlen = QSpinBox(); self.maxlen.setRange(50, 200000); self.maxlen.setValue(20000)
        self.lines = QPlainTextEdit(); self.lines.setReadOnly(True); self.lines.setMaximumHeight(110)
        self.view = ImageView(); self.log = LogBox(80)
        self.opt_vocab = QSpinBox(); self.opt_vocab.setRange(1, 25); self.opt_vocab.setValue(4)
        self.fn_vocab = QSpinBox(); self.fn_vocab.setRange(1, 25); self.fn_vocab.setValue(3)
        b_rand = QPushButton("Random genome"); b_rand.clicked.connect(self.randomize)
        self.b_render = QPushButton("Render"); self.b_render.clicked.connect(self.render)
        b_open = QPushButton("Open .json"); b_open.clicked.connect(self.open_genome)
        b_save = QPushButton("Save .json"); b_save.clicked.connect(self.save_genome)
        b_png = QPushButton("Save PNG"); b_png.clicked.connect(self.save_png)
        self.runner = Runner(self.log, (self.b_render,))
        form = QFormLayout()
        form.addRow("Start string", self.start)
        form.addRow("Vocabulary (extra constants / functions)", self._pair(self.opt_vocab, self.fn_vocab))
        form.addRow("Resolution", self.res); form.addRow("Colormap", self.cmap); form.addRow("Colour source", self.source)
        form.addRow("Max line length", self.maxlen)
        left = QVBoxLayout(); left.addLayout(form)
        left.addWidget(QLabel("Rewrite rules")); left.addWidget(self.rules, 1)
        left.addWidget(QLabel("Constants (complex numbers and the fractal's centre/range are editable; "
                              "function constants show their type; use Mutate to change them)"))
        left.addWidget(self.consts, 1)
        row = QHBoxLayout()
        for b in (b_rand, self.b_render, b_open, b_save, b_png):
            row.addWidget(b)
        left.addLayout(row)
        right = QVBoxLayout(); right.addWidget(self.view, 1)
        right.addWidget(QLabel("Rewritten lines")); right.addWidget(self.lines)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addLayout(right, 1)
        outer = QVBoxLayout(self); outer.addLayout(top); outer.addWidget(self.log)
        self.randomize()

    @staticmethod
    def _pair(a, b):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(a); l.addWidget(b)
        return w

    def vocabulary(self):
        return bridge.vocabulary_for(self.opt_vocab.value(), self.fn_vocab.value())

    def randomize(self):
        gen = bridge.Generator(self.vocabulary())
        self.set_genome(gen.new_genome())

    def set_genome(self, genome):
        self.genome = genome
        self.start.setText(genome[0])
        self.rules.setRowCount(len(genome[1]))
        for r, (k, v) in enumerate(genome[1].items()):
            self.rules.setItem(r, 0, QTableWidgetItem(k)); self.rules.setItem(r, 1, QTableWidgetItem(v))
        keys = list(genome[2].keys())
        self.consts.setRowCount(len(keys))
        for r, k in enumerate(keys):
            item = QTableWidgetItem(k); item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.consts.setItem(r, 0, item)
            self.consts.setItem(r, 1, QTableWidgetItem(self._const_text(k, genome[2][k])))
        self.genome_changed.emit(genome)

    @staticmethod
    def _const_text(key, value):
        if key == "center":
            return f"{value[0]:.12g}, {value[1]:.12g}"
        if key == "range":
            return f"{float(value):.12g}"
        if isinstance(value, complex):
            return f"{value.real:.8g}{value.imag:+.8g}j"
        return bridge.describe_constant(value)

    def current_genome(self):
        genome = [self.start.text().strip(), OrderedDict(), copy.deepcopy(self.genome[2])]
        for r in range(self.rules.rowCount()):
            k, v = self.rules.item(r, 0), self.rules.item(r, 1)
            if k and k.text():
                genome[1][k.text()] = v.text() if v else ""
        for r in range(self.consts.rowCount()):
            key, text = self.consts.item(r, 0).text(), self.consts.item(r, 1).text().strip()
            if key == "center":
                a, b = text.split(","); genome[2][key] = [float(a), float(b)]
            elif key == "range":
                genome[2][key] = float(text)
            elif key.isupper():
                genome[2][key] = complex(text.replace(" ", "").replace("i", "j"))
        if not genome[0]:
            raise ValueError("empty start string")
        return genome

    def render(self):
        try:
            genome = self.current_genome()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid genome", str(exc)); return
        self.genome = genome
        size = (self.res.value(), self.res.value())
        cmap, src, maxlen = self.cmap.currentText(), "exit" if self.source.currentIndex() == 0 else "bailout", self.maxlen.value()

        def work(job):
            return bridge.render_genome(genome, size, cmap, src, maxlen)

        def done(result):
            rgb, rendered = result
            self.rendered = rendered
            self.view.show_rgb(rgb)
            self.lines.setPlainText("\n".join(rendered.rewriter.stringlist[:12]))
            note = bridge.SIEVE.getimageinformationsnote(rendered.fractal.bailout)
            self.log.add(f"rendered {size}, bailout PNG bytes/pixel: {note}")
            self.genome_changed.emit(genome)
        self.runner.start(work, done)

    def open_genome(self):
        path, _ = QFileDialog.getOpenFileName(self, "Genome", "", "Genome (*.json)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            self.set_genome(bridge.load_genome(path))
            self.render()
        except Exception as exc:
            self.log.add(f"could not open {path}: {exc}")

    def save_genome(self):
        try:
            genome = self.current_genome()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid genome", str(exc)); return
        path, _ = QFileDialog.getSaveFileName(self, "Save genome", "genome.json", "Genome (*.json)")
        if path:
            bridge.save_genome(genome, path)

    def save_png(self):
        if self.view.rgb is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "fractal.png", "PNG (*.png)")
        if path:
            from matplotlib import pyplot as plt
            plt.imsave(path, self.view.rgb)


class GalleryTab(QWidget):
    open_genome = Signal(str)

    def __init__(self):
        super().__init__()
        self.folder = QLineEdit(); self.limit = QSpinBox(); self.limit.setRange(10, 5000); self.limit.setValue(200)
        b = QPushButton("Choose folder"); b.clicked.connect(self.choose)
        self.list = QListWidget(); self.list.setViewMode(QListWidget.IconMode); self.list.setIconSize(QPixmap(128, 128).size())
        self.list.setResizeMode(QListWidget.Adjust); self.list.itemDoubleClicked.connect(lambda i: self.open_genome.emit(i.data(Qt.UserRole)))
        self.log = LogBox(60)
        top = QHBoxLayout(); top.addWidget(self.folder, 1); top.addWidget(QLabel("max")); top.addWidget(self.limit); top.addWidget(b)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(QLabel("Folders can hold thousands of files; only the first N are listed. "
                                                                        "Double-click a genome to open it in Explore. Thumbnails use the PNG with the same "
                                                                        "name when there is one, otherwise a 64x64 render."))
        lay.addWidget(self.list, 1); lay.addWidget(self.log)

    def choose(self):
        path = QFileDialog.getExistingDirectory(self, "Genome folder", self.folder.text() or str(bridge.REPO))
        if path:
            self.folder.setText(path); self.refresh()

    def refresh(self):
        import os
        self.list.clear()
        files = []
        with os.scandir(self.folder.text()) as it:
            for e in it:
                if e.name.endswith(".json"):
                    files.append(e.path)
                    if len(files) >= self.limit.value():
                        break
        for path in sorted(files):
            item = QListWidgetItem(Path(path).stem[-8:])
            item.setData(Qt.UserRole, path)
            pngs = list(Path(path).parent.glob(Path(path).stem + "*.png"))
            try:
                if pngs:
                    pix = QPixmap(str(pngs[0])).scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    rgb, _ = bridge.render_genome(bridge.load_genome(path), (64, 64))
                    pix = to_pixmap(rgb).scaled(128, 128)
                item.setIcon(QIcon(pix))
            except Exception as exc:
                item.setToolTip(str(exc)[:200])
            self.list.addItem(item)
        self.log.add(f"{len(files)} genomes listed")


class EvolveTab(QWidget):
    send_genome = Signal(object)

    def __init__(self):
        super().__init__()
        self.best, self.history = None, []
        self.pop = QSpinBox(); self.pop.setRange(4, 500); self.pop.setValue(30)
        self.gens = QSpinBox(); self.gens.setRange(1, 100000); self.gens.setValue(20)
        self.res = QSpinBox(); self.res.setRange(24, 512); self.res.setValue(64)
        self.cx = QDoubleSpinBox(); self.cx.setRange(0, 1); self.cx.setSingleStep(.05); self.cx.setValue(.7)
        self.mut = QDoubleSpinBox(); self.mut.setDecimals(4); self.mut.setRange(0, 1); self.mut.setSingleStep(.005); self.mut.setValue(.05)
        self.tourn = QSpinBox(); self.tourn.setRange(1, 50); self.tourn.setValue(3)
        self.opt = QSpinBox(); self.opt.setRange(1, 25); self.opt.setValue(4)
        self.fns = QSpinBox(); self.fns.setRange(1, 25); self.fns.setValue(3)
        self.metric = QComboBox()
        for k, v in bridge.METRICS.items():
            self.metric.addItem(v, k)
        self.stag = QCheckBox("Restart with a new population when the best has not improved"); self.stag.setChecked(True)
        self.stag_n = QSpinBox(); self.stag_n.setRange(2, 1000); self.stag_n.setValue(15)
        legacy = QGroupBox("Legacy / experimental options")
        self.parents = QCheckBox("Select next generation from offspring + parents (the loop the GA used). Off = DEAP eaSimple (offspring only)")
        self.parents.setChecked(True)
        self.timepen = QCheckBox("Subtract evaluation time from the score (was in the last runs; makes results non-reproducible)")
        self.seed_folder = QLineEdit(); self.seed_folder.setPlaceholderText("folder of saved genomes to start from (the disabled 'best pool' branch)")
        bs = QPushButton("..."); bs.clicked.connect(self.pick_seed)
        self.seed_mut = QSpinBox(); self.seed_mut.setRange(0, 100); self.seed_mut.setValue(5)
        self.pos_classes = QLineEdit(); self.pos_classes.setPlaceholderText("autoencoder positive classes, comma separated")
        self.neg_classes = QLineEdit(); self.neg_classes.setPlaceholderText("autoencoder negative classes")
        lf = QFormLayout(legacy)
        lf.addRow(self.parents); lf.addRow(self.timepen)
        lf.addRow("Seed from folder", self._row(self.seed_folder, bs)); lf.addRow("Mutations applied to each seed", self.seed_mut)
        lf.addRow(self.pos_classes); lf.addRow(self.neg_classes)
        form = QFormLayout()
        for label, w in (("Population", self.pop), ("Generations", self.gens), ("Evaluation resolution", self.res),
                         ("Crossing probability", self.cx), ("Mutation probability", self.mut), ("Tournament size", self.tourn),
                         ("Extra constants in vocabulary", self.opt), ("Functions in vocabulary", self.fns), ("Fitness", self.metric)):
            form.addRow(label, w)
        form.addRow(self.stag); form.addRow("Stagnation generations", self.stag_n)
        self.log = LogBox(180); self.view = ImageView(300)
        self.b_run = QPushButton("Run"); b_stop = QPushButton("Stop"); b_send = QPushButton("Send best to Explore")
        self.b_run.clicked.connect(self.run); b_stop.clicked.connect(lambda: self.runner.stop()); b_send.clicked.connect(self.send)
        self.runner = Runner(self.log, (self.b_run,))
        row = QHBoxLayout(); [row.addWidget(w) for w in (self.b_run, b_stop, b_send)]
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(legacy); left.addLayout(row)
        right = QVBoxLayout(); right.addWidget(QLabel("Best so far")); right.addWidget(self.view, 1); right.addWidget(self.log)
        lay = QHBoxLayout(self); lay.addLayout(left, 1); lay.addLayout(right, 1)

    @staticmethod
    def _row(a, b):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(a, 1); l.addWidget(b); return w

    def pick_seed(self):
        path = QFileDialog.getExistingDirectory(self, "Seed folder")
        if path:
            self.seed_folder.setText(path)

    def params(self):
        p = bridge.GAParams()
        p.pop_size, p.generations, p.resolution = self.pop.value(), self.gens.value(), self.res.value()
        p.crossing, p.mutation, p.tournament = self.cx.value(), self.mut.value(), self.tourn.value()
        p.optional_vocab, p.function_vocab, p.metric = self.opt.value(), self.fns.value(), self.metric.currentData()
        p.restart_on_stagnation, p.stagnation_generations = self.stag.isChecked(), self.stag_n.value()
        p.select_from_parents, p.time_penalty = self.parents.isChecked(), self.timepen.isChecked()
        p.seed_folder = self.seed_folder.text() or None
        p.seed_mutations = self.seed_mut.value()
        split = lambda s: [c.strip() for c in s.split(",") if c.strip()]
        p.class_targets = (split(self.pos_classes.text()), split(self.neg_classes.text()))
        return p

    def run(self):
        p = self.params()

        def on_best(genome, score):
            self.best = genome
            try:
                rgb, _ = bridge.render_genome(genome, (192, 192))
                self.view.show_rgb(rgb)
            except Exception:
                pass

        self.runner.start(lambda job: bridge.run_ga(p, job, self.runner.job.log.emit, on_best),
                          lambda r: self.log.add(f"finished, best score {r[1]:.4f}"))

    def send(self):
        if self.best is not None:
            self.send_genome.emit(self.best)


class LegacyTab(QWidget):
    send_genome = Signal(object)

    def __init__(self, explore):
        super().__init__()
        self.explore = explore
        self.log = LogBox(120)

        metrics = QGroupBox("Compare every fitness function on the genome in Explore")
        self.metric_out = QPlainTextEdit(); self.metric_out.setReadOnly(True); self.metric_out.setMaximumHeight(130)
        bm = QPushButton("Score current genome"); bm.clicked.connect(self.compare_metrics)
        ml = QVBoxLayout(metrics); ml.addWidget(bm); ml.addWidget(self.metric_out)

        miner = QGroupBox("Random miner (Miner.py): random genomes, keep those whose note passes the thresholds")
        self.t1 = QDoubleSpinBox(); self.t1.setRange(0, 5); self.t1.setSingleStep(.02); self.t1.setValue(.1)
        self.t2 = QDoubleSpinBox(); self.t2.setRange(0, 5); self.t2.setSingleStep(.02); self.t2.setValue(.2)
        b1 = QPushButton("Start mining"); b2 = QPushButton("Stop"); self.r1 = Runner(self.log, (b1,))
        b1.clicked.connect(self.start_mine); b2.clicked.connect(lambda: self.r1.stop())
        mf = QFormLayout(miner); mf.addRow("Keep above", self.t1); mf.addRow("Mark as excellent above", self.t2)
        row = QHBoxLayout(); row.addWidget(b1); row.addWidget(b2); mf.addRow(row)
        self.found = QListWidget(); self.found.itemDoubleClicked.connect(lambda i: self.send_genome.emit(i.data(Qt.UserRole)))
        self.found.setMaximumHeight(90); mf.addRow(QLabel("Found (double-click to open in Explore):")); mf.addRow(self.found)

        var = QGroupBox("Taylor-series variations (draw_individual_with_variations)")
        self.key = QLineEdit(); self.key.setPlaceholderText("function constant name, e.g. a")
        bv = QPushButton("Render variations"); bv.clicked.connect(self.variations)
        self.grid = ImageView(300); self.r3 = Runner(self.log, (bv,))
        vl = QVBoxLayout(var); vl.addWidget(self.key); vl.addWidget(bv); vl.addWidget(self.grid)

        study = QGroupBox("Hyperparameter study (StudyManager + Optuna; in-memory, or set GAOPTUNA_STORAGE)")
        self.trials = QSpinBox(); self.trials.setRange(1, 1000); self.trials.setValue(5)
        self.sgens = QSpinBox(); self.sgens.setRange(1, 200); self.sgens.setValue(5)
        bs = QPushButton("Run study"); bs.clicked.connect(self.run_study); self.r4 = Runner(self.log, (bs,))
        sf = QFormLayout(study); sf.addRow("Trials", self.trials); sf.addRow("Generations per trial", self.sgens); sf.addRow(bs)

        up = QGroupBox("Batch upscale (FractalUpscaler): re-render every genome of a folder")
        self.up_in = QLineEdit(); self.up_out = QLineEdit(); self.up_size = QSpinBox(); self.up_size.setRange(256, 8000); self.up_size.setValue(2000)
        bu = QPushButton("Upscale"); bu.clicked.connect(self.upscale); self.r5 = Runner(self.log, (bu,))
        bi = QPushButton("Input..."); bi.clicked.connect(lambda: self._pick(self.up_in))
        bo = QPushButton("Output..."); bo.clicked.connect(lambda: self._pick(self.up_out))
        uf = QFormLayout(up); uf.addRow(bi, self.up_in); uf.addRow(bo, self.up_out); uf.addRow("Size", self.up_size); uf.addRow(bu)

        left = QVBoxLayout(); left.addWidget(metrics); left.addWidget(miner); left.addWidget(study)
        right = QVBoxLayout(); right.addWidget(var, 1); right.addWidget(up)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addLayout(right, 1)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)

    def _pick(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Folder")
        if path:
            edit.setText(path)

    def compare_metrics(self):
        try:
            genome = self.explore.current_genome()
        except Exception as exc:
            self.metric_out.setPlainText(str(exc)); return
        lines = []
        for key, label in bridge.METRICS.items():
            try:
                score = bridge.score_genome(genome, key, (64, 64))
                lines.append(f"{key:20s} {'invalid / degenerate' if score is None else format(score, '.4f')}")
            except Exception as exc:
                lines.append(f"{key:20s} unavailable: {str(exc).splitlines()[0][:70] if str(exc) else type(exc).__name__}")
        self.metric_out.setPlainText("\n".join(lines))

    def start_mine(self):
        def found(genome, note, excellent):
            self.r1.job.log.emit(f"kept {note:.3f}{' (excellent)' if excellent else ''}")
            item = QListWidgetItem(f"{note:.3f}  {genome[0]}"); item.setData(Qt.UserRole, genome)
            self._pending.append(item)
        self._pending = []
        timer_job = lambda job: bridge.mine((self.t1.value(), self.t2.value()), job, job.log.emit, found)
        self.r1.start(timer_job, lambda _: [self.found.addItem(i) for i in self._pending])

    def variations(self):
        try:
            genome = self.explore.current_genome()
        except Exception as exc:
            self.log.add(str(exc)); return
        key = self.key.text().strip()

        def work(job):
            tiles, per_row = bridge.taylor_variations(genome, key, (96, 96))
            rows = [np.concatenate(tiles[i:i + per_row], axis=1) for i in range(0, len(tiles), per_row)]
            return np.concatenate(rows, axis=0)
        self.r3.start(work, self.grid.show_rgb)

    def run_study(self):
        self.r4.start(lambda job: bridge.run_optuna_study(self.trials.value(), self.sgens.value(), job.log.emit, job),
                      lambda r: self.log.add(f"best {r[1]:.4f}: {r[0]}"))

    def upscale(self):
        if self.up_in.text() and self.up_out.text():
            s = self.up_size.value()
            self.r5.start(lambda job: bridge.upscale_folder(self.up_in.text(), self.up_out.text(), (s, s), job=job, log=job.log.emit))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RewriteFractal")
        self.explore, self.gallery, self.evolve = ExploreTab(), GalleryTab(), EvolveTab()
        self.legacy = LegacyTab(self.explore)
        tabs = QTabWidget()
        for w, name in ((self.explore, "Explore"), (self.gallery, "Gallery"), (self.evolve, "Evolve"),
                        (self.legacy, "Legacy / experimental")):
            tabs.addTab(w, name)
        self.setCentralWidget(tabs)

        def into_explore(genome):
            self.explore.set_genome(genome); tabs.setCurrentWidget(self.explore); self.explore.render()
        self.gallery.open_genome.connect(lambda p: (tabs.setCurrentWidget(self.explore), self.explore.load_path(p)))
        self.evolve.send_genome.connect(into_explore)
        self.legacy.send_genome.connect(into_explore)
        self.resize(1250, 850)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
