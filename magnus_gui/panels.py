"""As abas do MAGNUS. Nesta primeira leva: Modelo (funcional, com preview ao
vivo do motor) e placeholders informativos para as demais."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
                               QLabel, QPushButton, QComboBox, QSizePolicy, QGridLayout,
                               QCheckBox, QSpinBox, QProgressBar, QFileDialog,
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

from . import theme as T
from . import plots
from .widgets import Card, ParamRow
from .backend import (ModelState, Spot, run_spectral_grid, folded_and_spectrum,
                      default_table, read_events_meta, build_fit_request)
from .fitrunner import FitRunner


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setFixedWidth(360)
    return sa


class ModeloPanel(QWidget):
    """Monta uma estrela e mostra pulso + espectro, ao vivo."""
    def __init__(self, status_cb=None):
        super().__init__()
        self.status_cb = status_cb
        self.state = ModelState()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._recompute)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- coluna de parâmetros ----
        col = QWidget()
        cl = QVBoxLayout(col)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(11)
        self.rows = {}

        star = Card("Estrela", T.CYAN)
        self.rows["mass"] = ParamRow("Massa", self.state.mass, unit="M☉", lo=0.5, hi=2.6, step=0.05, on_change=self._changed)
        self.rows["radius"] = ParamRow("Raio", self.state.radius, unit="km", lo=8, hi=18, step=0.2, decimals=1, on_change=self._changed)
        self.rows["distance"] = ParamRow("Distância", self.state.distance, unit="kpc", lo=0.05, hi=5, step=0.05, decimals=2, on_change=self._changed)
        for r in ("mass", "radius", "distance"): star.add(self.rows[r])
        cl.addWidget(star)

        geo = Card("Geometria", T.VIOLET)
        self.rows["inclination"] = ParamRow("Inclinação i", self.state.inclination, unit="°", lo=0, hi=180, step=1, decimals=0, on_change=self._changed)
        self.rows["magnetic_colatitude"] = ParamRow("Colatitude mag.", self.state.magnetic_colatitude, unit="°", lo=0, hi=180, step=1, decimals=0, on_change=self._changed)
        self.rows["magnetic_azimuth"] = ParamRow("Azimute mag.", self.state.magnetic_azimuth, unit="°", lo=-180, hi=180, step=5, decimals=0, on_change=self._changed)
        self.rows["period"] = ParamRow("Período", self.state.period, unit="s", lo=0.01, hi=1e4, step=0.001, decimals=4, on_change=self._changed)
        for r in ("inclination", "magnetic_colatitude", "magnetic_azimuth", "period"): geo.add(self.rows[r])
        cl.addWidget(geo)

        atm = Card("Atmosfera T(θ)", T.MAGENTA)
        tbl_lbl = QLabel("magnus_campoB · MAGNUSI2" if default_table() else "corpo negro (sem tabela)")
        tbl_lbl.setObjectName("rowSub")
        atm.add(tbl_lbl)
        self.rows["log_field"] = ParamRow("Campo lg B", self.state.log_field, unit="G", lo=12.5, hi=13.8, step=0.05, decimals=2, on_change=self._changed)
        self.rows["base_kt_kev"] = ParamRow("T do polo", self.state.base_kt_kev, unit="keV", lo=0.02, hi=0.3, step=0.005, decimals=3, on_change=self._changed)
        self.rows["temperature_peaking"] = ParamRow("Concentração a", self.state.temperature_peaking, lo=0, hi=4, step=0.1, decimals=2, on_change=self._changed)
        self.rows["atmosphere_fraction"] = ParamRow("Espessura f", self.state.atmosphere_fraction, lo=0, hi=1, step=0.05, decimals=2, on_change=self._changed)
        for r in ("log_field", "base_kt_kev", "temperature_peaking", "atmosphere_fraction"): atm.add(self.rows[r])
        cl.addWidget(atm)

        cl.addStretch(1)
        root.addWidget(_scroll(col))

        # ---- área de gráficos ----
        stage = QWidget()
        sl = QVBoxLayout(stage)
        sl.setContentsMargins(16, 16, 16, 16)
        sl.setSpacing(12)
        head = QLabel("Modelo — pré-visualização ao vivo")
        head.setStyleSheet(f"color:{T.TEXT};font-size:15px;font-weight:600;background:transparent")
        sl.addWidget(head)
        grid = QHBoxLayout()
        grid.setSpacing(12)
        self.cv_pulse = plots.Canvas()
        self.cv_spec = plots.Canvas()
        for cv in (self.cv_pulse, self.cv_spec):
            cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            grid.addWidget(cv)
        sl.addLayout(grid, 1)
        root.addWidget(stage, 1)

        QTimer.singleShot(60, self._recompute)

    def _changed(self):
        self._debounce.start()

    def _pull(self):
        s = self.state
        s.mass = self.rows["mass"].value()
        s.radius = self.rows["radius"].value()
        s.distance = self.rows["distance"].value()
        s.inclination = self.rows["inclination"].value()
        s.magnetic_colatitude = self.rows["magnetic_colatitude"].value()
        s.magnetic_azimuth = self.rows["magnetic_azimuth"].value()
        s.period = self.rows["period"].value()
        s.log_field = self.rows["log_field"].value()
        s.base_kt_kev = self.rows["base_kt_kev"].value()
        s.temperature_peaking = self.rows["temperature_peaking"].value()
        s.atmosphere_fraction = self.rows["atmosphere_fraction"].value()

    def _recompute(self):
        self._pull()
        if self.status_cb:
            self.status_cb("calculando modelo…", live=True)
        try:
            grid = run_spectral_grid(self.state, timeout=60)
            phase, lc, energy, sp = folded_and_spectrum(grid)
            plots.draw_pulse(self.cv_pulse, phase, lc)
            plots.draw_spectrum(self.cv_spec, energy, sp)
            fp = (lc.max() - lc.min()) / (lc.max() + lc.min()) * 100 if lc.max() > 0 else 0
            if self.status_cb:
                self.status_cb(f"modelo pronto · pulso {fp:.0f}% · "
                               f"M={self.state.mass:.2f} R={self.state.radius:.1f} f={self.state.atmosphere_fraction:.2f}")
        except Exception as e:
            plots.draw_message(self.cv_pulse, f"erro no motor:\n{e}")
            plots.draw_message(self.cv_spec, "—")
            if self.status_cb:
                self.status_cb(f"erro: {e}")


class AjustePanel(QWidget):
    """Ajuste MCMC de dados reais: dados, o que ajustar, e o motor em processo
    separado com progresso ao vivo."""
    def __init__(self, modelo: "ModeloPanel", status_cb=None, on_result=None):
        super().__init__()
        self.modelo = modelo
        self.status_cb = status_cb
        self.on_result = on_result
        self.events_path = None
        self.background_path = None
        self.runner = None
        self.result = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        col = QWidget()
        cl = QVBoxLayout(col); cl.setContentsMargins(14, 14, 14, 14); cl.setSpacing(11)

        data = Card("Dados", T.CYAN)
        self.btn_ev = QPushButton("Escolher eventos…")
        self.btn_ev.clicked.connect(self._pick_events)
        self.lbl_ev = QLabel("nenhum arquivo"); self.lbl_ev.setObjectName("rowSub"); self.lbl_ev.setWordWrap(True)
        self.btn_bg = QPushButton("Escolher fundo…")
        self.btn_bg.clicked.connect(self._pick_bg)
        self.lbl_bg = QLabel("opcional"); self.lbl_bg.setObjectName("rowSub"); self.lbl_bg.setWordWrap(True)
        for w in (self.btn_ev, self.lbl_ev, self.btn_bg, self.lbl_bg): data.add(w)
        cl.addWidget(data)

        fit = Card("O que ajustar", T.VIOLET)
        self.flags = {}
        for key, label, default in [
            ("fitMR", "Massa e raio", False),
            ("fitMagneticField", "Campo lg B", True),
            ("fitBaseTemperature", "T do polo", True),
            ("fitTemperaturePeaking", "Concentração a", True),
            ("fitAtmosphereFraction", "Espessura f", True),
            ("fitMagneticColatitude", "Colatitude mag.", True),
            ("fitMagneticAzimuth", "Azimute mag.", True),
            ("fitLine", "Linha de absorção", True),
        ]:
            cb = QCheckBox(label); cb.setChecked(default)
            self.flags[key] = cb; fit.add(cb)
        cl.addWidget(fit)

        mc = Card("MCMC", T.AMBER)
        self.mc = {}
        for key, label, val, lo, hi in [
            ("walkers", "Caminhantes", 40, 8, 200),
            ("iterations", "Iterações", 400, 20, 5000),
            ("burnIn", "Warm-up", 150, 0, 3000),
            ("workers", "Workers", 4, 1, 24),
        ]:
            r = ParamRow(label, val, integer=True, lo=lo, hi=hi)
            self.mc[key] = r; mc.add(r)
        cl.addWidget(mc)

        self.btn_run = QPushButton("Rodar ajuste"); self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("Cancelar"); self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        brow = QHBoxLayout(); brow.addWidget(self.btn_run, 1); brow.addWidget(self.btn_cancel)
        cl.addLayout(brow)
        cl.addStretch(1)
        root.addWidget(_scroll(col))

        stage = QWidget()
        sl = QVBoxLayout(stage); sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(12)
        head = QLabel("Ajuste MCMC"); head.setStyleSheet(f"color:{T.TEXT};font-size:15px;font-weight:600;background:transparent")
        sl.addWidget(head)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        sl.addWidget(self.prog)
        self.live = QLabel("configure os dados e rode o ajuste")
        self.live.setStyleSheet(f"color:{T.MUTED};font-family:'JetBrains Mono',monospace;font-size:12px;background:transparent")
        sl.addWidget(self.live)
        self.cv_pulse = plots.Canvas()
        sl.addWidget(self.cv_pulse, 1)
        plots.draw_message(self.cv_pulse, "o pulso ajustado aparece aqui ao terminar")
        root.addWidget(stage, 1)

    def _pick_events(self):
        p, _ = QFileDialog.getOpenFileName(self, "Lista de eventos", str(ROOT_EXPLORA()), "CSV (*.csv);;Todos (*)")
        if p:
            self.events_path = p
            meta = read_events_meta(p)
            n = meta.get("detected_events", "?")
            self.lbl_ev.setText(f"{Path(p).name}\n{meta.get('instrument','?')} · {n} eventos")

    def _pick_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Fundo", str(ROOT_EXPLORA()), "CSV (*.csv);;Todos (*)")
        if p:
            self.background_path = p
            self.lbl_bg.setText(Path(p).name)

    def _run(self):
        if not self.events_path:
            self.live.setText("escolha uma lista de eventos primeiro"); return
        self.modelo._pull()
        meta = read_events_meta(self.events_path)
        flags = {k: cb.isChecked() for k, cb in self.flags.items()}
        mcmc = {k: r.value() for k, r in self.mc.items()}
        req = build_fit_request(self.modelo.state, flags, meta, mcmc)
        self.runner = FitRunner(python_exe=_python())
        self.runner.progress.connect(self._on_prog)
        self.runner.finished.connect(self._on_done)
        self.runner.failed.connect(self._on_fail)
        self.runner.log.connect(lambda m: self.live.setText(m))
        self.prog.setValue(0)
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.runner.start(req, self.events_path, self.background_path)
        if self.status_cb: self.status_cb("ajuste rodando…", live=True)

    def _cancel(self):
        if self.runner: self.runner.cancel()
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)

    def _on_prog(self, step, total, rhat):
        pct = int(100 * step / max(total, 1))
        self.prog.setValue(min(pct, 99))
        rh = f"{rhat:.2f}" if rhat == rhat else "—"
        self.live.setText(f"passo {step}/{total} · rhat≈{rh}")
        if self.status_cb: self.status_cb(f"ajuste {step}/{total} · rhat≈{rh}", live=True)

    def _on_done(self, result):
        self.result = result
        self.prog.setValue(100)
        best = result.get("bestLogLikelihood", float("nan"))
        self.live.setText(f"ajuste concluído · lnL {best:,.0f}")
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        try:
            import numpy as np
            nph, nen = result["phaseBins"], result["energyBins"]
            obs = np.array(result["observed"], float).reshape(nph, nen).sum(1)
            exp = np.array(result["expected"], float).reshape(nph, nen).sum(1)
            ax = self.cv_pulse.ax; self.cv_pulse.clear()
            ph = np.concatenate([(np.arange(nph)+0.5)/nph, (np.arange(nph)+0.5)/nph+1])
            ax.plot(ph, np.concatenate([exp, exp]), color=T.CYAN, lw=2, label="modelo")
            ax.errorbar(ph, np.concatenate([obs, obs]), yerr=np.sqrt(np.concatenate([obs, obs])),
                        fmt="o", ms=3, color=T.TEXT, label="dados")
            ax.set_title("Pulso ajustado", fontsize=10); ax.set_xlabel("fase"); ax.legend(fontsize=8)
            self.cv_pulse.fig.tight_layout(); self.cv_pulse.draw()
        except Exception:
            pass
        if self.status_cb: self.status_cb(f"ajuste concluído · lnL {best:,.0f}")
        if self.on_result: self.on_result(result)

    def _on_fail(self, msg):
        self.live.setText("erro: " + msg)
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        if self.status_cb: self.status_cb("erro: " + msg)


class AtmosferaPanel(QWidget):
    """Inspeciona a tabela de intensidade: eixos, feixe I(μ), endurecimento e o
    mapa de temperatura T(θ) na esfera."""
    def __init__(self, status_cb=None):
        super().__init__()
        self.status_cb = status_cb
        self.tab = None
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        col = QWidget(); cl = QVBoxLayout(col); cl.setContentsMargins(14, 14, 14, 14); cl.setSpacing(11)

        info = Card("Tabela", T.VIOLET)
        self.lbl = QLabel("carregando…"); self.lbl.setObjectName("rowSub"); self.lbl.setWordWrap(True)
        info.add(self.lbl)
        b = QPushButton("Abrir outra tabela…"); b.clicked.connect(self._open); info.add(b)
        cl.addWidget(info)

        cut = Card("Corte", T.CYAN)
        self.cb_b = QComboBox(); self.cb_t = QComboBox(); self.cb_th = QComboBox()
        for lab, cb in (("lg B", self.cb_b), ("lg T", self.cb_t), ("θ_B", self.cb_th)):
            r = QHBoxLayout(); q = QLabel(lab); q.setObjectName("rowLabel"); r.addWidget(q, 1); r.addWidget(cb)
            cut.add_row(r); cb.currentIndexChanged.connect(self._draw)
        cl.addWidget(cut)

        mapc = Card("Mapa T(θ)", T.MAGENTA)
        self.a = ParamRow("Concentração a", 0.5, lo=0, hi=4, step=0.1, decimals=2, on_change=self._draw_map)
        mapc.add(self.a)
        cl.addWidget(mapc)
        cl.addStretch(1)
        root.addWidget(_scroll(col))

        stage = QWidget(); sl = QVBoxLayout(stage); sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(12)
        row = QHBoxLayout(); row.setSpacing(12)
        self.cv_beam = plots.Canvas(); self.cv_ratio = plots.Canvas()
        for cv in (self.cv_beam, self.cv_ratio):
            cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); row.addWidget(cv)
        sl.addLayout(row, 1)
        self.cv_map = plots.Canvas(width=8, height=2.6)
        self.cv_map.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cv_map.setMinimumHeight(210)
        sl.addWidget(self.cv_map)
        root.addWidget(stage, 1)

        QTimer.singleShot(80, self._load_default)

    def _load_default(self):
        from .backend import default_table
        p = default_table()
        if p:
            self._load(p)
        else:
            self.lbl.setText("sem tabela em tabelas/ ou build/")
            plots.draw_message(self.cv_beam, "sem tabela de atmosfera")
            plots.draw_message(self.cv_ratio, "—")
        self._draw_map()

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Tabela", str(ROOT_EXPLORA().parent / "tabelas"), "MAGNUS (*.magnus);;Todos (*)")
        if p: self._load(p)

    def _load(self, path):
        from .backend import read_atmosphere_table
        try:
            self.tab = read_atmosphere_table(path)
        except Exception as e:
            self.lbl.setText(f"erro: {e}"); return
        import numpy as np
        t = self.tab
        self.lbl.setText(f"{Path(path).name} · {t['magic']}\n"
                         f"lg B: {len(t['log_b'])}  lg T: {len(t['log_t'])}  θ_B: {len(t['theta_b'])}  "
                         f"μ: {len(t['mu'])}  lg E: {len(t['log_e'])}")
        for cb, key, fmt in ((self.cb_b, "log_b", "{:.2f}"), (self.cb_t, "log_t", "{:.2f}"), (self.cb_th, "theta_b", "{:.0f}°")):
            cb.blockSignals(True); cb.clear()
            for v in t[key]: cb.addItem(fmt.format(v))
            cb.setCurrentIndex(len(t[key]) // 2); cb.blockSignals(False)
        self._draw()

    def _draw(self):
        if not self.tab: return
        import numpy as np
        t = self.tab; lw = t["log_w"]
        ib = max(0, self.cb_b.currentIndex()); it = max(0, self.cb_t.currentIndex()); ith = max(0, self.cb_th.currentIndex())
        mu = t["mu"]; E = 10 ** t["log_e"]
        # feixe I(mu) em 3 energias (razão para corpo negro)
        ax = self.cv_beam.ax; self.cv_beam.clear()
        eidx = [int(len(E) * f) for f in (0.15, 0.5, 0.85)]
        cols = [T.CYAN, T.VIOLET, T.MAGENTA]
        for k, ie in enumerate(eidx):
            w = 10 ** lw[ib, it, 0, ith, :, ie]
            ax.plot(mu, w, color=cols[k], lw=2, label=f"{E[ie]:.2f} keV")
        ax.set_title("Feixe I(μ) / corpo negro", fontsize=10); ax.set_xlabel("μ = cos θ"); ax.legend(fontsize=8)
        self.cv_beam.fig.tight_layout(); self.cv_beam.draw()
        # endurecimento: média em mu vs energia
        ax2 = self.cv_ratio.ax; self.cv_ratio.clear()
        hard = (10 ** lw[ib, it, 0, ith, :, :]).mean(axis=0)
        ax2.plot(E, hard, color=T.AMBER, lw=2)
        ax2.set_xscale("log"); ax2.set_title("Endurecimento (média em μ)", fontsize=10)
        ax2.set_xlabel("energia (keV)"); ax2.set_ylabel("I/B_E")
        self.cv_ratio.fig.tight_layout(); self.cv_ratio.draw()
        if self.status_cb: self.status_cb(f"tabela: corte lg B={t['log_b'][ib]:.2f}, lg T={t['log_t'][it]:.2f}, θ_B={t['theta_b'][ith]:.0f}°")

    def _draw_map(self):
        import numpy as np
        a = self.a.value()
        ax = self.cv_map.ax; self.cv_map.clear()
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        tilt = np.deg2rad(-20)
        axv = np.array([np.sin(tilt), -np.cos(tilt)])
        n = 160
        xs = np.linspace(-1, 1, n); ys = np.linspace(-0.5, 0.5, n // 2)
        pts_x, pts_y, cvals = [], [], []
        for yy in ys:
            for xx in xs:
                if xx * xx + (yy * 2) ** 2 > 1: continue
                c = abs(xx * axv[0] + (yy * 2) * axv[1]); c2 = c * c
                T4 = c2 / (c2 + a * (1 - c2)) + 0.3 ** 4
                pts_x.append(xx); pts_y.append(yy); cvals.append(T4 ** 0.25)
        sc = ax.scatter(pts_x, pts_y, c=cvals, cmap="magma", s=6, marker="s")
        ax.set_title("Temperatura T(θ) na esfera — polos quentes, equador frio", fontsize=10, color=T.TEXT)
        self.cv_map.fig.tight_layout(); self.cv_map.draw()


class ResultadosPanel(QWidget):
    """Lê um ajuste terminado: tabela de parâmetros + pulso e espectro vs dados."""
    def __init__(self, status_cb=None):
        super().__init__()
        self.status_cb = status_cb
        self.result = None
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        col = QWidget(); cl = QVBoxLayout(col); cl.setContentsMargins(14, 14, 14, 14); cl.setSpacing(11)
        card = Card("Parâmetros", T.CYAN)
        self.lbl_best = QLabel("nenhum ajuste carregado"); self.lbl_best.setObjectName("rowSub")
        card.add(self.lbl_best)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["parâmetro", "mediana", "1σ", "rhat"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(320)
        card.add(self.table)
        cl.addWidget(card)
        btns = Card("Exportar", T.AMBER)
        b_load = QPushButton("Carregar resultado…"); b_load.clicked.connect(self._load)
        b_corner = QPushButton("Gerar cornerplot (PNG)"); b_corner.clicked.connect(self._corner)
        b_csv = QPushButton("Exportar tabela (CSV)"); b_csv.clicked.connect(self._csv)
        for b in (b_load, b_corner, b_csv): btns.add(b)
        cl.addWidget(btns); cl.addStretch(1)
        root.addWidget(_scroll(col))

        stage = QWidget(); sl = QVBoxLayout(stage); sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(12)
        head = QLabel("Resultados do ajuste"); head.setStyleSheet(f"color:{T.TEXT};font-size:15px;font-weight:600;background:transparent")
        sl.addWidget(head)
        row = QHBoxLayout(); row.setSpacing(12)
        self.cv_pulse = plots.Canvas(); self.cv_spec = plots.Canvas()
        for cv in (self.cv_pulse, self.cv_spec):
            cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); row.addWidget(cv)
        sl.addLayout(row, 1)
        root.addWidget(stage, 1)
        for cv, m in ((self.cv_pulse, "pulso ajustado"), (self.cv_spec, "espectro ajustado")):
            plots.draw_message(cv, m + " aparece ao carregar um ajuste")

    def set_result(self, result: dict):
        self.result = result
        best = result.get("bestLogLikelihood", float("nan"))
        self.lbl_best.setText(f"lnL {best:,.0f}  ·  {result.get('dimensions','?')} parâmetros")
        summ = result.get("summary", [])
        free = [s for s in summ if abs(s.get("upper", 0) - s.get("lower", 0)) > 1e-9 or True]
        def num(v, default=float("nan")):
            return v if isinstance(v, (int, float)) else default
        self.table.setRowCount(len(free))
        for i, s in enumerate(free):
            med = num(s.get("median"))
            up, lo_ = num(s.get("upper"), med), num(s.get("lower"), med)
            sig = 0.5 * (up - lo_)
            rhat = num(s.get("rhat"))
            self.table.setItem(i, 0, QTableWidgetItem(str(s.get("name", ""))))
            self.table.setItem(i, 1, QTableWidgetItem(f"{med:.4g}" if med == med else "—"))
            self.table.setItem(i, 2, QTableWidgetItem(f"±{sig:.2g}" if sig == sig else "—"))
            it = QTableWidgetItem(f"{rhat:.2f}" if rhat == rhat else "—")
            if rhat == rhat and rhat > 1.6:
                it.setForeground(QColor(T.AMBER))
            elif rhat == rhat:
                it.setForeground(QColor(T.CYAN))
            self.table.setItem(i, 3, it)
        self._draw()
        if self.status_cb: self.status_cb(f"resultado carregado · lnL {best:,.0f}")

    def _draw(self):
        try:
            import numpy as np
            r = self.result; nph, nen = r["phaseBins"], r["energyBins"]
            obs = np.array(r["observed"], float).reshape(nph, nen)
            exp = np.array(r["expected"], float).reshape(nph, nen)
            lo, le = obs.sum(1), exp.sum(1); so, se = obs.sum(0), exp.sum(0)
            ph = (np.arange(nph) + 0.5) / nph
            ph2 = np.concatenate([ph, ph + 1])
            ax = self.cv_pulse.ax; self.cv_pulse.clear()
            ax.errorbar(ph2, np.concatenate([lo, lo]), yerr=np.sqrt(np.concatenate([lo, lo])),
                        fmt="o", ms=3, color=T.TEXT, label="dados")
            ax.plot(ph2, np.concatenate([le, le]), color=T.CYAN, lw=2, label="modelo")
            ax.set_title("Pulso", fontsize=10); ax.set_xlabel("fase"); ax.legend(fontsize=8)
            self.cv_pulse.fig.tight_layout(); self.cv_pulse.draw()
            en = np.linspace(r.get("energyMin", 0.15), r.get("energyMax", 1.2), nen)
            ax2 = self.cv_spec.ax; self.cv_spec.clear()
            ax2.errorbar(en, so, yerr=np.sqrt(np.maximum(so, 1)), fmt="o", ms=3, color=T.TEXT, label="dados")
            ax2.plot(en, se, color=T.CYAN, lw=2, label="modelo")
            ax2.set_yscale("log"); ax2.set_title("Espectro", fontsize=10); ax2.set_xlabel("keV"); ax2.legend(fontsize=8)
            self.cv_spec.fig.tight_layout(); self.cv_spec.draw()
        except Exception as e:
            plots.draw_message(self.cv_pulse, f"erro ao desenhar: {e}")

    def _load(self):
        import json
        p, _ = QFileDialog.getOpenFileName(self, "resultado.json", str(ROOT_EXPLORA()), "JSON (*.json)")
        if p:
            try:
                self.set_result(json.loads(Path(p).read_text()))
            except Exception as e:
                if self.status_cb: self.status_cb(f"erro ao carregar: {e}")

    def _corner(self):
        if not self.result:
            return
        import json, tempfile, subprocess
        d = Path(tempfile.mkdtemp(prefix="magnus_corner_"))
        (d / "r.json").write_text(json.dumps(self.result))
        png = d / "corner.png"
        from .backend import ROOT
        try:
            subprocess.run([_python(), str(ROOT / "scripts" / "cornerplot.py"),
                            str(d / "r.json"), str(png)], check=True, capture_output=True, text=True)
            import os
            os.system(f'xdg-open "{png}" 2>/dev/null &')
            if self.status_cb: self.status_cb(f"cornerplot salvo: {png}")
        except Exception as e:
            if self.status_cb: self.status_cb(f"corner falhou: {e}")

    def _csv(self):
        if not self.result:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Salvar tabela", "parametros.csv", "CSV (*.csv)")
        if not p:
            return
        lines = ["parametro,mediana,sigma,rhat"]
        for s in self.result.get("summary", []):
            med = s.get("median", ""); sig = 0.5 * (s.get("upper", 0) - s.get("lower", 0))
            lines.append(f"{s.get('name','')},{med},{sig},{s.get('rhat','')}")
        Path(p).write_text("\n".join(lines))
        if self.status_cb: self.status_cb(f"tabela salva: {p}")


def _python():
    import sys
    return sys.executable


def ROOT_EXPLORA():
    from .backend import ROOT
    d = ROOT / "exploracoes"
    return d if d.is_dir() else ROOT


class DadosPanel(QWidget):
    """Co-adiciona observações: dobra cada uma, alinha pela forma, soma e grava
    um .npz que a aba Ajuste pode usar (preparedDataNpz)."""
    def __init__(self, status_cb=None):
        super().__init__()
        self.status_cb = status_cb
        self.obs = []   # lista de (eventos, fundo)
        self.npz = None
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        col = QWidget(); cl = QVBoxLayout(col); cl.setContentsMargins(14, 14, 14, 14); cl.setSpacing(11)
        card = Card("Observações", T.CYAN)
        self.lst = QLabel("nenhuma observação"); self.lst.setObjectName("rowSub"); self.lst.setWordWrap(True)
        add = QPushButton("Adicionar observação…"); add.clicked.connect(self._add)
        clr = QPushButton("Limpar"); clr.clicked.connect(self._clear)
        card.add(self.lst); card.add(add); card.add(clr)
        cl.addWidget(card)
        run = Card("Co-adição", T.MAGENTA)
        self.btn = QPushButton("Co-adicionar e salvar .npz"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._coadd); run.add(self.btn)
        self.out = QLabel(""); self.out.setObjectName("rowSub"); self.out.setWordWrap(True); run.add(self.out)
        cl.addWidget(run); cl.addStretch(1)
        root.addWidget(_scroll(col))
        stage = QWidget(); sl = QVBoxLayout(stage); sl.setContentsMargins(16, 16, 16, 16); sl.setSpacing(12)
        head = QLabel("Co-adição de observações"); head.setStyleSheet(f"color:{T.TEXT};font-size:15px;font-weight:600;background:transparent")
        sl.addWidget(head)
        self.cv = plots.Canvas(width=8, height=4); self.cv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sl.addWidget(self.cv, 1)
        plots.draw_message(self.cv, "adicione observações e co-adicione para ver os perfis alinhados")
        root.addWidget(stage, 1)

    def _add(self):
        p, _ = QFileDialog.getOpenFileName(self, "Eventos", str(ROOT_EXPLORA()), "CSV (*.csv)")
        if not p: return
        bg = None
        cand = p.replace("events", "background")
        if cand != p and Path(cand).is_file():
            bg = cand
        else:
            b, _ = QFileDialog.getOpenFileName(self, "Fundo (opcional — cancele se não houver)", str(Path(p).parent), "CSV (*.csv)")
            bg = b or None
        self.obs.append((p, bg))
        self.lst.setText("\n".join(f"{i+1}. {Path(e).name}" + ("  +fundo" if b else "") for i, (e, b) in enumerate(self.obs)))

    def _clear(self):
        self.obs = []; self.lst.setText("nenhuma observação")

    def _coadd(self):
        if len(self.obs) < 1:
            self.out.setText("adicione ao menos uma observação"); return
        from .backend import coadd_observations, ROOT
        import time
        out = ROOT / "exploracoes" / f"combinado_gui_{int(time.time())}.npz"
        out.parent.mkdir(exist_ok=True)
        try:
            evs = [e for e, _ in self.obs]; bgs = [b for _, b in self.obs]
            res = coadd_observations(evs, bgs, str(out))
            self.npz = str(out)
            self.out.setText(f"{res['counts']} contagens · {res['exposure']:.0f} s · "
                             f"corr {res['corr']:.2f}\n{out.name}")
            self._draw(res)
            if self.status_cb: self.status_cb(f"co-adição: {res['counts']} contagens em {out.name}")
        except Exception as e:
            self.out.setText(f"erro: {e}")

    def _draw(self, res):
        import numpy as np
        ax = self.cv.ax; self.cv.clear()
        cols = [T.CYAN, T.MAGENTA, T.AMBER, T.VIOLET]
        for k, lc in enumerate(res["profiles"]):
            sh = res["shifts"][k]
            y = np.roll(lc, sh); y = y / y.mean()
            ph = (np.arange(len(y)) + 0.5) / len(y)
            ax.plot(np.concatenate([ph, ph + 1]), np.concatenate([y, y]),
                    color=cols[k % 4], lw=2, label=f"obs {k+1}" + (f" (+{sh})" if sh else ""))
        ax.set_title(f"Perfis alinhados · correlação {res['corr']:.2f}", fontsize=10)
        ax.set_xlabel("fase"); ax.set_ylabel("normalizado"); ax.legend(fontsize=8)
        self.cv.fig.tight_layout(); self.cv.draw()


class PlaceholderPanel(QWidget):
    """Aba ainda não implementada — mostra o que virá, do plano."""
    def __init__(self, title: str, lines: list[str]):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 34, 40, 34)
        lay.setSpacing(14)
        h = QLabel(title)
        h.setStyleSheet(f"color:{T.TEXT};font-size:19px;font-weight:600;background:transparent")
        lay.addWidget(h)
        tag = QLabel("planejada")
        tag.setStyleSheet(f"color:{T.AMBER};font-size:11px;letter-spacing:1px;background:transparent")
        lay.addWidget(tag)
        for ln in lines:
            b = QLabel("•  " + ln)
            b.setWordWrap(True)
            b.setStyleSheet(f"color:{T.MUTED};font-size:13px;background:transparent")
            lay.addWidget(b)
        lay.addStretch(1)
