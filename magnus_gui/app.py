"""Janela principal do MAGNUS: barra da marca, abas e barra de status."""
from __future__ import annotations

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QTabWidget, QPushButton)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QSize

from . import theme as T
from . import brand
from .panels import ModeloPanel, AjustePanel, PlaceholderPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAGNUS")
        self.setWindowIcon(brand.launcher_icon())
        self.resize(1180, 760)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._titlebar())
        root.addWidget(self._tabs(), 1)
        root.addWidget(self._statusbar())
        self.setCentralWidget(central)
        self._set_status("motor pronto", live=True)

    # ---- barra do topo com a marca ----
    def _titlebar(self):
        bar = QWidget()
        bar.setObjectName("titlebar")
        bar.setFixedHeight(56)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(11)
        mark = QLabel()
        pm = brand.logo_pixmap(34)
        if not pm.isNull():
            mark.setPixmap(pm)
        lay.addWidget(mark)
        text = QWidget()
        tl = QVBoxLayout(text)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        wm = QLabel("MAGNUS")
        wm.setObjectName("wordmark")
        de = QLabel("atmosferas magnetizadas de estrelas de nêutrons")
        de.setObjectName("descriptor")
        tl.addWidget(wm)
        tl.addWidget(de)
        lay.addWidget(text)
        lay.addStretch(1)
        ver = QLabel("v0.1.0")
        ver.setObjectName("ver")
        lay.addWidget(ver)
        return bar

    # ---- abas ----
    def _tabs(self):
        tabs = QTabWidget()
        tabs.setIconSize(QSize(19, 19))

        self.modelo = ModeloPanel(status_cb=self._set_status)
        tabs.addTab(self.modelo, brand.icon("modelo"), "  Modelo")

        tabs.addTab(PlaceholderPanel("Atmosfera", [
            "Ver os eixos da tabela de intensidade (lg B, lg T, θ_B, μ, lg E).",
            "Cortes de I(E, μ, θ_B) e o feixe angular.",
            "Mapa de temperatura T(θ) na esfera — o rotador oblíquo.",
        ]), brand.icon("atmosfera"), "  Atmosfera")

        self.ajuste = AjustePanel(self.modelo, status_cb=self._set_status)
        tabs.addTab(self.ajuste, brand.icon("ajuste"), "  Ajuste")

        tabs.addTab(PlaceholderPanel("Dados", [
            "Exportar eventos e fundo do XREDUX para o formato pulsaris.",
            "Co-adicionar observações por cross-correlação da forma do pulso.",
        ]), brand.icon("dados"), "  Dados")

        tabs.addTab(PlaceholderPanel("Resultados", [
            "Cornerplot, tabela de parâmetros (mediana, 1σ, rhat).",
            "Bandas 1σ/2σ do pulso e do espectro sobre os dados.",
            "Comparar modelos e exportar figuras, tabela e o pedido.",
        ]), brand.icon("resultados"), "  Resultados")

        tabs.setCurrentIndex(0)
        return tabs

    # ---- barra de status ----
    def _statusbar(self):
        bar = QWidget()
        bar.setObjectName("statusbar")
        bar.setFixedHeight(30)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 4, 16, 4)
        lay.setSpacing(18)
        self._dot = QLabel("●")
        self._msg = QLabel("iniciando…")
        lay.addWidget(self._dot)
        lay.addWidget(self._msg)
        lay.addStretch(1)
        src = QLabel("MAGNUS · motor local")
        lay.addWidget(src)
        return bar

    def _set_status(self, msg: str, live: bool = False):
        self._dot.setStyleSheet(
            f"color:{T.CYAN if live else T.MUTED};background:transparent")
        self._msg.setText(msg)


def main():
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("MAGNUS")
    T.apply(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
