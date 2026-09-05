"""Gráficos matplotlib embutidos, no tema da marca."""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import theme as T


class Canvas(FigureCanvasQTAgg):
    def __init__(self, width=4.6, height=3.0):
        self.fig = Figure(figsize=(width, height), facecolor=T.PANEL)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style()

    def _style(self):
        ax = self.ax
        ax.set_facecolor(T.PLATE)
        for s in ax.spines.values():
            s.set_color(T.LINE)
        ax.tick_params(colors=T.MUTED, labelsize=8)
        ax.grid(True, color=T.LINE_SOFT, lw=0.6)
        ax.title.set_color(T.TEXT)
        ax.xaxis.label.set_color(T.MUTED)
        ax.yaxis.label.set_color(T.MUTED)

    def clear(self):
        self.ax.clear()
        self._style()


def draw_pulse(canvas: Canvas, phase, lc):
    ax = canvas.ax
    canvas.clear()
    import numpy as np
    ph2 = np.concatenate([phase, phase + 1])
    y2 = np.concatenate([lc, lc])
    ax.plot(ph2, y2, color=T.CYAN, lw=2)
    ax.fill_between(ph2, y2 * 0.97, y2 * 1.03, color=T.CYAN, alpha=0.15)
    fp = (lc.max() - lc.min()) / (lc.max() + lc.min()) * 100 if lc.max() > 0 else 0
    ax.set_title(f"Curva de luz dobrada  ·  fp = {fp:.0f}%", fontsize=10)
    ax.set_xlabel("fase de rotação")
    ax.set_ylabel("contagens")
    canvas.fig.tight_layout()
    canvas.draw()


def draw_spectrum(canvas: Canvas, energy, sp):
    ax = canvas.ax
    canvas.clear()
    ax.plot(energy, sp, color=T.CYAN, lw=2)
    ax.set_yscale("log")
    ax.set_title("Espectro  ·  soma em fase", fontsize=10)
    ax.set_xlabel("energia (keV)")
    ax.set_ylabel("contagens")
    canvas.fig.tight_layout()
    canvas.draw()


def draw_message(canvas: Canvas, text: str):
    canvas.clear()
    canvas.ax.text(0.5, 0.5, text, color=T.MUTED, ha="center", va="center",
                   transform=canvas.ax.transAxes, fontsize=10, wrap=True)
    canvas.ax.set_xticks([]); canvas.ax.set_yticks([])
    canvas.draw()
