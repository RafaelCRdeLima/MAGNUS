"""Componentes reutilizáveis: cartão e linha de parâmetro."""
from __future__ import annotations

from PySide6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                               QDoubleSpinBox, QSpinBox, QCheckBox)
from PySide6.QtCore import Qt

from . import theme as T


class Card(QFrame):
    def __init__(self, title: str, dot: str = T.VIOLET):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(13, 11, 13, 13)
        lay.setSpacing(9)
        head = QHBoxLayout()
        head.setSpacing(8)
        d = QLabel("●")
        d.setStyleSheet(f"color:{dot};font-size:9px;background:transparent")
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        head.addWidget(d)
        head.addWidget(t)
        head.addStretch(1)
        lay.addLayout(head)
        self.body = lay

    def add(self, w):
        self.body.addWidget(w)
        return w

    def add_row(self, lay):
        self.body.addLayout(lay)


class ParamRow(QWidget):
    """Rótulo + campo numérico + (opcional) caixa 'ajustar'."""
    def __init__(self, label, value, *, unit="", decimals=3, step=0.1,
                 lo=-1e9, hi=1e9, integer=False, fit=None, on_change=None):
        super().__init__()
        self.setStyleSheet("background:transparent")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(9)
        lab = QLabel(label)
        lab.setObjectName("rowLabel")
        row.addWidget(lab, 1)
        if integer:
            self.field = QSpinBox()
            self.field.setRange(int(lo), int(hi))
            self.field.setValue(int(value))
        else:
            self.field = QDoubleSpinBox()
            self.field.setRange(lo, hi)
            self.field.setDecimals(decimals)
            self.field.setSingleStep(step)
            self.field.setValue(value)
        if unit:
            self.field.setSuffix(f" {unit}")
        self.field.setFixedWidth(122)
        self.field.setAlignment(Qt.AlignRight)
        if on_change:
            self.field.valueChanged.connect(lambda *_: on_change())
        row.addWidget(self.field)
        self.fit = None
        if fit is not None:
            self.fit = QCheckBox()
            self.fit.setChecked(bool(fit))
            self.fit.setToolTip("ajustar este parâmetro")
            row.addWidget(self.fit)
        elif fit is None:
            # espaço para alinhar com linhas que têm caixa
            spacer = QLabel(" ")
            spacer.setFixedWidth(17)
            spacer.setStyleSheet("background:transparent")
            row.addWidget(spacer)

    def value(self):
        return self.field.value()
