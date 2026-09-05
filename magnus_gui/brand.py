"""A marca do MAGNUS na interface: logo e ícones de UI, dos SVGs da identidade.

Os SVGs de ``identity/magnus-brand`` usam ``stroke="currentColor"``, então os
ícones herdam a cor do contexto quando recoloridos aqui.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import QByteArray, QSize, Qt

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "identity" / "magnus-brand"

# Ícone de UI -> arquivo. A ordem espelha as abas.
UI_ICONS = {
    "modelo": "icons/ui/perfil-de-pulso.svg",
    "atmosfera": "icons/ui/atmosfera.svg",
    "ajuste": "icons/ui/ajuste.svg",
    "dados": "icons/ui/exportar.svg",
    "resultados": "icons/ui/espectro.svg",
    "campo": "icons/ui/campo-b.svg",
    "mapa": "icons/ui/mapa-de-temperatura.svg",
    "ciclotron": "icons/ui/ciclotron.svg",
    "polarizacao": "icons/ui/polarizacao.svg",
    "exportar": "icons/ui/exportar.svg",
}


def _render(svg_path: Path, size: int, color: str | None) -> QPixmap:
    data = svg_path.read_bytes()
    if color is not None:
        # Recolore o traço base (currentColor) sem tocar nos acentos fixos.
        text = data.decode("utf-8").replace("currentColor", color)
        data = text.encode("utf-8")
    renderer = QSvgRenderer(QByteArray(data))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    renderer.render(p)
    p.end()
    return pm


def icon(name: str, color: str = "#8A93B2", size: int = 40) -> QIcon:
    """QIcon de um ícone de UI, recolorido no traço base."""
    path = BRAND / UI_ICONS.get(name, UI_ICONS["ajuste"])
    if not path.is_file():
        return QIcon()
    return QIcon(_render(path, size, color))


def logo_pixmap(size: int = 34) -> QPixmap:
    """A marca (dipolo oblíquo) para o topo — cores fixas da própria marca."""
    for cand in ("logo/magnus-mark-dark.svg", "logo/magnus-mark.svg"):
        path = BRAND / cand
        if path.is_file():
            return _render(path, size, None)
    return QPixmap()


def launcher_icon() -> QIcon:
    for cand in ("icons/magnus-launcher-512.svg", "logo/magnus-mark-dark.svg"):
        path = BRAND / cand
        if path.is_file():
            return QIcon(_render(path, 256, None))
    return QIcon()
