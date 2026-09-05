"""Tema do MAGNUS: os tokens da marca viram QPalette + folha de estilo Qt.

Fonte única da verdade da paleta — os mesmos valores de
``identity/magnus-brand/magnus-tokens.css``. Se mudar a marca, muda aqui.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette, QFont, QFontDatabase
from PySide6.QtCore import Qt

# Tokens da marca (magnus-tokens.css)
VOID = "#0B1020"
PANEL = "#141C33"
PLATE = "#0E1330"
CORE = "#241C4D"
VIOLET = "#7B4DF0"   # campo magnético / primário
CYAN = "#2ED3E0"     # modo O / linhas de campo / modelo
MAGENTA = "#F0479B"  # modo X / atmosfera / linha
AMBER = "#F0A63C"    # energia / avisos
TEXT = "#E8EAF2"
MUTED = "#8A93B2"
LINE = "#26304f"
LINE_SOFT = "#1b2440"


def load_fonts() -> str:
    """Registra Poppins se houver TTF em assets/; devolve a família a usar."""
    from pathlib import Path
    assets = Path(__file__).resolve().parent / "assets" / "fonts"
    family = None
    if assets.is_dir():
        for ttf in assets.glob("Poppins*.ttf"):
            fid = QFontDatabase.addApplicationFont(str(ttf))
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                family = fams[0]
    # Se Poppins já estiver no sistema, usa-a; senão, sans do sistema.
    if family is None and "Poppins" in QFontDatabase.families():
        family = "Poppins"
    return family or "Sans Serif"


def apply(app) -> None:
    """Aplica paleta escura + QSS ao QApplication."""
    family = load_fonts()
    app.setFont(QFont(family, 10))

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(VOID))
    pal.setColor(QPalette.Base, QColor(PLATE))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANEL))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(VIOLET))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(MUTED))
    app.setPalette(pal)

    app.setStyleSheet(QSS.format(
        void=VOID, panel=PANEL, plate=PLATE, violet=VIOLET, cyan=CYAN,
        magenta=MAGENTA, amber=AMBER, text=TEXT, muted=MUTED, line=LINE,
        line_soft=LINE_SOFT, family=family))


QSS = """
* {{ font-family: "{family}"; }}
QWidget {{ color: {text}; background: {void}; }}
QToolTip {{ background: {panel}; color: {text}; border: 1px solid {line}; padding: 4px; }}

#titlebar {{ background: {plate}; border-bottom: 1px solid {line}; }}
#wordmark {{ font-size: 15px; font-weight: 600; letter-spacing: 2px; color: {text}; }}
#descriptor {{ font-size: 10px; color: {muted}; letter-spacing: 1px; }}
#ver {{ color: {muted}; border: 1px solid {line}; border-radius: 11px; padding: 2px 10px; }}

QTabWidget::pane {{ border: 0; }}
QTabBar {{ background: {plate}; qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 10px 18px; margin-right: 2px;
    border-bottom: 2px solid transparent; font-size: 13px; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {violet}; }}

QScrollArea {{ border: 0; background: transparent; }}

#card {{ background: {panel}; border: 1px solid {line}; border-radius: 11px; }}
#cardTitle {{ color: {muted}; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; }}
#rowLabel {{ color: {text}; font-size: 12px; background: transparent; }}
#rowSub {{ color: {muted}; font-size: 11px; background: transparent; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {plate}; border: 1px solid {line}; border-radius: 7px; padding: 5px 8px;
    color: {text}; font-family: "JetBrains Mono", monospace; selection-background-color: {violet}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border: 1px solid {cyan}; }}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{ background: {panel}; border: 1px solid {line}; color: {text};
    selection-background-color: {violet}; }}

QCheckBox {{ color: {text}; spacing: 8px; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border: 1.5px solid {line}; border-radius: 5px;
    background: {plate}; }}
QCheckBox::indicator:checked {{ background: {cyan}; border-color: {cyan};
    image: none; }}

QPushButton {{ background: transparent; border: 1px solid {line}; border-radius: 9px;
    padding: 9px 14px; color: {muted}; font-weight: 600; font-size: 13px; }}
QPushButton:hover {{ color: {text}; }}
QPushButton#primary {{ background: {violet}; border: 0; color: white; }}
QPushButton#primary:hover {{ background: #8a5cff; }}
QPushButton#icon {{ border: 0; padding: 6px; color: {muted}; }}
QPushButton#icon:hover {{ background: #ffffff10; border-radius: 8px; color: {text}; }}

#statusbar {{ background: {plate}; border-top: 1px solid {line}; color: {muted};
    font-family: "JetBrains Mono", monospace; font-size: 11px; }}
#statusbar QLabel {{ color: {muted}; background: transparent; }}

QProgressBar {{ background: {plate}; border: 1px solid {line}; border-radius: 5px; height: 8px;
    text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {violet}; border-radius: 4px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {line}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
