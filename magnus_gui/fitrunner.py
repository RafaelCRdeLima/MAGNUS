"""Dispara o ajuste MCMC em processo separado e acompanha o progresso.

Usa QProcess (integra com o laço de eventos do Qt: não bloqueia a GUI) e lê o
checkpoint .npz periodicamente para estimar quantos passos já correram e um rhat
grosseiro. Cancelar mata o filho limpo.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

ROOT = Path(__file__).resolve().parents[1]


class FitRunner(QObject):
    progress = Signal(int, int, float)   # passo, total, rhat aproximado
    finished = Signal(dict)              # resultado.json
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, python_exe: str = "python3"):
        super().__init__()
        self.python = python_exe
        self.proc: QProcess | None = None
        self.request_path: Path | None = None
        self.checkpoint: Path | None = None
        self.iterations = 0
        self._poll = QTimer(self)
        self._poll.setInterval(3000)
        self._poll.timeout.connect(self._read_checkpoint)
        self._stdout = ""

    def start(self, request: dict, events: str, background: str | None):
        out_dir = Path(tempfile.mkdtemp(prefix="magnus_fit_"))
        self.request_path = out_dir / "pedido.json"
        self.checkpoint = out_dir / "ckpt.npz"
        request = dict(request)
        request["checkpointPath"] = str(self.checkpoint)
        request.setdefault("checkpointEvery", 25)
        self.iterations = int(request.get("iterations", 200))
        self.request_path.write_text(json.dumps(request, ensure_ascii=False, indent=1))

        args = [str(ROOT / "scripts" / "mcmc_fit.py"),
                "--events", events, "--request", str(self.request_path)]
        if background:
            args += ["--background", background]
        self._stdout = ""
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(self.python, args)
        self._poll.start()
        self.log.emit("ajuste iniciado…")

    def cancel(self):
        self._poll.stop()
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.emit("ajuste cancelado")

    def _on_stdout(self):
        self._stdout += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")

    def _on_stderr(self):
        msg = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace").strip()
        if msg:
            self.log.emit(msg.splitlines()[-1])

    def _read_checkpoint(self):
        if not (self.checkpoint and self.checkpoint.is_file()):
            return
        try:
            import numpy as np
            d = np.load(self.checkpoint, allow_pickle=True)
            S = d["samples"]                     # (walkers, passos, dim)
            steps = S.shape[1] if S.ndim == 3 else 0
            # rhat grosseiro na metade tardia
            rhat = float("nan")
            if S.ndim == 3 and steps > 6:
                half = S[:, steps // 2:, :]
                m = half.mean(1); v = half.var(1, ddof=1)
                B = half.shape[1] * m.var(0, ddof=1)
                W = v.mean(0)
                with np.errstate(all="ignore"):
                    r = np.sqrt(((half.shape[1] - 1) / half.shape[1] * W + B / half.shape[1]) / W)
                rhat = float(np.nanmax(r))
            self.progress.emit(steps, self.iterations, rhat)
        except Exception:
            pass

    def _on_finished(self, code, _status):
        self._poll.stop()
        if code != 0:
            self.failed.emit("o ajuste terminou com erro (código %s)" % code)
            return
        js = [l for l in self._stdout.splitlines() if l.strip().startswith("{")]
        if not js:
            self.failed.emit("o ajuste não devolveu resultado")
            return
        try:
            result = json.loads(js[-1])
        except Exception as e:
            self.failed.emit(f"resultado ilegível: {e}")
            return
        # grava um resultado.json ao lado do pedido, para a aba Resultados
        try:
            (self.request_path.parent / "resultado.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1))
        except Exception:
            pass
        self.finished.emit(result)
