#!/usr/bin/env python3
"""As duas conferencias que o relatorio da marginalizacao precisa.

O revisor, 10/10/2026, pediu separar tres coisas no relatorio. A terceira, a
mudanca dos contrastes, sai do testa_marginal_por_obs.py. Estas sao as outras
duas:

  1. INTEGRIDADE DOS TERMOS APOS CADA ESTAGIO. Dois estagios escrevem o mesmo
     npz, e o primeiro ja apagou a saida do segundo uma vez, em silencio. Aqui
     se verifica que os termos sobrevivem a sequencia inteira, por hash.

  2. A MARGINALIZACAO CONTRA A INTEGRACAO INDEPENDENTE. A quadratura do alvo usa
     64 nos por canal. Aqui a MESMA previsao e repontuada com 64, 256 e 1024
     nos: se o valor nao se mover, a quadratura esta resolvida; se mover, o
     numero de nos e insuficiente. Nao exige chamada nova do motor por nivel,
     logo nao e outra rodada.

E a ressalva do revisor, que entra no artefato: contrastes pouco alterados NAO
bastam para validar a integracao, e contrastes alterados NAO significam defeito
-- podem ser efeito legitimo do alvo novo.

Uso: python3 scripts/confere_marginal.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

CAMPOS = ("fundo_pesos", "fundo_contagens", "fundo_deslocamento",
          "fundo_canal_do_termo", "fundo_largura_do_canal")
#: Observacoes representativas: a de mais contagens, uma de cada submodo, e a de
#: fundo anomalo, que e a que mais estressa a marginalizacao.
REPRESENTATIVAS = ("0157360101", "0402850301", "0402851001")


def impressao(npz):
    """Hash dos campos de fundo, para comparar entre estagios."""
    d = np.load(npz, allow_pickle=True)
    h = hashlib.sha256()
    for k in CAMPOS:
        if k not in d.files:
            return None
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def main():
    rel = {"o_que_e": "integridade dos termos e conferencia da quadratura",
           "a_ressalva_do_revisor": (
             "contrastes pouco alterados NAO bastam para validar a integracao, e "
             "contrastes alterados NAO significam defeito: podem ser efeito "
             "legitimo do alvo novo")}

    print("=== 1: integridade dos termos de fundo apos cada estagio ===")
    npzs = sorted((PO.DESTINO).glob("obs_*.npz"))
    antes = {p.name: impressao(p) for p in npzs}
    faltam = [n for n, h in antes.items() if h is None]
    print("  estagio 0, como esta: %d npz, %d sem termos"
          % (len(antes), len(faltam)))
    # estagio que JA apagou os termos uma vez
    subprocess.run([sys.executable, "scripts/por_observacao.py", "--so-preparar"],
                   cwd=RAIZ, capture_output=True, text=True)
    depois = {p.name: impressao(p) for p in npzs}
    iguais = sum(1 for n in antes if antes[n] == depois[n])
    perdidos = [n for n in antes if antes[n] is not None and depois[n] is None]
    print("  estagio 1, depois de por_observacao.py --so-preparar:")
    print("    %d de %d com impressao IDENTICA" % (iguais, len(antes)))
    print("    %d perderam os termos" % len(perdidos))
    rel["integridade"] = {
      "npz": len(antes), "sem_termos_no_inicio": faltam,
      "identicos_apos_reexecutar_o_estagio_1": iguais,
      "perderam_os_termos": perdidos,
      "por_que_este_teste_existe": (
        "o prepara() regravava o npz com np.savez so com os campos base e "
        "APAGAVA os fundo_* do estagio seguinte, sem erro. Aconteceu de verdade "
        "e so foi descoberto ao tentar usar a marginalizacao"),
      "passa": iguais == len(antes) and not perdidos}
    print("  %s" % ("PASSA" if rel["integridade"]["passa"] else "FALHA"))

    print()
    print("=== 2: a quadratura do alvo contra integracao mais fina ===")
    print("a MESMA previsao, repontuada com 64, 256 e 1024 nos por canal")
    meta, ev = m.read_events(RAIZ / "dados/eventos.csv")
    obss = {o["obsid"]: o for o in PO.prepara(verbose=False)}
    nome, caminho, rotulo = PO.VETORES[0]
    rel["quadratura"] = {}
    print("%-12s %18s %18s %18s %12s" % ("obsid", "64 nos", "256 nos",
                                         "1024 nos", "64 -> 1024"))
    for obsid in REPRESENTATIVAS:
        base = PO.monta_pedido(obss[obsid], 120, 8)
        base["outputPrecision"] = 17
        base["marginalizeBackground"] = True
        probs = {}
        for n in (64, 256, 1024):
            b = json.loads(json.dumps(base))
            b["fundoNos"] = n
            p = m.FitProblem(b, ev, meta, None)
            p.fundo_nos = n          # o pedido pode nao expor a chave
            p._fundo_pronto = False
            probs[n] = p
        w = m.EngineWorker(list(probs[64].worker_command()))
        v = PO.le_vetor(probs[64], caminho, rotulo)
        doc = w.evaluate(probs[64].worker_line(v))
        w.close()
        if doc.get("status") != "ok":
            raise SystemExit("motor falhou em %s: %s" % (obsid, doc))
        vals = {n: float(probs[n].score_document(doc)) for n in (64, 256, 1024)}
        d = vals[1024] - vals[64]
        rel["quadratura"][obsid] = {"valores": vals, "de_64_a_1024": d}
        print("%-12s %18.8f %18.8f %18.8f %+12.2e"
              % (obsid, vals[64], vals[256], vals[1024], d), flush=True)
    pior = max(abs(v["de_64_a_1024"]) for v in rel["quadratura"].values())
    rel["quadratura_pior_mudanca_64_a_1024"] = pior
    print()
    print("pior mudanca de 64 para 1024 nos: %.3e" % pior)
    print("  %s" % ("a quadratura de 64 nos esta RESOLVIDA nestes casos"
                    if pior < 1e-6 else
                    "64 nos NAO resolvem: a mudanca e %.3e" % pior))
    rel["quadratura_resolvida_em_64_nos"] = bool(pior < 1e-6)
    rel["o_que_isto_NAO_estabelece"] = (
      "a correcao da formula marginal. Mede se a QUADRATURA dela esta resolvida "
      "e se os termos sobrevivem aos estagios. A correcao da formula foi "
      "verificada em separado, no limite degenerado que reproduz o controle e "
      "contra a inversao por funcao caracteristica")
    PO.grava_sem_sobrescrever(RAIZ / "docs/confere_marginal.json", rel)
    print("gravado em docs/confere_marginal.json")


if __name__ == "__main__":
    main()
