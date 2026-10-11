#!/usr/bin/env python3
"""Verossimilhança SOMANDO PREVISÕES POR OBSERVAÇÃO, com offset no modelo.

ROTA ESCOLHIDA PELO REVISOR em 07/10/2026: "prefiro deslocar o modelo,
preservando as contagens observadas. Os offsets relativos podem ser parâmetros
de perturbação, com uma referência global fixada para remover a degenerescência."

A FORMA. Em vez de uma grade coadicionada,

    ln L = soma_o soma_{p,e} [ n[o][p][e] ln mu[o][p][e] - mu[o][p][e] ]

com mu[o] de uma chamada do motor COM A RESPOSTA, A RESOLUÇÃO TEMPORAL e O
PERÍODO daquela observação, e o offset de fase daquela observação como
perturbação. As contagens nunca são deslocadas nem interpoladas.

O OFFSET É `magAzim`, O AZIMUTE MAGNÉTICO, NÃO `phaseOffset`. Eu havia escolhido
`phaseOffset` e estava errado: em `unpack_spots` ele entra como
`phi = phase_offset + relative_phi`, isto é o AZIMUTE DOS SPOTS, logo é inócuo
com `--no-spots` (a configuração A/B) e a priori o congela em 0 com limites de
±1e-6, de modo que variá-lo devolve -inf. Calibrado em `docs/calibracao_offset_fase.json`:

    deslocamento_em_bins = - delta_azimute / 11,25        (razão -1,0000 ± 0,0006)

A REGRA. A coadição faz `observed += roll(g_o, s_o)`, logo o bin `q` da
observação `o` aparece no bin `q + s_o` do coadicionado e é comparado com o
modelo de referência ali. Por observação o modelo precisa de
`m_o[q] = m_ref[q + s_o]`, isto é deslocamento `-s_o`, e portanto

    delta_azimute_o = +11,25 . s_o graus

com a observação de mais contagens, `0157360101`, fixa em 0 para remover a
degenerescência. O azimute é somado módulo 360, que é a identidade porque
32 bins x 11,25 = 360.

POR QUE ISSO NÃO É EQUIVALENTE À COADIÇÃO. Se todas as observações tivessem a
MESMA forma de modelo, `mu[o] = w_o m`, a coadição daria o mesmo ln L a menos da
constante `soma n_o ln w_o - (soma n_o) ln(soma w_o)`, que não depende dos
parâmetros. A forma difere por quatro coisas: resposta, offset de fase, largura
de suavização e forma da exposição por fase. Logo a equivalência tem de ser
MEDIDA, em DIFERENÇAS de ln L entre vetores, como o revisor exigiu, e vale como
diagnóstico inicial e não como validação em toda a posterior.

Uso: python3 scripts/por_observacao.py [--so-preparar] [--calibra OBSID]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import reotimiza as r  # noqa: E402
import coadicionar as co  # noqa: E402

RECON = Path(os.environ.get(
    "MAGNUS_RECONSTRUCAO",
    Path.home() / "Codes/XMM-NEWTON/reconstrucao_5906a57"))
PEDIDO = RAIZ / "exploracoes/reotimizacao_2026-10-04/pedido_uniao.json"
#: O manifesto LOCAL, com os 13 perfis por observacao registrados com
#: procedencia. O do PULSARIS continua sendo so de leitura e sem eles.
MANIFESTO = RAIZ / "instrumento/instrument_data/profiles/manifest.json"
#: O produto v3 coadicionado, de onde vem o s_o de cada observacao.
V3 = RAIZ / "exploracoes/dados_combinados_13obs_v3_2026-10-07/combinado13_v3.npz"
DESTINO = RAIZ / "exploracoes/por_observacao_2026-10-07"
VETORES = (("B colado", "docs/incidente_resolucao_2026-10-04/modosB.json", "colado"),
           ("B aberto", "docs/incidente_resolucao_2026-10-04/modosB.json", "aberto"),
           ("A", "docs/incidente_resolucao_2026-10-04/modoA.json", "aberto"))
FASES, ENERGIAS = 32, 40
GRAUS_POR_BIN = 360.0 / FASES          # 11,25, medido e nao suposto
#: A observacao de REFERENCIA GLOBAL, de offset fixo em 0: a de mais contagens,
#: a mesma que a coadicao usa como referencia da correlacao cruzada.
REFERENCIA = "0157360101"
I_AZIMUTE = None                       # resolvido por nome, nunca fixado na mao


def indice_do_azimute(prob) -> int:
    """O indice de `magAzim`, achado pelo nome. Um indice na mao e um bug."""
    nomes = list(prob.names)
    if "magAzim" not in nomes:
        raise SystemExit("o vetor nao tem magAzim: %s" % nomes)
    return nomes.index("magAzim")


def azimute_com_offset(valor: float, s_o: int) -> float:
    """OBSOLETO. Usar `offset_em_ciclos`.

    Isto punha o offset no `magAzim`, e estava errado: `magAzim` e um parametro
    FISICO livre e compartilhado, e medido NAO e um operador de fase exato.
    Rolar a grade e mover o azimute diferem por 0,061 em ln L e ate 3,0e-04 por
    celula, porque o azimute move a configuracao magnetica DENTRO da malha, que
    e fixa no referencial da estrela, mudando o erro de quadratura. Mantido so
    para reproduzir as medidas antigas.
    """
    x = valor + GRAUS_POR_BIN * int(s_o)
    x = ((x + 180.0) % 360.0) - 180.0
    if x <= -180.0 + 1e-9:             # a priori tem limites abertos em -180
        x += 360.0
    return x


def offset_em_ciclos(s_o: float) -> float:
    """O offset de fase OBSERVACIONAL daquela observacao, em ciclos.

    Convencao MEDIDA em docs/valida_offset.json, nao suposta:
        m_offset[p] = m_0[p + offset . N]
    isto e offset positivo move o padrao para bins MENORES, por offset . N bins,
    com razao 1,000000 nos dois sentidos.

    A coadicao faz `observed += roll(g_o, s_o)`, logo o bin q da observacao o
    aparece no bin q + s_o do coadicionado e e comparado com o modelo ali. Por
    observacao o modelo precisa de m_o[q] = m_ref[q + s_o], e portanto

        offset_o = s_o / N   ciclos, POSITIVO

    Aceita s_o fracionario: o offset e continuo no motor. Mas atencao, offset
    fracionario quebra o reuso barato, porque as observacoes passam a precisar de
    conjuntos de fases DIFERENTES.
    """
    return float(s_o) / FASES


def perfis_do_manifesto() -> dict:
    """{obsid: entrada}, casada pela PROCEDENCIA e nao pelo nome do arquivo."""
    perfis = json.loads(MANIFESTO.read_text())["profiles"]
    saida = {}
    for p in perfis:
        obsid = (p.get("procedencia") or {}).get("obsid")
        if obsid:
            saida[str(obsid)] = p
    return saida


#: As bases em que um caminho do manifesto pode estar, em ordem. Existem tres
#: porque as entradas nao tem uma convencao so: as herdadas do PULSARIS sao
#: relativas a raiz DELE, as 13 por observacao que eu registrei sao relativas a
#: raiz do MAGNUS, e as que o resposta_ponderada.py grava ficam em instrumento/.
BASES = ("raiz_do_magnus", "instrumento", "pulsaris")
#: Onde cada resolucao parou, com o hash. Preenchido por `caminho_absoluto`, para
#: que o artefato registre o arquivo que de fato entrou na conta.
RESOLVIDOS: dict = {}


def _hash(caminho: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def caminho_absoluto(rel: str, base: str | None = None) -> Path:
    """Resolve um caminho do manifesto, ACUSANDO ambiguidade.

    O revisor pediu a guarda, e ela e necessaria: tentar tres bases em ordem e
    pegar a primeira que existe escolhe em SILENCIO quando mais de uma tem o
    caminho, e se os arquivos forem DIFERENTES a conta roda com o errado sem
    dizer nada. Aqui, se duas bases tiverem o caminho com conteudos diferentes,
    isto para. Se tiverem o MESMO conteudo, segue e registra que havia copia.

    `base` declara a base explicitamente, que e a forma preferida: entrada do
    manifesto com `base` definida nao passa por busca nenhuma.
    """
    candidatas = {"raiz_do_magnus": RAIZ, "instrumento": RAIZ / "instrumento",
                  "pulsaris": m.PULSARIS}
    if base is not None:
        if base not in candidatas:
            raise SystemExit("base %r desconhecida; conheco %s"
                             % (base, list(candidatas)))
        alvo = candidatas[base] / rel
        if not alvo.is_file():
            raise SystemExit("a entrada declara base %r e o arquivo nao esta "
                             "la: %s" % (base, alvo))
        RESOLVIDOS[rel] = {"base": base, "caminho": str(alvo),
                           "sha256": _hash(alvo), "declarada": True}
        return alvo
    achados = [(nome, candidatas[nome] / rel) for nome in BASES
               if (candidatas[nome] / rel).is_file()]
    if not achados:
        raise SystemExit("arquivo do manifesto nao encontrado em nenhuma das "
                         "tres bases (%s): %s" % (", ".join(BASES), rel))
    hashes = {nome: _hash(caminho) for nome, caminho in achados}
    distintos = set(hashes.values())
    if len(distintos) > 1:
        raise SystemExit(
            "AMBIGUIDADE no manifesto: o caminho %r existe em mais de uma base "
            "com conteudos DIFERENTES, e escolher a primeira seria rodar a conta "
            "com o arquivo errado em silencio. Declare a base na entrada.\n"
            + "\n".join("  %-16s %s  sha %s" % (n, c, hashes[n][:16])
                         for n, c in achados))
    nome, caminho = achados[0]
    RESOLVIDOS[rel] = {"base": nome, "caminho": str(caminho),
                       "sha256": hashes[nome], "declarada": False,
                       "copias_identicas_em": [n for n, _ in achados[1:]] or None}
    return caminho


#: O cache dos metadados, para onde a reconstrucao NAO existe. No cluster ela
#: nunca vai existir, e nem deve: o que viaja sao os npz ja preparados e os
#: perfis. Sem este caminho, qualquer script que chame prepara() quebra la.
CACHE = RAIZ / "docs/por_observacao.json"


def _reenraiza(caminho: str, sha_esperado: str | None = None) -> str:
    """Troca a raiz de um caminho absoluto gravado em outra maquina.

    Os caminhos do cache sao absolutos e da maquina onde ele foi gerado. Aqui se
    procura o subdiretorio conhecido e se reenraiza em RAIZ, em vez de supor que
    o caminho inteiro valha, o que falharia em silencio devolvendo "arquivo nao
    encontrado" num lugar distante do erro real.
    """
    def confere(c):
        if sha_esperado is None:
            return str(c)
        obtido = _hash(c)
        if obtido != sha_esperado:
            raise SystemExit(
                "CONTEUDO DIFERENTE no caminho reenraizado. O arquivo existe mas "
                "nao e o mesmo, e rodar assim produziria ciencia errada em "
                "silencio.\n  caminho: %s\n  esperado: %s\n  obtido:   %s"
                % (c, sha_esperado, obtido))
        return str(c)

    if Path(caminho).is_file():
        return confere(Path(caminho))
    for sub in ("exploracoes/por_observacao_2026-10-07",
                "instrumento/instrument_data/profiles"):
        if sub in caminho:
            novo = RAIZ / sub / Path(caminho).name
            if novo.is_file():
                return confere(novo)
    raise SystemExit("caminho do cache nao resolve nem reenraizado: %s" % caminho)


def prepara_do_cache(verbose=True):
    """Os metadados do cache, com os caminhos reenraizados nesta maquina."""
    if not CACHE.is_file():
        raise SystemExit(
            "sem a reconstrucao em %s e sem o cache em %s: nao ha de onde tirar "
            "os metadados por observacao" % (RECON, CACHE))
    d = json.loads(CACHE.read_text())
    saida = []
    for o in d["observacoes"]:
        o = dict(o)
        for k, hk in (("npz", "sha256_npz"),
                      ("perfil_csv", "sha256_perfil_csv"),
                      ("perfil_rmfbin", "sha256_perfil_rmfbin")):
            if o.get(k):
                o[k] = _reenraiza(o[k], o.get(hk))
        saida.append(o)
    if verbose:
        print("metadados do CACHE %s, %d observacoes, caminhos reenraizados em %s"
              % (CACHE.name, len(saida), RAIZ))
    return saida


def prepara(verbose=True, regravar=True):
    """Prepara as treze, e com `regravar=False` NAO toca nos npz.

    O parametro existe porque quem apenas LE o produto nao precisa regravar
    nada, e regravar em paralelo foi o que corrompeu tres npz em 10/10/2026. A
    escrita e atomica agora, mas o caminho certo para um leitor e nao escrever.
    """
    """Um npz POR OBSERVACAO, contagens NAO deslocadas, com o perfil daquela obs."""
    if not (RECON / "indice.json").is_file():
        return prepara_do_cache(verbose)
    indice = json.loads((RECON / "indice.json").read_text())
    perfis = perfis_do_manifesto()
    v3 = np.load(V3, allow_pickle=True)
    desl = {str(o): int(s) for o, s in zip(v3["obsids"], v3["shifts"])}
    # a exposicao por fase de CADA observacao, nao a soma alinhada do v3
    import preparar_v3 as P
    DESTINO.mkdir(parents=True, exist_ok=True)
    saida = []
    for s in indice["sets"]:
        obsid = str(s["obsid"])
        ev = RECON / s["files"]["events"]
        grid, exp, meta, per = co.fold_obs(ev)
        # FUNDO pelo mesmo estimador por area do v2/v3, desta observacao
        # sozinha, sem a media ponderada que o coadicionado faz. O prefixo dos
        # espectros e CURTO e fica um nivel acima dos CSVs, que levam o longo.
        longo = ev.name.replace("_events.csv", "")
        curto = ("_".join(longo.split("_")[-2:]) if longo.count("_") >= 2
                 else longo)
        fundo, info_fundo = P.fundo_por_area(ev.parent.parent, curto)
        if fundo is None:
            fundo, info_fundo = P.fundo_por_area(ev.parent.parent, longo)
        if fundo is None:                       # ultimo recurso, declarado
            fundo = co.read_background(ev)
            info_fundo = {"estimador": "interpolado, por area indisponivel"}
        else:
            info_fundo = dict(info_fundo or {}, estimador="por area")
        pe = s["files"].get("phase_exposure")
        exp_fase, conv = None, None
        if pe:
            cab, tab = P.le_phase_exposure(RECON / pe)
            if len(tab.get("exposure_s", [])) == FASES:
                exp_fase = np.asarray(tab["exposure_s"], float)
                conv = {k: cab.get(k) for k in
                        ("phase_convention", "time_origin_s", "period_s",
                         "phase_reference_s", "frame_cycle_s")}
        p = perfis.get(obsid)
        if p is None:
            raise SystemExit("obsid %s sem perfil registrado no manifesto local"
                             % obsid)
        npz = DESTINO / ("obs_%s.npz" % obsid)
        guarda = dict(observed=grid.ravel(), exposure=float(exp),
                      background=np.asarray(fundo, float),
                      shifts=np.array([0]), exposures=np.array([exp]),
                      obsids=np.array([obsid]), periods=np.array([per]),
                      resolucao_temporal_us=np.array(
                          [float(s["time_resolution_us"])]),
                      submodos=np.array([s["submode"]]))
        if exp_fase is not None:
            guarda["exposicao_por_fase"] = exp_fase
        # PRESERVAR O QUE OUTRO ESTAGIO ACRESCENTOU. O
        # preparar_fundo_por_obs.py grava os termos `fundo_*` neste mesmo npz,
        # e um np.savez com so os campos base os APAGA. Aconteceu: rodei o
        # --so-preparar para instalar os hashes e perdi os termos de fundo das
        # treze, sem erro nenhum, e so descobri ao tentar usar a
        # marginalizacao. Terceira vez que sobrescrita silenciosa me custa
        # resultado nesta investigacao.
        if npz.is_file():
            try:
                antigo = dict(np.load(npz, allow_pickle=True))
            except Exception:
                antigo = {}
            extras = {k: v for k, v in antigo.items()
                      if k.startswith("fundo_") and k not in guarda}
            if extras:
                guarda.update(extras)
                if verbose:
                    print("  %s: preservados %d campos de outro estagio (%s)"
                          % (obsid, len(extras), ", ".join(sorted(extras))))
        # ESCRITA ATOMICA, e isto nao e zelo: e correcao de um dano que
        # aconteceu. Em 10/10/2026 o benchmark de paralelizacao rodou quatro
        # processos que construiam o alvo ao mesmo tempo, e cada um chamava este
        # prepara(). Com `np.savez` gravando DIRETO no caminho final, um leitor
        # pegou o arquivo pela metade e TRES dos treze npz ficaram corrompidos,
        # com "Bad magic number for file header". Foram restaurados da copia do
        # cluster, cujo sha256 confere com o manifesto.
        #
        # Gravar num temporario do MESMO diretorio e depois `os.replace` torna a
        # troca atomica: um leitor concorrente ve a versao antiga ou a nova,
        # nunca um arquivo parcial. Isso nao substitui coordenar quem escreve,
        # mas tira a janela em que ler e destrutivo.
        # O nome do temporario TERMINA em .npz de proposito: np.savez
        # acrescenta a extensao quando o nome nao a tem, e um temporario
        # chamado "...npz.parcial.123" virava "...npz.parcial.123.npz", de modo
        # que o os.replace falhava com FileNotFoundError e deixava lixo no
        # diretorio. Aconteceu na primeira versao desta correcao.
        if regravar:
            tmp = npz.with_name("%s.parcial.%d.npz" % (npz.stem, os.getpid()))
            try:
                np.savez(tmp, **guarda)
                os.replace(tmp, npz)
            finally:
                if tmp.exists():
                    tmp.unlink()
        elif not npz.is_file():
            raise FileNotFoundError(
                "regravar=False mas %s nao existe: um leitor nao pode criar o "
                "produto. Rode prepara() uma vez, em serie, antes." % npz)
        saida.append({
            "obsid": obsid, "npz": str(npz), "exposicao": float(exp),
            "periodo": float(per), "contagens": int(grid.sum()),
            "resolucao_us": float(s["time_resolution_us"]),
            "submodo": s["submode"],
            "desloc_v3_bins": desl.get(obsid),
            "delta_azimute_graus": (None if desl.get(obsid) is None else
                                    GRAUS_POR_BIN * desl[obsid]),
            "perfil_id": p["id"], "perfil_label": p["label"],
            "perfil_csv": str(caminho_absoluto(p["profile_file"],
                                                p.get("base"))),
            "perfil_rmfbin": str(caminho_absoluto(p["response_file"],
                                                   p.get("base"))),
            # SHA256 DE CADA ENTRADA CIENTIFICA. O plano exige validar caches
            # por hash, porque um caminho REENRAIZADO em outra maquina pode
            # resolver para um arquivo de MESMO NOME e conteudo diferente, e a
            # conta rodaria errada sem erro nenhum. Conferido retroativamente
            # nos jobs 47139 e 47228: os hashes batiam, mas estavam
            # desprotegidos.
            "sha256_npz": _hash(npz),
            "sha256_perfil_csv": _hash(caminho_absoluto(p["profile_file"],
                                                        p.get("base"))),
            "sha256_perfil_rmfbin": _hash(caminho_absoluto(p["response_file"],
                                                           p.get("base"))),
            "perfil_resolucao_us": float(p["time_resolution_us"]),
            "exposicao_por_fase": (None if exp_fase is None
                                   else exp_fase.tolist()),
            "convencao_de_fase": conv,
            "fundo": info_fundo,
            "procedencia_do_perfil": p.get("procedencia")})
    return saida


def monta_pedido(obs, bandas=120, refino=8, s_o=None):
    """O pedido daquela observacao: npz, periodo e a RESPOSTA DAQUELA obs.

    A resposta entra por `responseProfile`, que no adaptador VENCE o catalogo
    (`if isinstance(uploaded, dict) and uploaded.get("response_file")`), porque
    ela e *daquela* observacao e o perfil do manifesto generico e de uma
    configuracao representativa. O offset NAO entra aqui: e parametro.
    """
    base = r.monta_pedido(PEDIDO, bandas, refino, False)
    base["preparedDataNpz"] = obs["npz"]
    base["period"] = obs["periodo"]
    if s_o is not None:
        base["observationPhaseOffsetCycles"] = offset_em_ciclos(s_o)
    base["responseProfile"] = {
        "id": obs["perfil_id"], "label": obs["perfil_label"],
        "profile_file": obs["perfil_csv"],
        "response_file": obs["perfil_rmfbin"],
        "time_resolution_us": obs["perfil_resolucao_us"],
        "dead_time_us": 0.0,
        "redistribution": "full_sparse_ogip_rmf"}
    return base


def problema_de(obs, bandas=120, refino=8):
    """O FitProblem e o trabalhador daquela observacao."""
    base = monta_pedido(obs, bandas, refino)
    meta, eventos = m.read_events(RAIZ / "dados/eventos.csv")
    prob = m.FitProblem(base, eventos, meta, None)
    trab = m.EngineWorker(list(prob.worker_command()))
    return prob, trab


def le_vetor(prob, caminho, rotulo):
    dados = json.loads((RAIZ / caminho).read_text())
    r.valida_ordem(dados["nomes"], prob.names)
    v = list(dados[rotulo]["vetor"])
    for nome, valor in prob.frozen.items():
        v[prob.names.index(nome)] = float(valor)
    return v


def verifica_offset(obsid, bandas=120, refino=8):
    """VERIFICACAO EMPIRICA da regra delta = 11,25 . s_o numa observacao.

    Varre o offset em todos os 32 bins e pergunta onde o ln L DAQUELA
    observacao e maximo. Se a regra estiver certa, o maximo cai no s_o que a
    coadicao mediu por correlacao cruzada, e isso e o teste: a regra foi
    DERIVADA da algebra da coadicao, nao medida.
    """
    obss = {o["obsid"]: o for o in prepara(verbose=False)}
    if obsid not in obss:
        raise SystemExit("obsid %s ausente; tem %s" % (obsid, sorted(obss)))
    obs = obss[obsid]
    prob, trab = problema_de(obs, bandas, refino)
    ia = indice_do_azimute(prob)
    nome, caminho, rotulo = VETORES[0]
    v0 = le_vetor(prob, caminho, rotulo)
    print("=== verificacao do offset em %s ===" % obsid)
    print("vetor %s, %d bandas, refino %d; magAzim no indice %d, valor %.4f"
          % (nome, bandas, refino, ia, v0[ia]))
    print("a coadicao mediu s_o = %d bins por correlacao cruzada, logo a regra "
          "derivada preve delta = %.2f graus"
          % (obs["desloc_v3_bins"], obs["delta_azimute_graus"]))
    print("%6s %10s %14s" % ("s", "delta", "ln posterior"))
    linhas, t0 = [], time.time()
    for s in range(FASES):
        v = list(v0)
        delta = GRAUS_POR_BIN * s
        v[ia] = azimute_com_offset(v0[ia], s)
        try:
            lnp = float(prob.evaluate(trab, v))
        except Exception as exc:                # noqa: BLE001
            lnp = float("nan")
            print("  s=%d falhou: %s" % (s, exc))
        linhas.append({"s": s, "delta_graus": delta, "azimute": v[ia],
                       "ln_posterior": lnp})
        print("%6d %10.2f %14.4f" % (s, delta, lnp))
    bons = [l for l in linhas if np.isfinite(l["ln_posterior"])]
    melhor = max(bons, key=lambda l: l["ln_posterior"])
    esperado = int(obs["desloc_v3_bins"])
    # a distancia CIRCULAR em bins, porque a fase e periodica
    d = (melhor["s"] - esperado) % FASES
    d = min(d, FASES - d)
    segundo = sorted(bons, key=lambda l: -l["ln_posterior"])[1]
    print()
    print("maximo em s = %d (delta %.2f), esperado %d; distancia circular %d bins"
          % (melhor["s"], melhor["delta_graus"], esperado, d))
    print("margem sobre o segundo melhor: %.1f em ln L (s = %d)"
          % (melhor["ln_posterior"] - segundo["ln_posterior"], segundo["s"]))
    print("%s" % ("CONFERE: a regra derivada acerta o bin" if d == 0 else
                  "NAO CONFERE: a regra derivada erra por %d bins" % d))
    print("%.1f s, %.2f s por avaliacao" % (time.time() - t0,
                                            (time.time() - t0) / FASES))
    trab.close()
    return {"obsid": obsid, "vetor": nome, "bandas": bandas, "refino": refino,
            "indice_do_azimute": ia, "azimute_base": v0[ia],
            "s_esperado_pela_regra": esperado,
            "delta_esperado_graus": obs["delta_azimute_graus"],
            "s_do_maximo": melhor["s"], "distancia_circular_bins": d,
            "margem_sobre_o_segundo": (melhor["ln_posterior"]
                                       - segundo["ln_posterior"]),
            "s_do_segundo": segundo["s"],
            "confere": d == 0, "varredura": linhas,
            "segundos_por_avaliacao": (time.time() - t0) / FASES}


def manifesto_em_execucao(obss, extras=None) -> dict:
    """Manifesto emitido PELO PROPRIO JOB, dos arquivos que ele de fato abriu.

    O revisor, 09/10/2026: "os manifestos retroativos recuperam rastreabilidade,
    mas nao provam sozinhos qual codigo executou". Esta e a correcao: o manifesto
    sai em tempo de EXECUCAO, com o hash do executavel que sera chamado e das
    entradas cientificas como estao no disco agora, e vai para o log e para a
    saida. Um manifesto que eu gero depois registra o que eu ACHO que
    sincronizei, nao o que rodou.
    """
    import os
    import platform
    import socket
    import subprocess
    man = {
      "emitido_em_execucao": True,
      "maquina": socket.gethostname(),
      "slurm": {k: os.environ.get(k) for k in
                ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
                 "SLURM_MEM_PER_NODE") if os.environ.get(k)},
      "threads_declaradas": {k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
      "python": platform.python_version(),
      "executavel_do_motor": {
          "caminho": str(m.ENGINE),
          "sha256": _hash(m.ENGINE) if Path(m.ENGINE).is_file() else None},
      "entradas": [{"obsid": o["obsid"], "npz": o["npz"],
                    "sha256_npz": o.get("sha256_npz"),
                    "perfil_rmfbin": o["perfil_rmfbin"],
                    "sha256_perfil_rmfbin": o.get("sha256_perfil_rmfbin")}
                   for o in obss],
      "extras": extras or {},
    }
    try:
        man["commit"] = subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None
    except Exception:
        man["commit"] = None
    faltam = [k for k, v in man["threads_declaradas"].items() if not v]
    if faltam:
        print("AVISO: threads NAO declaradas: %s. O plano pede declaracao "
              "explicita para evitar paralelismo aninhado." % ", ".join(faltam))
    print("manifesto em execucao: maquina %s, motor sha %s, %d entradas com hash"
          % (man["maquina"], (man["executavel_do_motor"]["sha256"] or "?")[:12],
             len(man["entradas"])), flush=True)
    return man


def grava_sem_sobrescrever(caminho, dados) -> None:
    """Escrita ATOMICA que RECUSA sobrescrever, como o plano pede.

    Pre-requisito 3: "recusar sobrescrita por padrao. Escrita atomica". Os
    scripts sobrescreviam calados, e as corridas de M=15 apagaram os JSON de
    M=1, 3 e 9 do par B antes de eu notar; os valores sobreviveram so nos logs.
    """
    import json as _json
    import os
    c = Path(caminho)
    if c.exists():
        raise SystemExit(
            "RECUSANDO SOBRESCREVER %s. O plano pede recusa por padrao, e eu ja "
            "perdi resultados assim. Apague ou escolha outro caminho" % c)
    c.parent.mkdir(parents=True, exist_ok=True)
    tmp = c.with_suffix(c.suffix + ".parcial")
    tmp.write_text(_json.dumps(dados, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, c)       # atomico no mesmo sistema de arquivos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-preparar", action="store_true")
    ap.add_argument("--calibra", default=None,
                    help="obsid onde verificar a regra do offset")
    ap.add_argument("--bandas", type=int, default=120)
    ap.add_argument("--refino", type=int, default=8)
    ap.add_argument("--saida", default=str(RAIZ / "docs/por_observacao.json"))
    args = ap.parse_args()
    t0 = time.time()
    obss = prepara()
    print("=== uma observacao, um npz, um perfil, um offset ===")
    print("%-12s %-22s %7s %10s %9s %5s %8s"
          % ("obsid", "submodo", "res us", "exposicao", "contagens", "s_o",
             "delta"))
    for o in obss:
        print("%-12s %-22s %7.0f %10.1f %9d %5s %8s"
              % (o["obsid"], o["submodo"], o["resolucao_us"], o["exposicao"],
                 o["contagens"],
                 "-" if o["desloc_v3_bins"] is None else o["desloc_v3_bins"],
                 "-" if o["delta_azimute_graus"] is None
                 else "%.2f" % o["delta_azimute_graus"]))
    sem_pe = [o["obsid"] for o in obss if o["exposicao_por_fase"] is None]
    sem_area = [o["obsid"] for o in obss
                if (o["fundo"] or {}).get("estimador") != "por area"]
    print()
    print("exposicao por fase ausente em: %s" % (sem_pe or "nenhuma"))
    print("fundo sem estimador por area em: %s" % (sem_area or "nenhuma"))
    print("contagens somadas: %d, contra 329972 do v3 coadicionado"
          % sum(o["contagens"] for o in obss))
    print("exposicao somada: %.1f s, contra 125667,3 do v3"
          % sum(o["exposicao"] for o in obss))
    saida = {
        "o_que_e": "previsoes por observacao, com offset de fase no MODELO",
        "rota": ("deslocar o MODELO preservando as contagens; offsets como "
                 "perturbacoes; referencia global fixa"),
        "o_parametro_do_offset": {
            "nome": "magAzim", "achado_por": "nome, nunca por indice na mao",
            "regra": "delta_azimute_o = 11,25 . s_o graus",
            "calibracao": "docs/calibracao_offset_fase.json",
            "por_que_nao_phaseOffset": (
                "phaseOffset e o azimute dos SPOTS, inocuo com --no-spots e "
                "congelado pela priori em +-1e-6")},
        "referencia_global": REFERENCIA,
        "a_exposicao_por_fase_ainda_nao_entra_na_verossimilhanca": (
            "o npz a CARREGA, mas o adaptador ainda nao a consome; o efeito "
            "medido de ate 1,0105 em diferencas de ln L continua sendo um item "
            "aberto da frente A"),
        "observacoes": obss,
        "custo": "uma chamada do motor por observacao por avaliacao"}
    if args.calibra:
        saida["verificacao_do_offset"] = verifica_offset(
            args.calibra, args.bandas, args.refino)
    Path(args.saida).write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em %s; %.1f s" % (args.saida, time.time() - t0))


if __name__ == "__main__":
    main()
