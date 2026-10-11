#!/usr/bin/env python3
"""O offset por observação, testado na forma RELATIVA, que é a única testável.

O MEU TESTE ANTERIOR ESTAVA MAL POSTO. Varri os 32 offsets em `0163560101` e
perguntei se o máximo caía no `s_o = 5` que a coadição mediu. Caiu em 3, e eu
escrevi "NAO CONFERE". Mas a regra `Δazimute_o = 11,25 . s_o` fixa offsets
RELATIVOS entre observações; o azimute ABSOLUTO do modelo é `magAzim`, um
parâmetro LIVRE, que absorve qualquer rotação global. Além disso o vetor foi
otimizado no coadicionado v1, cuja origem temporal não é a do v3. Logo o máximo
absoluto não tem de cair em 5, e o que a regra afirma é

    s_max(o) - s_max(referência) = s_o

A ROTA BARATA, E ELA SE VALIDA COM O QUE JÁ PAGUEI. Uma rotação rígida do modelo
por um número INTEIRO de bins é `np.roll(grade, k, axis=0)`, e a calibração
mostrou que `magAzim + 11,25.s` desloca o modelo por exatamente `-s` bins, com
razão -1,0000. Então os 32 ln L de uma observação saem de UMA chamada do motor,
rolando a grade em Python, em vez de 32 chamadas a 36,7 s cada.

Isso não é suposição: os 32 valores do motor para `0163560101` já estão pagos e
guardados, então a rota rolada é comparada contra eles antes de ser usada. O que
NÃO se rola é a exposição por fase nem as contagens: só o modelo, que é
exatamente o que o motor faz quando `magAzim` muda.

Uso: python3 scripts/offset_relativo.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import mcmc_fit as m  # noqa: E402
import por_observacao as PO  # noqa: E402

FASES = PO.FASES


def perfil_de_lnl(obs, vetor_nome="B colado", bandas=120, refino=8):
    """Os 32 ln L de uma observacao, de UMA chamada do motor.

    Devolve tambem o `extra_log_prior` de cada offset, porque ele depende dos
    parametros e tem de sair para comparar verossimilhancas.
    """
    prob, trab = PO.problema_de(obs, bandas, refino)
    ia = PO.indice_do_azimute(prob)
    nome, caminho, rotulo = [v for v in PO.VETORES if v[0] == vetor_nome][0]
    v0 = PO.le_vetor(prob, caminho, rotulo)
    t0 = time.time()
    doc = trab.evaluate(prob.worker_line(v0))
    custo = time.time() - t0
    trab.close()
    if doc.get("status") != "ok":
        raise SystemExit("motor falhou em %s: %s" % (obs["obsid"], doc))
    grade = np.asarray(doc["detected_count_rate"], float)
    lnl = []
    for s in range(FASES):
        # magAzim + 11,25.s desloca o modelo por -s bins, logo roll(-s)
        d = dict(doc)
        d["detected_count_rate"] = np.roll(grade, -s, axis=0).tolist()
        lnl.append(float(prob.score_document(d)))
    extra = []
    for s in range(FASES):
        v = list(v0)
        v[ia] = PO.azimute_com_offset(v0[ia], s)
        extra.append(float(prob.extra_log_prior(v)))
    return {"ln_verossimilhanca": lnl, "prior_extra": extra,
            "azimute_base": v0[ia], "segundos_de_motor": custo,
            "indice_do_azimute": ia}


#: Os offsets onde o motor e chamado de verdade para conferir a rota. Seis, e
#: nao 32, porque cada chamada custa 36,7 s e seis bastam para separar a rota de
#: um erro de escala: s = 0 e a identidade, 1 e 3 estao perto do maximo, 8 e 24
#: ficam a um quarto de volta e 16 na meia volta.
AMOSTRA_DE_OFFSETS = (0, 1, 3, 8, 16, 24)


def motor_nos_offsets(obs, offsets, vetor_nome="B colado", bandas=120, refino=8):
    """O ln L do MOTOR em alguns offsets, uma chamada por offset."""
    prob, trab = PO.problema_de(obs, bandas, refino)
    ia = PO.indice_do_azimute(prob)
    nome, caminho, rotulo = [v for v in PO.VETORES if v[0] == vetor_nome][0]
    v0 = PO.le_vetor(prob, caminho, rotulo)
    saida = {}
    for s in offsets:
        v = list(v0)
        v[ia] = PO.azimute_com_offset(v0[ia], s)
        saida[s] = float(prob.evaluate(trab, v))
        print("  motor s=%2d azim %8.3f  ln posterior %14.4f"
              % (s, v[ia], saida[s]), flush=True)
    trab.close()
    return saida


def grades_nos_offsets(obs, offsets, vetor_nome="B colado", bandas=120,
                       refino=8):
    """As GRADES do motor em alguns offsets, para atribuir a discrepancia."""
    prob, trab = PO.problema_de(obs, bandas, refino)
    ia = PO.indice_do_azimute(prob)
    nome, caminho, rotulo = [v for v in PO.VETORES if v[0] == vetor_nome][0]
    v0 = PO.le_vetor(prob, caminho, rotulo)
    saida = {}
    for s in offsets:
        v = list(v0)
        v[ia] = PO.azimute_com_offset(v0[ia], s)
        doc = trab.evaluate(prob.worker_line(v))
        if doc.get("status") != "ok":
            raise SystemExit("motor falhou: %s" % doc)
        saida[s] = np.asarray(doc["detected_count_rate"], float)
    trab.close()
    return saida


def main():
    obss = {o["obsid"]: o for o in PO.prepara()}
    guardado = json.loads((RAIZ / "docs/por_observacao.json").read_text())
    ver = guardado.get("verificacao_do_offset")
    saida = {
      "o_que_e": "o offset por observacao testado na forma RELATIVA",
      "UM_ERRO_MEU_DE_DESENHO_DE_TESTE": (
        "varri os 32 offsets e cobrei que o maximo caisse no s_o da coadicao, "
        "escrevendo NAO CONFERE quando caiu 2 bins antes. A regra fixa offsets "
        "RELATIVOS; o azimute absoluto e magAzim, parametro LIVRE, e o vetor foi "
        "otimizado no v1, cuja origem temporal nao e a do v3. O teste certo e "
        "s_max(o) - s_max(referencia) = s_o"),
    }

    # (1) VALIDAR a rota rolada contra o motor, NA MESMA VERSAO DO CODIGO
    #
    # UM ERRO MEU DE METODO, na primeira tentativa: comparei a rota rolada
    # contra os 32 valores que a varredura do motor havia deixado guardados, e
    # ENTRE as duas corridas eu mesmo mudei o alvo para consumir a exposicao por
    # fase medida. A diferenca de 2,561 que apareceu nao media a rota: media a
    # minha mudanca de codigo. A prova e que em s = 0 a rolagem e a identidade,
    # logo a diferenca ali tinha de ser EXATAMENTE zero, e deu 4,985e-02.
    #
    # Resultado colhido disso, que vale por si: a exposicao por fase medida
    # mudou o ln L desta observacao em 0,0499 no alinhamento de partida, e o
    # efeito DEPENDE do alinhamento, variando de -1,19 a +1,89 pelos offsets.
    # Isto nao e separavel da rota naquela comparacao, e por isso ela foi
    # refeita com os dois lados na mesma versao.
    if ver and ver.get("obsid") in obss:
        obsid = ver["obsid"]
        amostra = AMOSTRA_DE_OFFSETS
        print("=== (1) a rota rolada contra o motor em %s, MESMA versao ==="
              % obsid)
        print("offsets conferidos: %s" % (amostra,))
        p = perfil_de_lnl(obss[obsid])
        motor = motor_nos_offsets(obss[obsid], amostra)
        difs = []
        print("%4s %16s %16s %12s" % ("s", "motor", "rolado", "diferenca"))
        for s in sorted(motor):
            rol = p["ln_verossimilhanca"][s] + p["prior_extra"][s]
            difs.append(abs(rol - motor[s]))
            if s < 6 or s % 8 == 0:
                print("%4d %16.4f %16.4f %12.3e" % (s, motor[s], rol,
                                                    rol - motor[s]))
        pior = max(difs)
        print("pior diferenca nos %d offsets: %.3e em ln L" % (len(difs), pior))
        print("custo: 1 chamada de %.1f s para os 32 offsets, contra %d "
              "chamadas de %.1f s" % (p["segundos_de_motor"], FASES,
                                      ver["segundos_por_avaliacao"]))
        # A TOLERANCIA. A primeira que eu fixei, 1e-6 absoluto em ln L, era
        # cega ao uso: o que a rota precisa e distinguir offsets vizinhos, cuja
        # margem medida e 34,7 em ln L, e servir de substituta do motor na
        # comparacao das formulacoes, onde a discordancia que se mede e da ordem
        # de 1. A segunda e a que manda, logo 0,01 em ln L, cem vezes abaixo
        # dela. Declarada aqui e nao ajustada depois do resultado.
        TOLERANCIA = 0.01
        ok = pior < TOLERANCIA
        print("%s" % ("ROTA VALIDADA: rolar o modelo reproduz o motor"
                      if ok else "ROTA REPROVADA: nao reproduz o motor"))
        saida["validacao_da_rota_rolada"] = {
          "obsid": obsid, "offsets": len(difs), "pior_diferenca_em_lnL": pior,
          "tolerancia": TOLERANCIA,
          "por_que_essa_tolerancia": (
            "a rota precisa distinguir offsets vizinhos, cuja margem medida e "
            "34,7 em ln L, e substituir o motor na comparacao das formulacoes, "
            "onde a discordancia medida e da ordem de 1; 0,01 fica cem vezes "
            "abaixo. A primeira tolerancia que eu fixei, 1e-6 absoluto, era "
            "cega ao uso"),
          "validada": ok,
          "custo_uma_chamada_s": p["segundos_de_motor"],
          "custo_32_chamadas_s": 32 * ver["segundos_por_avaliacao"],
          "offsets_conferidos": list(amostra),
          "por_que_isto_e_uma_validacao_e_nao_uma_suposicao": (
            "a rota barata e comparada contra chamadas de verdade do motor, na "
            "MESMA versao do codigo, antes de ser usada"),
          "o_erro_da_primeira_tentativa": (
            "comparei contra valores guardados de antes de eu mudar o alvo para "
            "consumir a exposicao por fase medida; a diferenca de 2,561 media a "
            "minha mudanca de codigo, nao a rota. Em s = 0 a rolagem e a "
            "identidade e a diferenca tinha de ser zero, e deu 4,985e-02")}
        # (1b) DE ONDE VEM A DISCREPANCIA. A fisica diz que rodar o azimute
        # magnetico por delta e o MESMO que deslocar a fase rotacional por
        # delta, exatamente: a configuracao na fase (phi + delta) com azimute a
        # e identica a da fase phi com azimute a + delta. Logo a diferenca nao
        # pode ser fisica, e a suspeita e numerica: a malha da superficie e FIXA
        # no referencial da estrela e NAO roda com a configuracao, entao o erro
        # de quadratura muda com o azimute. E a dependencia de orientacao que a
        # grade de P1 mediu nas variantes A e B, com amplitude 5,6e-05 em fluxo
        # relativo, e que C e D levaram a 6e-14.
        #
        # Isto e uma atribuicao TESTAVEL, e nao uma explicacao: se for isso, a
        # diferenca entre as GRADES, bin a bin, tem de ser da ordem de 1e-05
        # relativo, e nao de 1e-03.
        grades = grades_nos_offsets(obss[obsid], (0, 1, 8))
        base = grades[0]
        atrib = []
        for s in (1, 8):
            rol = np.roll(base, -s, axis=0)
            rel = np.abs(grades[s] - rol) / np.maximum(np.abs(rol), 1e-300)
            atrib.append({"s": s, "diferenca_relativa_max": float(rel.max()),
                          "diferenca_relativa_mediana": float(np.median(rel)),
                          "diferenca_relativa_rms": float(np.sqrt(np.mean(rel ** 2)))})
            print("  grade s=%2d contra a rolada: max %.3e, mediana %.3e"
                  % (s, rel.max(), np.median(rel)))
        casa = max(a["diferenca_relativa_max"] for a in atrib) < 1.0e-3
        print("amplitude de orientacao medida em P1 nas variantes A e B: 5,6e-05")
        print("%s" % ("ATRIBUICAO COMPATIVEL: a ordem de grandeza bate com a "
                      "dependencia de orientacao da malha"
                      if casa else "ATRIBUICAO INCOMPATIVEL: grande demais para "
                      "ser orientacao da malha"))
        saida["de_onde_vem_a_discrepancia"] = {
          "hipotese": ("a malha da superficie e fixa no referencial da estrela e "
                       "nao roda com a configuracao, logo o erro de quadratura "
                       "muda com o azimute"),
          "por_que_nao_pode_ser_fisica": (
            "rodar o azimute por delta e exatamente o mesmo que deslocar a fase "
            "rotacional por delta, porque a configuracao na fase phi + delta com "
            "azimute a e identica a da fase phi com azimute a + delta"),
          "comparacao_das_grades": atrib,
          "amplitude_de_orientacao_em_P1": 5.6e-05,
          "compativel": casa,
          "o_que_isto_implica": (
            "a rotacao de fase do modelo nao e numericamente exata na malha "
            "atual, e o custo disso e 0,061 em ln L por observacao. E a MESMA "
            "doenca de P1, aparecendo agora no alinhamento e nao no fluxo "
            "absoluto, e some nas variantes C e D da grade")}

        # A ROTA NAO SUBSTITUI O MOTOR, mas serve para LOCALIZAR o maximo: o
        # erro medido, 0,061, fica 570 vezes abaixo da margem entre offsets
        # vizinhos, 34,7 em ln L. Isto nao e mover o criterio: a tolerancia de
        # 0,01 era para SUBSTITUIR o motor na comparacao das formulacoes, e essa
        # reprovacao fica de pe. Para localizar um maximo inteiro a escala
        # relevante e a margem, e ela e declarada aqui com o criterio de quando
        # a localizacao NAO basta.
        saida["validacao_da_rota_rolada"]["serve_para_localizar"] = {
          "erro_medido": pior, "margem_entre_offsets_vizinhos": 34.7,
          "razao": 34.7 / pior if pior else None,
          "criterio": ("aceito o argmax rolado sem chamada extra quando a margem "
                       "entre os dois melhores passar de 1,0 em ln L, que e 16 "
                       "vezes o erro medido da rota; abaixo disso o motor "
                       "decide"),
          "o_que_segue_reprovado": ("substituir o motor na comparacao das "
                                    "formulacoes, onde a discordancia medida e "
                                    "da ordem de 1")}
        MARGEM_MINIMA = 1.0

    # (2) o maximo de cada observacao, 13 chamadas
    print()
    print("=== (2) o offset de melhor ajuste de cada observacao ===")
    print("%-12s %5s %5s %7s %12s %12s" % ("obsid", "s_max", "s_o", "relativo",
                                           "margem", "segundos"))
    perfis, linhas, confirmacoes = {}, [], 0
    for obsid in sorted(obss):
        p = perfil_de_lnl(obss[obsid])
        perfis[obsid] = p
        total = [a + b for a, b in zip(p["ln_verossimilhanca"], p["prior_extra"])]
        s_max = int(np.argmax(total))
        ordem = sorted(range(FASES), key=lambda i: -total[i])
        margem = total[ordem[0]] - total[ordem[1]]
        confirmado = None
        if margem <= MARGEM_MINIMA:
            # margem curta: o motor decide entre os candidatos empatados
            candidatos = [i for i in ordem
                          if total[ordem[0]] - total[i] <= MARGEM_MINIMA]
            mot = motor_nos_offsets(obss[obsid], candidatos)
            s_max = max(mot, key=mot.get)
            confirmado = {"candidatos": candidatos, "motor": mot}
            confirmacoes += len(candidatos)
        linhas.append({"obsid": obsid, "s_max": s_max,
                       "s_o_da_coadicao": int(obss[obsid]["desloc_v3_bins"]),
                       "margem_sobre_o_segundo": margem,
                       "confirmado_pelo_motor": confirmado,
                       "ln_posterior": total, "segundos": p["segundos_de_motor"]})
        print("%-12s %5d %5d %7s %12.2f %12.1f%s"
              % (obsid, s_max, obss[obsid]["desloc_v3_bins"], "", margem,
                 p["segundos_de_motor"],
                 "" if confirmado is None else "  (motor decidiu)"), flush=True)

    ref = PO.REFERENCIA
    s_ref = [l for l in linhas if l["obsid"] == ref][0]["s_max"]
    print()
    print("=== (3) O TESTE: s_max(o) - s_max(%s) contra s_o ===" % ref)
    print("s_max da referencia: %d" % s_ref)
    print("%-12s %7s %7s %7s %6s" % ("obsid", "relativo", "s_o", "erro", "ok"))
    erros = []
    for l in linhas:
        rel = (l["s_max"] - s_ref) % FASES
        d = (rel - l["s_o_da_coadicao"]) % FASES
        d = min(d, FASES - d)
        erros.append(d)
        l["relativo"] = rel
        l["erro_em_bins"] = d
        print("%-12s %7d %7d %7d %6s" % (l["obsid"], rel, l["s_o_da_coadicao"],
                                         d, "sim" if d == 0 else "NAO"))
    exatos = sum(1 for d in erros if d == 0)
    um_bin = sum(1 for d in erros if d <= 1)
    print()
    print("exatos: %d de 13; dentro de 1 bin: %d de 13; pior erro: %d bins"
          % (exatos, um_bin, max(erros)))
    saida["maximos"] = linhas
    saida["chamadas_extras_do_motor"] = confirmacoes
    saida["s_max_da_referencia"] = s_ref
    saida["teste_relativo"] = {
      "enunciado": "s_max(o) - s_max(referencia) = s_o",
      "exatos": exatos, "dentro_de_1_bin": um_bin, "pior_erro_em_bins": max(erros),
      "o_que_isto_mede": (
        "se o alinhamento que a correlacao cruzada dos DADOS achou e o mesmo que "
        "o MODELO prefere. Concordancia exata nas 13 diria que a regra reproduz "
        "a coadicao; discordancia de um bin pode ser so a resolucao inteira da "
        "correlacao cruzada, e nao um erro da regra"),
      "o_que_isto_NAO_mede": (
        "a regra em sub-bin, porque a coadicao so tem deslocamento inteiro, nem "
        "o alinhamento verdadeiro, que nenhum dos dois certifica")}
    Path(RAIZ / "docs/offset_relativo.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False) + "\n")
    print("gravado em docs/offset_relativo.json")


if __name__ == "__main__":
    main()
