#!/bin/bash
# Reotimizacao dos tres pontos de partida a 120 bandas, refino 8.
#
# Decidido em 04/10/2026: direto a 120 bandas, sem polimento previo a 26, para
# que os tres pontos sejam otimizados sob O MESMO operador e nenhum otimo seja
# transportado de uma resolucao inferior.
#
# ONDE RODAR. Local, e nao no cluster. Por avaliacao o sci-com saiu 1,9 a 2,4
# vezes mais lento nas proprias corridas do projeto: a 46150 deu 1,56 s por
# avaliacao a 26 bandas e a 45831 deu 1,20 s, contra 0,643 s medidos aqui na
# mesma configuracao. A 120 bandas e refino 8, com 35,3 s por avaliacao aqui,
# 2000 avaliacoes dao 19,6 h locais contra 37 a 47 h la. O fator foi medido a
# 26 bandas, onde parte do custo e sobrecarga de Python por iteracao e nao
# trabalho do motor, entao ele e LIMITE SUPERIOR da lentidao a 120 bandas: o
# custo real no cluster fica entre 35 e 85 s por avaliacao. Para decidir pelo
# cluster, medir la primeiro, o que exige recompilar o motor, o que por sua vez
# nao pode ser feito com job nenhum rodando.
#
# Memoria e trivial, cerca de 100 MB por motor, e sao 3 processos em 12 nucleos.
#
# QUATRO RODADAS, escalas 1,0 / 0,5 / 0,25 / 0,125 sobre o passo calibrado. O
# PLATO entre elas e a evidencia de estabilidade numerica pedida, e vale mais que
# o numero final: medido a 26 bandas no vetor B aberto, os incrementos foram
# +1,1677, +0,1651, +0,0101 e +0,0008, cada rodada ganhando cerca de dez vezes
# menos, e a ultima trezentas vezes abaixo da variacao de 0,25 entre malhas. Duas
# rodadas dariam 99 por cento do ganho, mas sem demonstrar o plato. Aquele plato
# foi medido a 26 bandas: a 120 a funcao e outra e o perfil pode diferir.
#
# Powell foi testado e REJEITADO: com o mesmo orcamento ele declarou sucesso nas
# duas rodadas e parou 0,46 abaixo do ponto do simplex, usando 708 avaliacoes
# contra 1039. A busca em linha estagna num vale estreito e chama isso de
# convergencia; aceitar aquele "terminated successfully" seria reportar
# convergencia que nao existe.
#
# O otimizador empurra CONTRA O TETO DE MASSA: a partida de B aberto tem
# M = 2,4877 e o otimo a 26 bandas foi para 2,494, contra o teto de 2,5. Os tres
# pontos de partida estao a 99,4 a 99,6 por cento do teto, com folga em u de 0,4
# a 0,6 por cento da largura do prior. O maximo em compactness e de BORDA DO
# PRIOR, nao dos dados, e a sensibilidade ao teto e item proprio do plano.
#
# Depois das tres partidas, valida_otimo.py reavalia cada otimo a 240 e 400
# bandas, com uma e duas imagens, e por bin, como o revisor exigiu: a variacao
# de 0,25 medida nos vetores antigos nao se transfere para um otimo que caia em
# outra regiao.
set -u
cd "$(dirname "$0")/.." || exit 1
export MAGNUS_INSTRUMENT_DIR="${MAGNUS_INSTRUMENT_DIR:-$PWD/instrumento}"
D=exploracoes/reotimizacao_2026-10-04
mkdir -p "$D"

[ -f "$D/pedido_uniao.json" ] || { echo "pedido ausente: $D/pedido_uniao.json"; exit 1; }
[ -x build/magnus_engine ] || { echo "motor ausente: build/magnus_engine"; exit 1; }
# o --help do motor passou a listar as flags da quadratura, mas o teste olha o
# BINARIO: um help desatualizado nao prova nada sobre o que o parser aceita, e
# foi assim que eu confirmei que o binario do cluster nao refinava.
for f in --sphere-bands --boundary-refine; do
    strings build/magnus_engine | grep -qx -- "$f" || {
        echo "o binario nao contem $f: recompile antes de reotimizar"; exit 1; }
done

echo "=== reotimizacao a 120 bandas, refino 8, $(date '+%F %T')"
echo "=== motor $(sha256sum build/magnus_engine | cut -c1-16), commit $(git rev-parse --short HEAD)"

lanca() {  # rotulo arquivo-de-vetores etiqueta
    python3 scripts/reotimiza.py \
        --vetor "$2" --rotulo "$3" \
        --pedido "$D/pedido_uniao.json" \
        --bandas 120 --refino 8 \
        --avaliacoes 500 500 500 500 --passo 1.0 0.5 0.25 0.125 \
        --saida "$D/$1.json" > "$D/$1.log" 2>&1 &
    echo "  partida $1 (pid $!)"
}

lanca B_colado docs/incidente_resolucao_2026-10-04/modosB.json colado
lanca B_aberto docs/incidente_resolucao_2026-10-04/modosB.json aberto
lanca A        docs/incidente_resolucao_2026-10-04/modoA.json  aberto
wait
echo "=== partidas terminadas $(date '+%F %T')"

for f in B_colado B_aberto A; do
    [ -f "$D/$f.json" ] || { echo "FALHOU: $f, veja $D/$f.log"; tail -5 "$D/$f.log"; }
done

echo "=== revalidacao dos otimos"
python3 scripts/valida_otimo.py "$D"/B_colado.json "$D"/B_aberto.json "$D"/A.json \
    --saida "$D/validacao_otimos.json" 2>&1 | tee "$D/validacao.log"
echo "=== fim $(date '+%F %T')"
