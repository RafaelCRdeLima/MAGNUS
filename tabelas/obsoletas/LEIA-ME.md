# Tabelas obsoletas: livre-livre errado

Estas tabelas usam o fator de Gaunt NÃO MAGNÉTICO no livre-livre. Está errado no
regime quantizante, que é o desta estrela: a correção é o logaritmo de Coulomb de
Potekhin & Chabrier (2003), Eq. 44, aplicado por componente cíclica.

Elas **não foram apagadas** por três razões:

1. 47 ajustes registrados em `exploracoes/` apontam para elas. Apagar tornaria
   esses ajustes irreprodutíveis.
2. O erro que elas carregam é conhecido e medido, então elas servem de controle:
   é comparando com elas que se mostra o tamanho do efeito.
3. Estão no histórico do git de qualquer forma.

Movê-las para cá faz qualquer uso acidental **falhar alto**, com arquivo não
encontrado, em vez de produzir silenciosamente um resultado errado. É o modo de
falha seguro.

## Correspondência

| Obsoleta | Substituta |
|---|---|
| `magnus_campoBg_denso_ext.magnus` | `magnus_campoBg_denso_v2.magnus` (eixos menores; a extensão está em curso) |
| `magnus_campoBg_denso.magnus` | `magnus_campoBg_denso_v2.magnus` |
| `magnus_campoBg.magnus` | `magnus_campoBg_denso_v2.magnus` |
| `magnus_campoB.magnus` | `magnus_campoBg_denso_v2.magnus` |

Para reproduzir um ajuste antigo, aponte o `atmosphereTable` do `pedido.json`
para `tabelas/obsoletas/<nome>` e saiba que o resultado carrega o erro de
opacidade.

## O que mudou, em uma linha

O livre-livre corrigido torna o modo X bem menos transparente, o que muda o feixe
emergente e, com ele, a modulação prevista. Qualquer comparação entre estas
tabelas e as atuais tem de ser feita com o ajuste LIVRE: congelar a geometria no
melhor ajuste de uma delas mede a outra fora da casa dela.
