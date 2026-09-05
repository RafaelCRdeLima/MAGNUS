# Magnus — identidade visual

Programa de análise de atmosferas magnetizadas de estrelas de nêutrons.
Mesma família visual de ODEROM, Pulsaris e ÁLETRA: marca abstrata que
codifica a física + wordmark em Poppins Medium.

## Arquivos

    logo/magnus-mark.svg                 marca isolada (fundo claro)
    logo/magnus-mark-dark.svg            marca isolada (fundo escuro)
    logo/magnus-logo-horizontal.svg      marca + wordmark + descritor
    logo/magnus-logo-horizontal-dark.svg idem, para fundo escuro
    icons/magnus-launcher-512.svg        ícone mestre do launcher
    icons/magnus-launcher-64.svg         ícone médio
    icons/magnus-launcher-32.svg         ícone reduzido (marca simplificada)
    icons/ui/*.svg                       nove ícones de interface, 24 px
    magnus-tokens.css                    paleta como custom properties

Os ícones de interface usam `stroke="currentColor"` no traço base, então
herdam a cor do contexto. Os acentos ciano, violeta e magenta são fixos
por carregarem significado físico.

## A marca

Núcleo denso, casca atmosférica (magenta) e linhas dipolares atravessando
essa casca, com o eixo inclinado 20° — o rotador oblíquo.

## Cores e o que significam

    #0B1020  fundo do aplicativo
    #141C33  painéis e cartões
    #7B4DF0  campo magnético, elementos primários
    #2ED3E0  modo ordinário (O), linhas de campo
    #F0479B  modo extraordinário (X), atmosfera
    #F0A63C  energia, avisos
    #E8EAF2  texto sobre fundo escuro
    #8A93B2  texto secundário

Ciano e magenta ficam reservados aos modos O e X em toda a interface e em
todos os gráficos. Não use esse par para outra distinção — a convenção
quebra. Magenta sobre violeta tem contraste baixo; separe os dois com um
neutro.

## Tipografia

Poppins Medium (500) no wordmark, com espaçamento entre letras de 9 em
62 px. O descritor vai em Poppins Regular, 17 px, espaçamento 1,6.
Nenhum dos SVGs converte o texto em curvas, então instale Poppins ou
rode `text-to-path` antes de distribuir.

## Área de respiro

Mantenha ao redor da marca uma margem livre igual ao raio do núcleo
(18 unidades na escala do arquivo de 200 px).
