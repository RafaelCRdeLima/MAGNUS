"""O solucionador de atmosfera do MAGNUS.

Roda FORA do laço de verossimilhança: produz uma tabela de intensidade, uma vez,
que o motor em C++ depois consome. Por isso é Python com numpy — a velocidade
não paga a diferença de tempo para depurar física nova.
"""
