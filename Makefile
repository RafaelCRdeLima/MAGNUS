CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic
ENGINE := build/magnus_engine

.PHONY: all engine auditoria test clean

all: engine

engine: $(ENGINE)

# O motor partiu do tracado de raios do PULSARIS e ganhou o backend de
# atmosfera lido de tabela, o eixo de campo (lg B, theta_B) e o dipolo de
# Schwarzschild.
$(ENGINE): engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

# O aferidor do estágio 0: lê os 26 gabaritos, mede sigma T^4 arquivo a arquivo
# e diz quais servem de referência. Sai com código 1 se algum reprovar.
auditoria:
	python3 scripts/auditoria_gabaritos.py

# O portao do leitor de tabela: zeros reproduzem o corpo negro, o eixo de mu
# reproduz o feixe analitico do proprio motor, e theta_B sai do dipolo.
test: engine
	python3 -m unittest discover -s tests -v

clean:
	rm -f $(ENGINE) build/*.magnus
