CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic
ENGINE := build/magnus_engine
#: O motor com OpenMP vai para OUTRO binario, de proposito: o laco de fases
#: paralelizado deve ser COMPARADO com o binario validado, e substitui-lo antes
#: dessa comparacao tiraria a referencia de onde ela sai. `make engine-omp`
#: produz build/magnus_engine_omp, e a producao segue usando build/magnus_engine
#: ate a identidade da previsao estar medida.
ENGINE_OMP := build/magnus_engine_omp

DIAG_IMAGEM := build/magnus_diag_imagem
DIAG_SUPERFICIE := build/magnus_diag_superficie
DIAG_ANTIPODAL := build/magnus_diag_antipodal

.PHONY: all engine engine-omp diag-imagem diag-superficie diag-antipodal auditoria test clean

all: engine

engine: $(ENGINE)

engine-omp: $(ENGINE_OMP)

# O motor partiu do tracado de raios do PULSARIS e ganhou o backend de
# atmosfera lido de tabela, o eixo de campo (lg B, theta_B) e o dipolo de
# Schwarzschild.
$(ENGINE): engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

#: Mesma fonte, mesmas flags, mais -fopenmp. Nao se acrescenta -ffast-math nem
#: -funsafe-math-optimizations: elas autorizam reassociar somas de ponto
#: flutuante, e e justamente a ordem das somas que faz a previsao ser identica.
$(ENGINE_OMP): engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

# EXECUTAVEL EXCLUSIVO do diagnostico de P1: controle independente no plano da
# imagem. Compila o MESMO main.cpp com MAGNUS_SEM_MAIN, para reaproveitar a
# Config, a RayTable e a intensidade de Planck, e so trocar a QUADRATURA. O
# binario de producao e compilado sem a macro e sai byte a byte identico.
diag-imagem: $(DIAG_IMAGEM)

$(DIAG_IMAGEM): engine/diag_plano_imagem.cpp engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I engine $< -o $@

# EXECUTAVEL EXCLUSIVO da separacao dos candidatos de P1: reimplementa a soma de
# superficie com a regra do motor, instrumentada, e oferece as variantes que
# isolam um candidato cada. A fisica vem da trace_spectral_contribution do motor.
diag-superficie: $(DIAG_SUPERFICIE)

$(DIAG_SUPERFICIE): engine/diag_superficie.cpp engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I engine $< -o $@

# EXECUTAVEL EXCLUSIVO do tratamento antipodal: malha em coordenadas CENTRADAS
# NO OBSERVADOR, onde o sin(psi_0) do elemento de area cancela o 1/sin(psi_0) do
# jacobiano. Binario separado de proposito, para nao recompilar o
# diag-superficie enquanto a grade pre-registrada roda com ele.
diag-antipodal: $(DIAG_ANTIPODAL)

$(DIAG_ANTIPODAL): engine/diag_antipodal.cpp engine/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I engine $< -o $@

# O aferidor do estágio 0: lê os 26 gabaritos, mede sigma T^4 arquivo a arquivo
# e diz quais servem de referência. Sai com código 1 se algum reprovar.
auditoria:
	python3 scripts/auditoria_gabaritos.py

# O portao do leitor de tabela: zeros reproduzem o corpo negro, o eixo de mu
# reproduz o feixe analitico do proprio motor, e theta_B sai do dipolo.
test: engine
	python3 -m unittest discover -s tests -v

clean:
	rm -f $(ENGINE) $(DIAG_IMAGEM) $(DIAG_SUPERFICIE) $(DIAG_ANTIPODAL) build/*.magnus
