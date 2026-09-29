#!/usr/bin/env python3
"""Curvas massa-raio de equacoes de estado modernas, por integracao TOV.

SLy, APR4 e MPA1 pelos politropos por partes de Read et al. (2009), uma estrela
de quarks auto-ligada (MIT bag) e a tabela BBB2, esta ultima como verificacao
independente do integrador.

Versionado a partir de exploracoes/obsoletos/eos_mxr_2026-09-10/eos_mr.py, com
duas mudancas. A primeira: aquele arquivo declarava a crosta por trechos da
Tabela II de Read et al. e NAO a usava, o que a auditoria de 29/09/2026 apontou
no achado 7. As constantes mortas foram removidas, e fica explicito o que o
codigo faz: a crosta e um politropo UNICO de indice efetivo 1,35, ancorado por
continuidade de pressao na transicao. Isso reproduz as massas maximas a melhor
que 0,03 massas solares, e NAO e preciso o bastante para certificar raios na
escala que uma exclusao de equacao de estado exigiria. A segunda: o modulo nao
faz mais nada ao ser importado, o calculo esta em curvas().
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- constantes (SI) ----
G = 6.67430e-11
c = 2.99792458e8
Msun = 1.98892e30          # kg
Rsun_km = G * Msun / c**2 / 1e3   # GM_sun/c^2 em km = 1.4766 km
MEV_FM3_TO_PA = 1.602176634e32    # 1 MeV/fm^3 -> Pa (=J/m^3)

# =========================================================================
# EoS: cada uma devolve energy density eps [J/m^3] dada a pressao P [Pa]
# =========================================================================

RHO1, RHO2 = 10**14.7, 10**15.0   # g/cm^3, densidades divisorias do nucleo
# (logp1 em dyn/cm^2 a rho1; Gamma1,2,3)
CORE_PARAMS = {
    "SLy":  (34.384, 3.005, 2.988, 2.851),
    "APR4": (34.269, 2.830, 3.445, 3.348),
    "MPA1": (34.495, 3.446, 3.572, 2.887),
}

RHO0 = 1.4e14        # g/cm^3, transicao crosta-nucleo padrao (~0.5 rho_nuc)
GAMMA_CRUST = 1.35   # indice adiabatico efetivo da crosta

def make_pp_eos(name):
    logp1, g1, g2, g3 = CORE_PARAMS[name]
    p1 = 10**logp1                       # dyn/cm^2 a RHO1
    K1 = p1 / RHO1**g1
    K2 = K1 * RHO1**(g1 - g2)
    K3 = K2 * RHO2**(g2 - g3)
    # crosta: politropo unico ancorado por continuidade de pressao em RHO0
    p0 = K1 * RHO0**g1                    # pressao do nucleo em RHO0
    Kc = p0 / RHO0**GAMMA_CRUST
    # tramos (rho_max, K, Gamma) em cgs (rho g/cm^3, p dyn/cm^2)
    segs = [(RHO0, Kc, GAMMA_CRUST), (RHO1, K1, g1), (RHO2, K2, g2), (np.inf, K3, g3)]
    # constantes 'a' de energia (continuidade de eps); crosta parte de a=0
    c2 = (c * 100)**2                    # (cm/s)^2
    a = [0.0]
    for i in range(len(segs) - 1):
        rb, Ki, Gi = segs[i][0], segs[i][1], segs[i][2]
        eps_b = (1 + a[i]) * rb * c2 + Ki / (Gi - 1) * rb**Gi
        Kn, Gn = segs[i+1][1], segs[i+1][2]
        a.append(eps_b / (rb * c2) - 1 - Kn / (Gn - 1) * rb**(Gn - 1) / c2)
    # limites de pressao de cada tramo (ascendente)
    p_bounds = [K * (rmax**G_) if np.isfinite(rmax) else np.inf
                for (rmax, K, G_) in segs]

    def eps_of_p(P):  # P [Pa] -> eps [J/m^3]
        p_cgs = np.atleast_1d(P) * 10.0                 # Pa -> dyn/cm^2
        out = np.empty_like(p_cgs, dtype=float)
        for j, pc in enumerate(p_cgs):
            i = 0
            while i < len(segs) - 1 and pc > p_bounds[i]:
                i += 1
            K, G_ = segs[i][1], segs[i][2]
            rho = (pc / K)**(1.0 / G_)
            out[j] = (1 + a[i]) * rho * c2 + K / (G_ - 1) * rho**G_
        return out * 0.1                                 # erg/cm^3 -> J/m^3
    return eps_of_p

# ---- MIT bag (estrela de quarks) ----
def make_mit_eos(B14_MeV=145.0):
    B = (B14_MeV**4 / 197.33**3) * MEV_FM3_TO_PA   # bag constant em Pa
    def eps_of_p(P):     # p = (eps - 4B)/3  ->  eps = 3p + 4B
        return 3.0 * np.atleast_1d(P) + 4.0 * B
    return eps_of_p, 4.0 * B   # eps na superficie = 4B

# ---- BBB2 (tabela real baixada, LAL geometrizado m^-2) ----
def make_table_eos(path):
    d = np.loadtxt(path, comments="#")
    p_geo, e_geo = d[:, 0], d[:, 1]
    fac = c**4 / G        # geometrizado (m^-2) -> SI (Pa)
    p_si, e_si = p_geo * fac, e_geo * fac
    lp, le = np.log(p_si), np.log(e_si)
    def eps_of_p(P):
        return np.exp(np.interp(np.log(np.atleast_1d(P)), lp, le))
    return eps_of_p, p_si.min(), p_si.max()

# =========================================================================
# TOV (SI, RK4 em r [m])
# =========================================================================
def tov(eps_of_p, Pc, p_surf):
    def derivs(r, y):
        P, m = y
        if P <= p_surf:
            return None
        eps = float(np.ravel(eps_of_p(P))[0])
        rho_e = eps / c**2
        denom = r * (r - 2 * G * m / c**2)
        if denom <= 0:
            return None
        dP = -G * (rho_e + P / c**2) * (m + 4 * np.pi * r**3 * P / c**2) / denom
        dm = 4 * np.pi * r**2 * rho_e
        return np.array([dP, dm])
    r = 1.0
    eps_c = float(np.ravel(eps_of_p(Pc))[0])
    m = 4.0 / 3.0 * np.pi * r**3 * eps_c / c**2
    y = np.array([Pc, m])
    dr = 20.0
    while True:
        k1 = derivs(r, y)
        if k1 is None: break
        k2 = derivs(r + dr/2, y + dr/2*k1)
        if k2 is None: break
        k3 = derivs(r + dr/2, y + dr/2*k2)
        if k3 is None: break
        k4 = derivs(r + dr, y + dr*k3)
        if k4 is None: break
        y = y + dr/6*(k1 + 2*k2 + 2*k3 + k4)
        r += dr
        if y[0] <= p_surf or r > 5e4:
            break
    return r/1e3, y[1]/Msun    # R [km], M [Msun]

def mr_curve(eps_of_p, pc_lo, pc_hi, p_surf, n=60, rmin=5.0, mmin=0.1):
    Rs, Ms = [], []
    for Pc in np.logspace(np.log10(pc_lo), np.log10(pc_hi), n):
        R, M = tov(eps_of_p, Pc, p_surf)
        if rmin < R < 30 and mmin < M < 3.5:
            Rs.append(R); Ms.append(M)
    return np.array(Rs), np.array(Ms)


# =========================================================================
LITERATURA = {"SLy": (2.05, 11.7), "APR4": (2.20, 11.4), "MPA1": (2.46, 12.4),
              "MIT bag (145 MeV)": (2.01, 10.9), "BBB2 (tabela)": (1.92, 11.1)}
TABELA_BBB2 = "tabelas/eos/BBB2.dat"


def curvas(raiz="."):
    """Devolve {nome: (R em km, M em massas solares)} no ramo completo."""
    from pathlib import Path
    saida = {}
    for nome in ("SLy", "APR4", "MPA1"):
        saida[nome] = mr_curve(make_pp_eos(nome), 5e32, 3e36, 1e27, n=80)
    eos_q, _ = make_mit_eos(145.0)
    saida["MIT bag (145 MeV)"] = mr_curve(eos_q, 2e32, 2e36, 1e27, n=120,
                                          rmin=2.0, mmin=0.03)
    eos_b, pmin_b, pmax_b = make_table_eos(str(Path(raiz) / TABELA_BBB2))
    saida["BBB2 (tabela)"] = mr_curve(eos_b, pmin_b * 5, pmax_b * 0.98,
                                      pmin_b * 1.5, n=70)
    return saida


def ramo_estavel(R, M):
    """So ate a massa maxima: acima dela a estrela e instavel."""
    fim = int(np.argmax(M)) + 1
    return R[:fim], M[:fim]


def compactness(R, M):
    """u = 2GM/Rc^2 = R_s/R, a mesma convencao do motor, onde z = 1/sqrt(1-u) - 1."""
    return 2.0 * Rsun_km * M / R


def valida(cs):
    linhas = []
    for nome, (R, M) in cs.items():
        Re, Me = ramo_estavel(R, M)
        r14 = float(np.interp(1.4, Me, Re)) if Me.max() >= 1.4 else float("nan")
        lm, lr = LITERATURA.get(nome, (float("nan"), float("nan")))
        linhas.append(dict(eos=nome, m_max=float(Me.max()), m_max_literatura=lm,
                           r_1p4=r14, r_1p4_literatura=lr,
                           u_max=float(compactness(Re, Me).max())))
    return linhas


if __name__ == "__main__":
    import json
    print(json.dumps(valida(curvas()), indent=2))
