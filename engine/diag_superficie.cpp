// Separa os candidatos do deficit de P1, em executavel exclusivo.
//
// Reimplementa a SOMA DE SUPERFICIE com a regra atual do motor, instrumentada
// por imagem e por faixa de psi_0, e oferece variantes que isolam UM candidato
// cada. A fisica nao e reimplementada: cada celula vai para a
// trace_spectral_contribution DO MOTOR, de modo que o unico que muda entre as
// variantes e COMO as celulas sao geradas e onde sao avaliadas.
//
// Desenho, casos e tolerancias em
// docs/incidente_resolucao_2026-10-04/pre_registro_candidatos_p1.md, commit
// ea2ae91, anterior a esta medida.
//
// AS VARIANTES, separadas de proposito para nao confundir os efeitos:
//   A  atual: bandas uniformes em theta, azimute por banda com
//      ceil(2 nlat sin theta), que SALTA entre bandas, avaliacao no ponto medio
//      em theta
//   B  A com avaliacao no CENTROIDE em cos theta, arccos((cos ta + cos tb)/2)
//   C  A com azimute GLOBAL unico, sem saltos entre bandas
//   D  B e C juntas
//
// O PORTAO: a variante A tem de reproduzir o omega por imagem do motor de
// producao com erro relativo <= 1e-12. Sem isso as variantes nao sao
// comparaveis.
//
// Uso: build/magnus_diag_superficie <args do motor> --variante A|B|C|D
//          --bandas N [--offset F] [--faixas-psi 40]
//          [--refino-por-fluxo FRACAO] [--refino-k 4]

#define MAGNUS_SEM_MAIN
#include "main.cpp"

namespace {

//: A `number()` do motor usa setprecision(10), o que limita qualquer comparacao
//: a 1e-10 relativo. O portao pre-registrado pedia 1e-12, abaixo disso, e era
//: inalcancavel por formatacao. Aqui a saida vai com 17 digitos.
std::string num17(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << v;
    return out.str();
}

struct Celula {
    double t0{}, t1{}, phi0{}, dphi{};
};

struct Pedido {
    std::string variante{"A"};
    int bandas{240};
    double offset{0.0};
    int faixas_psi{40};
    double refino_fracao{0.0};
    int refino_k{4};
    //: Fase de chegada. O motor usa 2 pi (it + 0.5)/time_bins, logo com
    //: --time-bins 1 ele roda em fase PI, nao em 0. Para o portao de
    //: reproducao isso tem de casar: a fisica e estatica, mas o ALINHAMENTO da
    //: malha com a linha de visada muda com a fase, e e justamente o que o
    //: pre-registro trata como alinhamento.
    double fase{0.0};
    //: Quantas fases MEDIAR. O motor grampeia time_bins em [16, 1000], logo o
    //: omega que ele relata e a MEDIA sobre 16 fases, em 2 pi (it+0.5)/16, e
    //: nao um instantaneo. O portao de reproducao exige o mesmo. Com --fases 1
    //: vale a fase unica de --fase, que e o modo do estudo de alinhamento.
    int fases{16};
};

Pedido le_pedido(int argc, char** argv, std::vector<char*>& restante) {
    Pedido p;
    restante.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--variante" && i + 1 < argc) p.variante = argv[++i];
        else if (key == "--bandas" && i + 1 < argc) p.bandas = std::stoi(argv[++i]);
        else if (key == "--offset" && i + 1 < argc) p.offset = std::stod(argv[++i]);
        else if (key == "--faixas-psi" && i + 1 < argc) p.faixas_psi = std::stoi(argv[++i]);
        else if (key == "--refino-por-fluxo" && i + 1 < argc) p.refino_fracao = std::stod(argv[++i]);
        else if (key == "--refino-k" && i + 1 < argc) p.refino_k = std::stoi(argv[++i]);
        else if (key == "--fase" && i + 1 < argc) p.fase = std::stod(argv[++i]);
        else if (key == "--fases" && i + 1 < argc) p.fases = std::stoi(argv[++i]);
        else restante.push_back(argv[i]);
    }
    if (p.variante != "A" && p.variante != "B" && p.variante != "C" && p.variante != "D") {
        throw std::runtime_error("--variante deve ser A, B, C ou D");
    }
    if (p.bandas < 8 || p.bandas > 20000) throw std::runtime_error("--bandas fora de [8, 20000]");
    return p;
}

//: Azimute GLOBAL que iguala a contagem de celulas da regra atual, para que a
//: comparacao nao seja entre malhas de tamanhos diferentes. A soma da regra
//: atual e sum_l max(4, ceil(2 nlat sin theta_l)); dividida por nlat da o
//: azimute medio.
int azimute_global(int nlat) {
    long total = 0;
    for (int il = 0; il < nlat; ++il) {
        const double tmid = pi * (il + 0.5) / nlat;
        total += std::max(4, static_cast<int>(std::ceil(2.0 * nlat * std::sin(tmid))));
    }
    return std::max(4, static_cast<int>(std::llround(
        static_cast<double>(total) / static_cast<double>(nlat))));
}

std::vector<Celula> gera_celulas(const Pedido& p) {
    const int nlat = std::max(8, p.bandas);
    const bool azimute_sem_salto = (p.variante == "C" || p.variante == "D");
    const int nphi_global = azimute_sem_salto ? azimute_global(nlat) : 0;
    std::vector<Celula> celulas;
    for (int il = 0; il < nlat; ++il) {
        const double t0 = pi * il / nlat;
        const double t1 = pi * (il + 1) / nlat;
        const double tmid = 0.5 * (t0 + t1);
        const int azimuths = azimute_sem_salto ? nphi_global
            : std::max(4, static_cast<int>(std::ceil(2.0 * nlat * std::sin(tmid))));
        const double dph = 2.0 * pi / azimuths;
        for (int ia = 0; ia < azimuths; ++ia) {
            const double ph0 = 2.0 * pi * ia / azimuths + p.offset * dph;
            celulas.push_back({t0, t1, ph0, dph});
        }
    }
    return celulas;
}

//: Ponto de avaliacao em theta: ponto medio (A, C) ou centroide da medida
//: d(cos theta) (B, D). A area e EXATA nas quatro.
double theta_de_avaliacao(const Pedido& p, double ta, double tb) {
    if (p.variante == "B" || p.variante == "D") {
        return std::acos(clamp_unit(0.5 * (std::cos(ta) + std::cos(tb))));
    }
    return 0.5 * (ta + tb);
}

struct Acumulado {
    double total{0.0};
    std::vector<double> por_imagem;
    std::vector<double> por_faixa_psi;   // imagem 0 e 1 somadas
    std::vector<double> por_faixa_psi_i0;
    std::vector<double> por_faixa_psi_i1;
    long avaliacoes{0};
    long aceitos{0};
};

//: Contribuicao de UMA (sub)celula, acumulada. Devolve o ganho em omega, para o
//: indicador do refino por fluxo.
double contribui(const Config& cfg, const RayTable& rays, double u,
                 const Pedido& p, const Celula& c, int ordem,
                 double fase_de_chegada, const Vec3& observador,
                 Acumulado* acc) {
    const double tm = theta_de_avaliacao(p, c.t0, c.t1);
    const double pm = c.phi0 + 0.5 * c.dphi;
    const double area = (std::cos(c.t0) - std::cos(c.t1)) * c.dphi;
    SpectralSurfaceSample s{};
    s.normal = {std::sin(tm) * std::cos(pm), std::sin(tm) * std::sin(pm),
                std::cos(tm)};
    s.weight = area;
    s.t0 = c.t0; s.t1 = c.t1; s.phi0 = c.phi0; s.dphi = c.dphi;
    s.temperature_mk = cfg.base_temperature_mk;
    RayContribution r{};
    if (acc) ++acc->avaliacoes;
    if (!trace_spectral_contribution(cfg, rays, u, s, fase_de_chegada, ordem, r)) {
        return 0.0;
    }
    if (acc) {
        ++acc->aceitos;
        acc->total += r.solid_angle_sr;
        if (ordem < static_cast<int>(acc->por_imagem.size())) {
            acc->por_imagem[static_cast<std::size_t>(ordem)] += r.solid_angle_sr;
        }
        const Vec3 n = rotate_about_z(s.normal, r.emission_phase);
        const double psi0 = std::acos(clamp_unit(dot(n, observador)));
        const int nf = static_cast<int>(acc->por_faixa_psi.size());
        int faixa = static_cast<int>(psi0 / pi * nf);
        faixa = std::clamp(faixa, 0, nf - 1);
        acc->por_faixa_psi[static_cast<std::size_t>(faixa)] += r.solid_angle_sr;
        (ordem == 0 ? acc->por_faixa_psi_i0 : acc->por_faixa_psi_i1)
            [static_cast<std::size_t>(faixa)] += r.solid_angle_sr;
    }
    return r.solid_angle_sr;
}

double soma_das_subcelulas(const Config& cfg, const RayTable& rays, double u,
                           const Pedido& p, const Celula& c, int ordem,
                           double fase, const Vec3& obs, int k,
                           Acumulado* acc) {
    double soma = 0.0;
    for (int iu = 0; iu < k; ++iu) {
        for (int iv = 0; iv < k; ++iv) {
            Celula sub{c.t0 + (c.t1 - c.t0) * iu / k,
                       c.t0 + (c.t1 - c.t0) * (iu + 1) / k,
                       c.phi0 + c.dphi * iv / k, c.dphi / k};
            soma += contribui(cfg, rays, u, p, sub, ordem, fase, obs, acc);
        }
    }
    return soma;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<char*> restante;
        const Pedido pedido = le_pedido(argc, argv, restante);
        const Config cfg = parse_args(static_cast<int>(restante.size()),
                                      restante.data());
        const double u = 2.0 * gm_sun_over_c2_km * cfg.mass_solar / cfg.radius_km;
        if (!(u > 0.0 && u < 2.0 / 3.0)) {
            throw std::runtime_error("o controle vale para u < 2/3; recebi u = "
                                     + std::to_string(u));
        }
        const RayTable rays(u, cfg.ray_table_size);
        const double incl = deg(cfg.inclination_deg);
        const Vec3 observador{std::sin(incl), 0.0, std::cos(incl)};
        const std::vector<Celula> celulas = gera_celulas(pedido);

        Acumulado acc;
        acc.por_imagem.assign(static_cast<std::size_t>(std::max(1, cfg.max_images)), 0.0);
        acc.por_faixa_psi.assign(static_cast<std::size_t>(pedido.faixas_psi), 0.0);
        acc.por_faixa_psi_i0.assign(static_cast<std::size_t>(pedido.faixas_psi), 0.0);
        acc.por_faixa_psi_i1.assign(static_cast<std::size_t>(pedido.faixas_psi), 0.0);

        // As fases a mediar. Uma -> a de --fase; mais de uma -> a MESMA grade do
        // motor, 2 pi (it + 0.5)/N, porque ele grampeia time_bins em 16.
        std::vector<double> fases;
        if (pedido.fases <= 1) {
            fases.push_back(pedido.fase);
        } else {
            for (int it = 0; it < pedido.fases; ++it) {
                fases.push_back(2.0 * pi * (it + 0.5) / pedido.fases);
            }
        }
        const double peso_de_fase = 1.0 / static_cast<double>(fases.size());

        // Refino guiado pela CONTRIBUICAO AO FLUXO, nao pelo jacobiano: o
        // indicador e |contribuicao da celula - soma das subdivisoes|, que e o
        // que o revisor exigiu, porque variacao do jacobiano sozinha pode nao
        // representar o erro do fluxo.
        std::vector<char> refinar(celulas.size(), 0);  // ver fases acima
        long avaliacoes_do_indicador = 0;
        if (pedido.refino_fracao > 0.0) {
            std::vector<double> indicador(celulas.size(), 0.0);
            for (std::size_t ic = 0; ic < celulas.size(); ++ic) {
                for (int ordem = 0; ordem < cfg.max_images; ++ordem) {
                    Acumulado lixo;
                    lixo.por_imagem.assign(1, 0.0);
                    lixo.por_faixa_psi.assign(1, 0.0);
                    lixo.por_faixa_psi_i0.assign(1, 0.0);
                    lixo.por_faixa_psi_i1.assign(1, 0.0);
                    // O indicador usa a PRIMEIRA fase, declarado: marcar
                    // celulas tem de ser decisao da malha, nao da fase, senao o
                    // conjunto refinado muda a cada fase e o custo explode.
                    const double grosso = contribui(cfg, rays, u, pedido,
                                                    celulas[ic], ordem, fases[0],
                                                    observador, &lixo);
                    const double fino = soma_das_subcelulas(
                        cfg, rays, u, pedido, celulas[ic], ordem, fases[0],
                        observador, pedido.refino_k, &lixo);
                    indicador[ic] += std::abs(grosso - fino);
                    avaliacoes_do_indicador += lixo.avaliacoes;
                }
            }
            std::vector<std::size_t> ordem_idx(celulas.size());
            for (std::size_t i = 0; i < ordem_idx.size(); ++i) ordem_idx[i] = i;
            const std::size_t quantas = static_cast<std::size_t>(
                pedido.refino_fracao * static_cast<double>(celulas.size()));
            std::partial_sort(ordem_idx.begin(),
                              ordem_idx.begin() + std::min(quantas, ordem_idx.size()),
                              ordem_idx.end(),
                              [&](std::size_t a, std::size_t b) {
                                  return indicador[a] > indicador[b];
                              });
            for (std::size_t i = 0; i < std::min(quantas, ordem_idx.size()); ++i) {
                refinar[ordem_idx[i]] = 1;
            }
        }

        for (const double fase : fases) {
            for (std::size_t ic = 0; ic < celulas.size(); ++ic) {
                for (int ordem = 0; ordem < cfg.max_images; ++ordem) {
                    if (refinar[ic]) {
                        soma_das_subcelulas(cfg, rays, u, pedido, celulas[ic],
                                            ordem, fase, observador,
                                            pedido.refino_k, &acc);
                    } else {
                        contribui(cfg, rays, u, pedido, celulas[ic], ordem, fase,
                                  observador, &acc);
                    }
                }
            }
        }
        // Media sobre as fases, exatamente como o motor faz ao dividir por
        // cfg.time_bins ao acumular.
        acc.total *= peso_de_fase;
        for (auto& v : acc.por_imagem) v *= peso_de_fase;
        for (auto& v : acc.por_faixa_psi) v *= peso_de_fase;
        for (auto& v : acc.por_faixa_psi_i0) v *= peso_de_fase;
        for (auto& v : acc.por_faixa_psi_i1) v *= peso_de_fase;

        std::cout << "{\"status\":\"ok\",\"produto\":\"diag_superficie\""
                  << ",\"variante\":\"" << pedido.variante << "\""
                  << ",\"bandas\":" << pedido.bandas
                  << ",\"offset\":" << num17(pedido.offset)
                  << ",\"fase\":" << num17(pedido.fase)
                  << ",\"fases\":" << fases.size()
                  << ",\"inclinacao_deg\":" << num17(cfg.inclination_deg)
                  << ",\"compactness\":" << num17(u)
                  << ",\"psi_max\":" << num17(rays.psi_max())
                  << ",\"celulas\":" << celulas.size()
                  << ",\"celulas_refinadas\":"
                  << std::count(refinar.begin(), refinar.end(), 1)
                  << ",\"refino_k\":" << pedido.refino_k
                  << ",\"avaliacoes\":" << acc.avaliacoes
                  << ",\"avaliacoes_do_indicador\":" << avaliacoes_do_indicador
                  << ",\"aceitos\":" << acc.aceitos
                  << ",\"omega_total\":" << num17(acc.total)
                  << ",\"omega_por_imagem\":[";
        for (std::size_t i = 0; i < acc.por_imagem.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(acc.por_imagem[i]);
        }
        std::cout << "],\"omega_por_faixa_psi\":[";
        for (std::size_t i = 0; i < acc.por_faixa_psi.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(acc.por_faixa_psi[i]);
        }
        std::cout << "],\"omega_por_faixa_psi_imagem0\":[";
        for (std::size_t i = 0; i < acc.por_faixa_psi_i0.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(acc.por_faixa_psi_i0[i]);
        }
        std::cout << "],\"omega_por_faixa_psi_imagem1\":[";
        for (std::size_t i = 0; i < acc.por_faixa_psi_i1.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(acc.por_faixa_psi_i1[i]);
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MAGNUS diag superficie: " << error.what() << '\n';
        return 2;
    }
}
