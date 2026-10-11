// Trata a vizinhanca antipodal em coordenadas CENTRADAS NO OBSERVADOR.
//
// O QUE O REVISOR CORRIGIU, em 06/10/2026, e que define este arquivo:
//
//   "A singularidade pode ser da representacao: se o jacobiano contem
//    1/sin psi_0, o elemento de area em coordenadas centradas no observador
//    contem sin psi_0 dpsi_0 dphi. O produto pode permanecer integravel mesmo
//    quando o jacobiano isolado diverge."
//
// E a derivacao confirma. A contribuicao de um elemento de superficie e
//
//   dOmega_imagem = (R^2/(D^2(1-u))) cos(a) [sin(a) |da/dpsi| / sin(psi_0)]
//                   dOmega_superficie
//
// e em coordenadas centradas no observador dOmega_superficie =
// sin(psi_0) dpsi_0 dphi, de modo que
//
//   dOmega_imagem = (R^2/(D^2(1-u))) cos(a) sin(a) |da/dpsi| dpsi_0 dphi
//
// SEM sin(psi_0). O fator divergente cancela EXATAMENTE. E como
// cos(a) sin(a) da = (1/2) d(sin^2 a) e b^2 = R^2 sin^2(a)/(1-u), o produto e
// (1-u)/(2R^2) d(b^2) dphi, isto e o elemento de area da IMAGEM. Logo o
// integrando, nessas coordenadas, e regular na antipoda.
//
// A malha em (theta, phi) do motor nao esta alinhada com psi_0, e por isso
// avalia 1/sin(psi_0) no ponto medio da celula contra uma area que integra
// sin(psi_0) exatamente, sem que os dois se cancelem dentro da celula. Aqui a
// malha E em psi_0, e o cancelamento fica alinhado.
//
// TAMBEM MEDE O CLAMP, porque o revisor exigiu: "para responsabilizar o clamp,
// precisamos medir quantos pontos o acionam e quanto contribuem ao fluxo". O
// motor faz sin_psi0 = max(sin(psi_0), 1e-12); aqui contamos os pontos abaixo de
// varios limiares e somamos o que eles contribuem.
//
// Uso: build/magnus_diag_antipodal <args do motor>
//        --modo esfera|capa  [--capa-graus G]  --psi-bandas N  [--fases 16]
//        [--phi-por-banda]  [--centroide]

#define MAGNUS_SEM_MAIN
#include "main.cpp"

namespace {

std::string num17(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << v;
    return out.str();
}

struct Pedido {
    std::string modo{"esfera"};
    double capa_graus{5.0};
    int psi_bandas{240};
    int fases{16};
    bool phi_por_banda{true};
    bool centroide{false};
    //: Insere uma QUEBRA de banda exatamente em psi_0 = 2 pi - psi_max, que e
    //: onde a imagem 1 liga, e em psi_max quando psi_max < pi, que e onde a
    //: imagem 0 desliga. Sao NOS do integrando: a aceitacao troca ali, e uma
    //: banda que atravessa o no da erro de primeira ordem. Com a malha alinhada
    //: a psi_0 o no e um ponto so, e alinhar a borda da banda a ele e exato.
    bool quebra{false};
};

Pedido le_pedido(int argc, char** argv, std::vector<char*>& restante) {
    Pedido p;
    restante.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--modo" && i + 1 < argc) p.modo = argv[++i];
        else if (k == "--capa-graus" && i + 1 < argc) p.capa_graus = std::stod(argv[++i]);
        else if (k == "--psi-bandas" && i + 1 < argc) p.psi_bandas = std::stoi(argv[++i]);
        else if (k == "--fases" && i + 1 < argc) p.fases = std::stoi(argv[++i]);
        else if (k == "--phi-global") p.phi_por_banda = false;
        else if (k == "--centroide") p.centroide = true;
        else if (k == "--quebra") p.quebra = true;
        else restante.push_back(argv[i]);
    }
    if (p.modo != "esfera" && p.modo != "capa") {
        throw std::runtime_error("--modo deve ser esfera ou capa");
    }
    if (p.psi_bandas < 2 || p.psi_bandas > 40000) {
        throw std::runtime_error("--psi-bandas fora de [2, 40000]");
    }
    return p;
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
            throw std::runtime_error("o controle vale para u < 2/3");
        }
        const RayTable rays(u, cfg.ray_table_size);
        const double incl = deg(cfg.inclination_deg);
        const Vec3 obs{std::sin(incl), 0.0, std::cos(incl)};
        // Base ortonormal perpendicular ao observador, para montar a malha em
        // (psi_0, phi_obs).
        Vec3 e1 = std::abs(obs.z) < 0.9 ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
        e1 = unit(e1 + (-dot(e1, obs)) * obs);
        const Vec3 e2 = unit(Vec3{obs.y * e1.z - obs.z * e1.y,
                                  obs.z * e1.x - obs.x * e1.z,
                                  obs.x * e1.y - obs.y * e1.x});

        const double psi_lo = pedido.modo == "capa"
            ? pi - deg(pedido.capa_graus) : 0.0;
        const double psi_hi = pi;
        const int nb = pedido.psi_bandas;

        // Os NOS do integrando em psi_0, onde a aceitacao troca:
        //   imagem 1 liga em psi_0 = 2 pi - psi_max, quando psi_max > pi
        //   imagem 0 desliga em psi_0 = psi_max, quando psi_max < pi
        // Com --quebra, as bordas das bandas sao alinhadas a eles.
        std::vector<double> nos{psi_lo, psi_hi};
        if (pedido.quebra) {
            const double pm_ = rays.psi_max();
            for (const double no : {2.0 * pi - pm_, pm_}) {
                if (no > psi_lo + 1.0e-12 && no < psi_hi - 1.0e-12) {
                    nos.push_back(no);
                }
            }
            std::sort(nos.begin(), nos.end());
        }
        // Distribui as bandas entre os subintervalos, proporcional ao tamanho.
        std::vector<std::pair<double, double>> bandas;
        for (std::size_t i = 0; i + 1 < nos.size(); ++i) {
            const double a0 = nos[i], a1 = nos[i + 1];
            const int quantas = std::max(1, static_cast<int>(std::llround(
                nb * (a1 - a0) / (psi_hi - psi_lo))));
            for (int j = 0; j < quantas; ++j) {
                bandas.emplace_back(a0 + (a1 - a0) * j / quantas,
                                    a0 + (a1 - a0) * (j + 1) / quantas);
            }
        }

        std::vector<double> fases;
        if (pedido.fases <= 1) fases.push_back(0.0);
        else for (int it = 0; it < pedido.fases; ++it)
            fases.push_back(2.0 * pi * (it + 0.5) / pedido.fases);
        const double peso_fase = 1.0 / static_cast<double>(fases.size());

        double total = 0.0;
        std::vector<double> por_imagem(static_cast<std::size_t>(
            std::max(1, cfg.max_images)), 0.0);
        long avaliacoes = 0, aceitos = 0;
        // Contabilidade do CLAMP: quantos pontos tem sin(psi_0) abaixo de cada
        // limiar e quanto somam em omega.
        const std::vector<double> limiares{1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0e-2};
        std::vector<long> conta_clamp(limiares.size(), 0);
        std::vector<double> omega_clamp(limiares.size(), 0.0);

        for (const double fase : fases) {
            for (const auto& banda : bandas) {
                const double pa = banda.first;
                const double pb = banda.second;
                const double pm = pedido.centroide
                    ? std::acos(clamp_unit(0.5 * (std::cos(pa) + std::cos(pb))))
                    : 0.5 * (pa + pb);
                const int azimutes = pedido.phi_por_banda
                    ? std::max(4, static_cast<int>(std::ceil(2.0 * nb * std::sin(pm))))
                    : std::max(4, static_cast<int>(std::ceil(2.0 * nb)));
                // MEDIDO: o omega sai identico a 12 digitos com qualquer
                // contagem de azimutes, porque o integrando do controle e
                // axissimetrico em torno do eixo do observador. Nessas
                // coordenadas o problema e 1-D, e o azimute nao entra no erro.
                const double dph = 2.0 * pi / azimutes;
                const double area = (std::cos(pa) - std::cos(pb)) * dph;
                for (int ia = 0; ia < azimutes; ++ia) {
                    const double ph = dph * (ia + 0.5);
                    // normal no instante da chegada, a psi_0 = pm do observador
                    const Vec3 n_agora = std::cos(pm) * obs +
                        (std::sin(pm) * std::cos(ph)) * e1 +
                        (std::sin(pm) * std::sin(ph)) * e2;
                    // o motor gira a normal por emission_phase, que para
                    // periodo longo e a fase de chegada; desgiramos antes
                    SpectralSurfaceSample s{};
                    s.normal = rotate_about_z(n_agora, -fase);
                    s.weight = area;
                    s.temperature_mk = cfg.base_temperature_mk;
                    for (int ordem = 0; ordem < cfg.max_images; ++ordem) {
                        RayContribution r{};
                        ++avaliacoes;
                        if (!trace_spectral_contribution(cfg, rays, u, s, fase,
                                                         ordem, r)) continue;
                        ++aceitos;
                        total += r.solid_angle_sr;
                        por_imagem[static_cast<std::size_t>(ordem)] += r.solid_angle_sr;
                        const double sp = std::sin(pm);
                        for (std::size_t il = 0; il < limiares.size(); ++il) {
                            if (sp < limiares[il]) {
                                ++conta_clamp[il];
                                omega_clamp[il] += r.solid_angle_sr;
                            }
                        }
                    }
                }
            }
        }
        total *= peso_fase;
        for (auto& v : por_imagem) v *= peso_fase;
        for (auto& v : omega_clamp) v *= peso_fase;

        std::cout << "{\"status\":\"ok\",\"produto\":\"diag_antipodal\""
                  << ",\"modo\":\"" << pedido.modo << "\""
                  << ",\"capa_graus\":" << num17(pedido.modo == "capa"
                                                 ? pedido.capa_graus : 180.0)
                  << ",\"psi_bandas\":" << nb
                  << ",\"phi_por_banda\":" << (pedido.phi_por_banda ? "true" : "false")
                  << ",\"centroide\":" << (pedido.centroide ? "true" : "false")
                  << ",\"quebra\":" << (pedido.quebra ? "true" : "false")
                  << ",\"bandas_efetivas\":" << bandas.size()
                  << ",\"fases\":" << fases.size()
                  << ",\"compactness\":" << num17(u)
                  << ",\"psi_max\":" << num17(rays.psi_max())
                  << ",\"inclinacao_deg\":" << num17(cfg.inclination_deg)
                  << ",\"avaliacoes\":" << avaliacoes
                  << ",\"aceitos\":" << aceitos
                  << ",\"omega_total\":" << num17(total)
                  << ",\"omega_por_imagem\":[";
        for (std::size_t i = 0; i < por_imagem.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(por_imagem[i]);
        }
        std::cout << "],\"clamp\":{\"limiares_de_sin_psi0\":[";
        for (std::size_t i = 0; i < limiares.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(limiares[i]);
        }
        std::cout << "],\"pontos\":[";
        for (std::size_t i = 0; i < conta_clamp.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << conta_clamp[i];
        }
        std::cout << "],\"omega\":[";
        for (std::size_t i = 0; i < omega_clamp.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << num17(omega_clamp[i]);
        }
        std::cout << "]}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MAGNUS diag antipodal: " << error.what() << '\n';
        return 2;
    }
}
