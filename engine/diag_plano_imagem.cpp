// Controle INDEPENDENTE no plano da imagem, para o portao P1.
//
// POR QUE EXISTE. P1 esta reprovado na NORMALIZACAO: contra a esfera estatica,
// isotermica e isotropica, a forma espectral sai exata a 6 digitos e sobra um
// fator com vies de 1,3e-4 a 4,5e-4, acima da meta de 1e-4. A quadratura
// composta sobre offsets de malha reduziu a dispersao em tres ordens e DEIXOU o
// vies, e o mecanismo nao esta identificado.
//
// O QUE ESTE PROGRAMA FAZ. Calcula a MESMA grandeza por uma quadratura
// INDEPENDENTE: integra no plano da imagem, em parametro de impacto b, em vez de
// somar celulas da malha de superficie. Para a esfera isotermica e isotropica
// fora da esfera de fotons,
//
//     b_max = R / sqrt(1-u) = R_inf ,   sin(alpha) = b sqrt(1-u) / R
//     F_E = (2 pi / D^2) integral_0^{b_max} I_E(mu(b)) b db
//         = pi B_E(T_inf) (R_inf/D)^2   quando I_E e isotropica
//
// e a integral de b db e EXATA em Gauss-Legendre sobre b^2.
//
// O QUE ELE COMPARTILHA DE PROPOSITO com o motor: a Config e o parse_args, a
// RayTable, a blackbody_photon_intensity, as constantes e a convencao de desvio
// para o vermelho. Um controle que NAO compartilhasse esses componentes nao
// distinguiria erro de quadratura de erro de componente compartilhado, que e
// exatamente o que P1 precisa separar. O que NAO e compartilhado e a quadratura.
//
// O desenho, os casos e as tolerancias estao em
// docs/incidente_resolucao_2026-10-04/pre_registro_plano_imagem.md, commit
// anterior a esta medida.
//
// UM INVARIANTE EXATO QUE TESTA O MAPEAMENTO, acrescentado depois do
// pre-registro e declarado como extensao. O caso isotropico sozinho e quase
// tautologico aqui: com I independente de mu, a integral fatora e devolve a
// formula analitica por construcao, de modo que ela so verifica a AREA e as
// CONSTANTES compartilhadas, nunca o mapeamento b <-> mu. O caso com feixe
// verifica o mapeamento, e tem solucao fechada:
//
//     F = 2 pi B(T_inf) (R_inf/D)^2 integral_0^1 I(mu) mu dmu
//
// e como a beaming_factor do motor divide por (1 + 2a/3 + b/2), que e
// exatamente integral_0^1 (1 + a mu + b mu^2) 2 mu dmu, o resultado e
//
//     F = pi B(T_inf) (R_inf/D)^2   PARA QUALQUER a e b.
//
// Logo o fluxo total e INVARIANTE sob o feixe, e qualquer dependencia medida em
// a ou b acusa erro no mapeamento mu(b), nos pesos das celulas ou na propria
// normalizacao do feixe. E um teste do que o caso isotropico nao alcanca.
//
// Uso: build/magnus_diag_imagem --mass M --radius R --distance D \
//          --base-kt-kev KT --energy-min E0 --energy-max E1 --energy-bins N \
//          [--beaming A] [--beaming2 B] [--nos-imagem 128]
//
// Saida: JSON com o fluxo por bin, o angulo solido aparente, e o analitico.

#define MAGNUS_SEM_MAIN
#include "main.cpp"

namespace {

struct Pedido {
    int nos{128};
    bool emissao_beamed{false};   // reservado: feixe (1 + a mu + b mu^2)
};

//: Le as flags DESTE programa e devolve o argv sem elas, porque o parse_args do
//: motor recusa argumento desconhecido, e recusar e o comportamento certo dele.
Pedido le_pedido(int argc, char** argv, std::vector<char*>& restante) {
    Pedido p;
    restante.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--nos-imagem" && i + 1 < argc) {
            p.nos = std::stoi(argv[++i]);
        } else if (key == "--imagem-beamed") {
            p.emissao_beamed = true;
        } else {
            restante.push_back(argv[i]);
        }
    }
    if (p.nos < 8 || p.nos > 100000) {
        throw std::runtime_error("--nos-imagem fora de [8, 100000]");
    }
    return p;
}

//: Integral no plano da imagem, em Gauss-Legendre sobre s = b^2.
//: F_E = (pi / D^2) integral_0^{b_max^2} I_E(mu(s)) ds, porque b db = ds/2.
struct Resultado {
    std::vector<double> fluxo;        // ph cm^-2 s^-1 keV^-1, por bin
    double angulo_solido{0.0};        // sr
    double b_max_km{0.0};
};

Resultado integra_plano_imagem(const Config& cfg, double u, int nos) {
    const double raio_inf_km = cfg.radius_km / std::sqrt(1.0 - u);
    const double kt_inf_mk = cfg.base_temperature_mk * std::sqrt(1.0 - u);
    const double distancia_cm = cfg.distance_kpc * kpc_to_cm;
    const double b_max_cm = raio_inf_km * 1.0e5;

    std::vector<double> x, w;
    gauss_legendre(nos, x, w);

    Resultado r;
    r.fluxo.assign(cfg.energy_bins, 0.0);
    r.b_max_km = raio_inf_km;
    const double largura = (cfg.energy_max_kev - cfg.energy_min_kev)
                           / static_cast<double>(cfg.energy_bins);

    // s percorre [0, b_max^2]; o jacobiano de b db = ds/2 ja esta no pi.
    const double s_max = b_max_cm * b_max_cm;
    double soma_angulo = 0.0;
    for (int i = 0; i < nos; ++i) {
        const double s = 0.5 * s_max * (x[i] + 1.0);
        const double peso = 0.5 * s_max * w[i];
        const double b_cm = std::sqrt(s);
        // sin(alpha) = b sqrt(1-u) / R, com b e R na mesma unidade
        const double sin_alpha = std::min(1.0, b_cm * std::sqrt(1.0 - u)
                                               / (cfg.radius_km * 1.0e5));
        const double mu = std::sqrt(std::max(0.0, 1.0 - sin_alpha * sin_alpha));
        // A beaming_factor DO MOTOR, nao uma reimplementacao: ela e componente
        // compartilhado, e um erro nela tem de aparecer nos dois caminhos.
        const double feixe = beaming_factor(mu, cfg.beaming_a, cfg.beaming_b);
        soma_angulo += peso * M_PI / (distancia_cm * distancia_cm);
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            const double energia = cfg.energy_min_kev + (ie + 0.5) * largura;
            const double intensidade = blackbody_photon_intensity(energia, kt_inf_mk);
            r.fluxo[ie] += peso * M_PI * intensidade * feixe
                           / (distancia_cm * distancia_cm);
        }
    }
    r.angulo_solido = soma_angulo;
    return r;
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
            throw std::runtime_error(
                "o controle analitico vale FORA da esfera de fotons, u < 2/3; "
                "recebi u = " + std::to_string(u));
        }
        const double raio_inf_km = cfg.radius_km / std::sqrt(1.0 - u);
        const double kt_inf_mk = cfg.base_temperature_mk * std::sqrt(1.0 - u);
        const double distancia_cm = cfg.distance_kpc * kpc_to_cm;
        const double angulo_exato = M_PI * std::pow(raio_inf_km * 1.0e5
                                                    / distancia_cm, 2.0);
        const Resultado r = integra_plano_imagem(cfg, u, pedido.nos);

        const double largura = (cfg.energy_max_kev - cfg.energy_min_kev)
                               / static_cast<double>(cfg.energy_bins);
        std::cout << "{\"status\":\"ok\",\"produto\":\"plano_imagem\""
                  << ",\"compactness\":" << number(u)
                  << ",\"raio_inf_km\":" << number(raio_inf_km)
                  << ",\"kt_inf_keV\":" << number(kt_inf_mk * kt_kev_per_mk)
                  << ",\"nos_imagem\":" << pedido.nos
                  << ",\"angulo_solido_sr\":" << number(r.angulo_solido)
                  << ",\"angulo_solido_exato_sr\":" << number(angulo_exato)
                  << ",\"erro_relativo_do_angulo\":"
                  << number(std::abs(r.angulo_solido / angulo_exato - 1.0))
                  << ",\"compartilhado_com_o_motor\":\"Config, parse_args, "
                     "RayTable, blackbody_photon_intensity, constantes e "
                     "convencao de desvio para o vermelho\""
                  << ",\"nao_compartilhado\":\"a quadratura\""
                  << ",\"flux_unit\":\"ph cm^-2 s^-1 keV^-1\""
                  << ",\"fluxo\":[";
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            if (ie) std::cout << ',';
            std::cout << number(r.fluxo[ie]);
        }
        std::cout << "],\"fluxo_exato\":[";
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            const double energia = cfg.energy_min_kev + (ie + 0.5) * largura;
            const double exato = M_PI * blackbody_photon_intensity(energia, kt_inf_mk)
                                 * std::pow(raio_inf_km * 1.0e5 / distancia_cm, 2.0);
            if (ie) std::cout << ',';
            std::cout << number(exato);
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MAGNUS diag imagem: " << error.what() << '\n';
        return 2;
    }
}
