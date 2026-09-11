#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double gm_sun_over_c2_km = 1.4766250385;
constexpr double c_km_s = 299792.458;
constexpr double c_cm_s = 2.99792458e10;
constexpr double planck_erg_s = 6.62607015e-27;
constexpr double erg_per_kev = 1.602176634e-9;
constexpr double kpc_to_cm = 3.0856775814913673e21;
constexpr double kt_kev_per_mk = 0.08617333262145;

struct Vec3 {
    double x{}, y{}, z{};
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 unit(Vec3 a) {
    const double n = norm(a);
    return n > 0.0 ? (1.0 / n) * a : Vec3{};
}

struct Spot {
    double theta_deg{30.0};
    double phi_deg{0.0};
    double radius_deg{10.0};
    double temperature_mk{2.5};
};

struct Config {
    double mass_solar{1.4};
    double radius_km{12.0};
    double distance_kpc{2.0};
    double period_s{1.0};
    double inclination_deg{80.0};
    double position_angle_deg{0.0};
    int phase_samples{241};
    int surface_rings{13};
    int max_images{2};
    bool render_image{false};
    bool transfer_table{false};
    bool spectral_grid{false};
    bool event_list{false};
    bool fit_worker{false};
    int image_width{420};
    int image_height{420};
    double observed_phase{0.0};
    int transfer_samples{1024};
    double energy_min_kev{0.2};
    double energy_max_kev{12.0};
    int energy_bins{96};
    int time_bins{128};
    std::string instrument_id{"generic"};
    std::string instrument_label{"Genérico ideal"};
    std::string instrument_profile;
    std::string instrument_response;
    double instrument_time_resolution_us{0.0};
    double instrument_dead_time_us{0.0};
    // Coluna de hidrogênio no caminho, em 1e22 cm^-2, e a tabela de seção de
    // choque que a converte em transmissão. Sem absorção o espectro térmico
    // mole chega ao detector como se o meio interestelar não existisse.
    double nh_1e22{0.0};
    // Linha de absorção gaussiana na superfície, opcional. Profundidade zero a
    // desliga, que é o padrão: o corpo negro puro continua sendo o modelo base.
    double line_energy_kev{0.3};
    bool line_cyclotron{false};
    double line_width_kev{0.1};
    double line_depth{0.0};
    // Segunda linha de absorção gaussiana. RBS 1223 mostra estrutura de absorção
    // complexa (Hambaryan et al.: linhas em ~0,23 e ~0,46 keV); e no campo
    // ajustado a linha de cíclotron de próton cai perto de ~0,9 keV, na região
    // do excesso a alta energia. Esta segunda linha testa isso.
    double line2_energy_kev{0.6};
    double line2_width_kev{0.1};
    double line2_depth{0.0};
    bool fit_line2{false};
    // Spots de CORPO NEGRO: quando ligado, os spots (spot_id>=0) emitem corpo
    // negro puro em vez da atmosfera — uma calota de superfície CONDENSADA
    // (blackbody-like, mais mole) por cima do fundo de atmosfera T(θ). A área do
    // spot já é descontada do fundo por sample_full_sphere, então ali fica só o
    // corpo negro. Motivação: a atmosfera de H é dura demais a alta energia; a
    // superfície condensada, mais mole, pode absorver esse excesso.
    bool blackbody_spots{false};
    bool spot_overlay{false};
    // Sem spots: quando ligado (--no-spots), NÃO cria o spot default. Necessário
    // para a superfície de dois polos de Pérez-Azorín, que é toda no "fundo".
    bool no_spots{false};
    // Modo de camadas (atmosfera fina sobre superfície condensada, à la
    // Hambaryan): em TODA a superfície o contínuo vira corpo negro (mole) e a
    // atmosfera fina só imprime o FEIXE (o pulso) e a linha — o endurecimento
    // espectral da tabela é dividido para fora. Quebra o trade-off de área do
    // spot condensado: conserta o alta-energia sem diluir o pulso.
    bool layered_atmosphere{false};
    // Espessura efetiva da atmosfera fina, f in [0,1]: interpola o ENDURECIMENTO
    // espectral entre corpo negro (f=0, condensada pura) e atmosfera cheia (f=1).
    // O feixe (o pulso) é sempre mantido inteiro; só o contínuo médio-em-ângulo é
    // escalado por f. É o parâmetro que une a medida de B (que vive no
    // endurecimento) com o contínuo mole — o dado escolhe quanto de cada.
    // --layered-atmosphere é o atalho para f=0.
    double atmosphere_fraction{1.0};
    bool fit_atmosphere_fraction{false};
    // Padrão de feixe I(mu) proporcional a 1 + a*mu, com mu o cosseno do ângulo
    // de emissão. a = 0 é a emissão isotrópica de sempre; a > 0 concentra ao
    // longo da normal (pencil), a < 0 achata contra a superfície (fan).
    double beaming_a{0.0};
    double beaming_b{0.0};
    // Atmosfera de dois modos: razão de opacidades X/O em 1 keV. Zero desliga e
    // o modelo volta a ser corpo negro, que continua sendo a base.
    double atmosphere_hardening{0.0};
    // Campo magnético na superfície, em gauss, e a tabela de anisotropia de
    // opacidade. Sem os dois a atmosfera fica isotrópica, que é o modelo cinza
    // de antes — e ele não consegue fazer leque nenhum.
    double magnetic_field_g{0.0};
    // Ajustar o campo B: quando ligado, o worker recebe lg B por avaliação e a
    // tabela de atmosfera é interpolada no eixo de B (formato MAGNUSI2). Sem
    // isso, B é fixo pela tabela. O bloco vem DEPOIS da atmosfera e ANTES do
    // feixe na linha do worker — a mesma ordem do worker_line do mcmc_fit.
    bool fit_log_field{false};
    // Temperatura de fundo da estrela inteira. Quando > 0, a superfície inteira
    // emite a essa temperatura (a média da emissão), e os spots entram POR CIMA,
    // substituindo o fundo na sua área — o pedaço de superfície coberto pelo spot
    // é descontado do fundo para não contar duas vezes. É o que dá o pulso SUAVE:
    // com fundo sempre visível, o fluxo nunca zera, e os spots modulam uma
    // fração do total, em vez do liga-desliga de um spot sobre estrela escura.
    double base_temperature_mk{0.0};
    bool fit_base_temperature{false};
    // Lei de temperatura dipolar T(theta) de Pérez-Azorín/Hambaryan: a superfície
    // não é isotérmica; o calor flui melhor ao longo de B (condução anisotrópica),
    // então os polos MAGNÉTICOS ficam quentes e o equador frio —
    //   T^4(theta) = T_p^4 * cos^2/(cos^2 + a*sin^2) + T_min^4,
    // com theta a colatitude MAGNÉTICA de cada ponto, T_p = base_temperature_mk,
    // T_min = temperature_min_frac * T_p, e `a` (temperature_peaking) o quão
    // concentrado é o calor: a=0 devolve o fundo uniforme (comportamento antigo),
    // a=1/4 é o dipolo clássico, a>1 uma calota apertada. É o que dá, de uma vez,
    // o pulso (polos quentes girando) e o espectro mole (equador frio, área grande).
    double temperature_peaking{0.0};       // o parâmetro `a`
    double temperature_min_frac{0.3};      // T_min / T_p (Pérez-Azorín ~0,3)
    bool fit_temperature_peaking{false};
    // SEGUNDO polo de Pérez-Azorín (modelo de dois polos de Hambaryan et al. 2011,
    // Eqs. 4-5): a hemisfera cmag<0 (em torno do polo oposto) recebe seu PRÓPRIO
    // par (T_p2, a2), de modo que os dois polos podem ser DESIGUAIS e a superfície
    // fica suave (gradiente), não casquetes de borda dura. Ligado quando
    // base_temperature2_mk>0; senão a lei de um polo (simétrica) vale como antes.
    double base_temperature2_mk{0.0};      // T_p2 (polo 2), em MK
    bool fit_base_temperature2{false};
    double temperature_peaking2{0.0};      // a2 (polo 2)
    bool fit_temperature_peaking2{false};
    // Desvio da ANTIPODALIDADE do polo 2: seu eixo e -B girado por beta no plano
    // magneto-rotacional (o de B e Omega). beta=0 => polos antipodais (dipolo).
    // beta!=0 => polos nao-antipodais, com a assimetria geometrica que os
    // casquetes livres tinham, agora dentro da lei suave de Perez-Azorin.
    double pole2_tilt_deg{0.0};
    bool fit_pole2_tilt{false};
    // Inclinação do eixo do dipolo em relação ao eixo de rotação. Quando LIVRE,
    // o fundo axissimétrico deixa de ser: a atmosfera é anisotrópica em theta_B,
    // então um dipolo inclinado faz o disco visível varrer theta_B ao girar e o
    // fundo pulsa SUAVEMENTE sozinho — o mecanismo de pulso das XDINS, sem spot.
    bool fit_magnetic_colatitude{false};
    // Azimute do eixo do dipolo. É o knob de FASE do pulso do fundo dipolar: gira
    // o padrão magnético em torno do eixo de rotação, deslocando onde o máximo do
    // pulso cai. Sem ele livre, o pulso do fundo não alinha com o da observação
    // (o phaseOffset só move os spots), e o ajuste foge para pole-on.
    bool fit_magnetic_azimuth{false};
    std::string anisotropy_table;
    std::string nsmaxg_table;
    // Tabela de intensidade do MAGNUS: lg da razão para corpo negro, resolvida
    // em (lg T, lg g, theta_B, mu, lg E). Ela entra no lugar do espectro E do
    // feixe, porque traz os dois; vazia, o motor fica no modelo anterior.
    std::string atmosphere_table;
    // Eixo do dipolo magnético, em relação ao eixo de rotação. O ângulo entre B
    // e a normal da superfície NÃO é parâmetro: sai daqui, ponto a ponto, pela
    // geometria do dipolo. Alinhado com a rotação é o padrão.
    double magnetic_colatitude_deg{0.0};
    double magnetic_azimuth_deg{0.0};
    bool fit_atmosphere{false};
    // Dizem ao worker persistente quantos campos esperar em cada linha. Sem
    // isto o protocolo posicional não distingue "linha ajustada" de "linha fixa
    // com valor da configuração base".
    bool fit_line{false};
    bool fit_beaming{false};
    std::string absorption_table;
    double exposure_s{200.0};
    std::uint64_t random_seed{1234};
    std::size_t max_events{500000};
    std::vector<Spot> spots{{30.0, 0.0, 10.0, 2.5}, {120.0, 200.0, 12.0, 2.5}};
};

double deg(double value) { return value * pi / 180.0; }
double clamp_unit(double value) { return std::clamp(value, -1.0, 1.0); }

class RayTable {
  public:
    explicit RayTable(double compactness) : u_(compactness) {
        if (!(u_ > 0.0 && u_ < 1.0)) {
            throw std::runtime_error("compactness must satisfy 0 < 2GM/(Rc^2) < 1");
        }
        const bool inside_photon_sphere = u_ > 2.0 / 3.0;
        double alpha_escape = pi / 2.0;
        if (inside_photon_sphere) {
            const double m_over_r = u_ / 2.0;
            const double critical_sine = 3.0 * std::sqrt(3.0) * m_over_r * std::sqrt(1.0 - u_);
            alpha_escape = std::asin(clamp_unit(critical_sine));
        }

        constexpr int table_size = 1800;
        alpha_.reserve(table_size);
        psi_.reserve(table_size);
        for (int i = 0; i < table_size; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(table_size - 1);
            // Concentrate points near the escape direction where psi changes fastest.
            const double shaped = 1.0 - std::pow(1.0 - t, 3.0);
            const double alpha = alpha_escape * shaped * (1.0 - 2.0e-7);
            const double psi = bending(alpha);
            if (std::isfinite(psi) && (psi_.empty() || psi > psi_.back())) {
                alpha_.push_back(alpha);
                psi_.push_back(psi);
                delay_.push_back(time_delay(alpha));
            }
        }
    }

    double psi_max() const { return psi_.empty() ? 0.0 : psi_.back(); }
    double alpha_max() const { return alpha_.empty() ? 0.0 : alpha_.back(); }

    bool trace_alpha(double alpha, double& psi, double& delay_seconds_per_km) const {
        if (alpha < 0.0 || alpha_.size() < 2 || alpha > alpha_.back()) return false;
        auto upper = std::lower_bound(alpha_.begin(), alpha_.end(), alpha);
        std::size_t hi = static_cast<std::size_t>(upper - alpha_.begin());
        if (hi == 0) hi = 1;
        if (hi >= alpha_.size()) hi = alpha_.size() - 1;
        const std::size_t lo = hi - 1;
        const double da = alpha_[hi] - alpha_[lo];
        if (da <= 0.0) return false;
        const double f = (alpha - alpha_[lo]) / da;
        psi = psi_[lo] + f * (psi_[hi] - psi_[lo]);
        // delay_ is the dimensionless integral; R/c is applied by the renderer.
        delay_seconds_per_km = delay_[lo] + f * (delay_[hi] - delay_[lo]);
        return true;
    }

    bool invert(double target_psi, double& alpha, double& dalpha_dpsi) const {
        if (target_psi < 0.0 || psi_.size() < 2 || target_psi > psi_.back()) return false;
        auto upper = std::lower_bound(psi_.begin(), psi_.end(), target_psi);
        std::size_t hi = static_cast<std::size_t>(upper - psi_.begin());
        if (hi == 0) hi = 1;
        if (hi >= psi_.size()) hi = psi_.size() - 1;
        const std::size_t lo = hi - 1;
        const double dpsi = psi_[hi] - psi_[lo];
        if (dpsi <= 0.0) return false;
        const double f = (target_psi - psi_[lo]) / dpsi;
        alpha = alpha_[lo] + f * (alpha_[hi] - alpha_[lo]);
        dalpha_dpsi = (alpha_[hi] - alpha_[lo]) / dpsi;
        return true;
    }

  private:
    double bending(double alpha) const {
        const double k = std::sin(alpha) / std::sqrt(1.0 - u_); // b/R
        constexpr int intervals = 2400;
        double sum = 0.0;
        // Midpoint integration after x=1-y^2 removes the tangential endpoint singularity.
        for (int i = 0; i < intervals; ++i) {
            const double y = (static_cast<double>(i) + 0.5) / intervals;
            const double x = 1.0 - y * y;
            const double radicand = 1.0 - k * k * x * x * (1.0 - u_ * x);
            if (radicand <= 0.0) return INFINITY;
            sum += 2.0 * y * k / std::sqrt(radicand);
        }
        return sum / intervals;
    }

    double time_delay(double alpha) const {
        const double k = std::sin(alpha) / std::sqrt(1.0 - u_);
        constexpr int intervals = 1600;
        double sum = 0.0;
        for (int i = 0; i < intervals; ++i) {
            const double y = (static_cast<double>(i) + 0.5) / intervals;
            const double x = 1.0 - y * y;
            const double one_minus_ux = 1.0 - u_ * x;
            const double radicand = 1.0 - k * k * x * x * one_minus_ux;
            if (radicand <= 0.0) return INFINITY;
            const double correction = 1.0 / std::sqrt(radicand) - 1.0;
            const double x2 = std::max(x * x, 1.0e-18);
            sum += 2.0 * y * correction / (x2 * one_minus_ux);
        }
        return sum / intervals;
    }

    double u_;
    std::vector<double> alpha_;
    std::vector<double> psi_;
    std::vector<double> delay_;
};

struct SurfaceSample {
    Vec3 normal;
    double weight{};
    double intensity{};
};

struct SpectralSurfaceSample {
    Vec3 normal;
    double weight{};
    double temperature_mk{};
    //: Ângulo entre B e a normal, em graus. É propriedade do PONTO e não da
    //: fase — a normal e o eixo magnético giram juntos —, então se calcula uma
    //: vez, quando a superfície é amostrada.
    double theta_b_deg{};
    int spot_id{};
};

struct RayContribution {
    double arrival_phase{};
    double emission_phase{};
    double lensing_jacobian{};
    double solid_angle_sr{};
    double energy_shift_g{};
    double delay_s{};
    double temperature_mk{};
    double theta_b_deg{};
    double cos_emission{};
    int spot_id{};
    int image_order{};
};

std::vector<SurfaceSample> sample_spot(const Spot& spot, int rings) {
    const double theta = deg(spot.theta_deg);
    const double phi = deg(spot.phi_deg);
    const double rho = deg(spot.radius_deg);
    const Vec3 center{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
    const Vec3 e_theta{std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi), -std::sin(theta)};
    const Vec3 e_phi{-std::sin(phi), std::cos(phi), 0.0};
    std::vector<SurfaceSample> samples;
    const double intensity = std::pow(std::max(spot.temperature_mk, 0.01), 4.0);
    for (int ir = 0; ir < rings; ++ir) {
        const double r0 = rho * ir / rings;
        const double r1 = rho * (ir + 1) / rings;
        const double rmid = 0.5 * (r0 + r1);
        const int azimuths = std::max(8, static_cast<int>(std::ceil(2.0 * pi * (ir + 0.5))));
        const double ring_area = 2.0 * pi * (std::cos(r0) - std::cos(r1));
        for (int ia = 0; ia < azimuths; ++ia) {
            const double q = 2.0 * pi * (ia + 0.5) / azimuths;
            const Vec3 tangent = std::cos(q) * e_theta + std::sin(q) * e_phi;
            const Vec3 normal = unit(std::cos(rmid) * center + std::sin(rmid) * tangent);
            samples.push_back({normal, ring_area / azimuths, intensity});
        }
    }
    return samples;
}

// A estrela inteira a uma temperatura de fundo, descartando os ladrilhos cujo
// centro cai dentro de algum spot: ali quem emite é o spot, e o fundo é
// subtraído para não contar duas vezes o mesmo pedaço de superfície. É o modelo
// "temperatura média da estrela + spots" — o fundo sempre visível dá o pulso
// SUAVE que um spot sobre estrela escura não consegue (aquele ou fica sempre
// visível, e não pulsa, ou some atrás da estrela, e pulsa cem por cento).
//
// A tesselação é em bandas de colatitude com azimutes proporcionais ao seno,
// para ladrilhos de área quase igual. Ela é independente da resolução dos
// spots: o fundo é suave e não precisa de malha fina para a sua contribuição de
// fase (que vem só da variação de theta_B pelo disco e da curvatura da luz).
std::vector<SurfaceSample> sample_full_sphere(double base_temperature_mk, int bands,
                                              const std::vector<Spot>& spots) {
    std::vector<SurfaceSample> samples;
    const double intensity = std::pow(std::max(base_temperature_mk, 0.01), 4.0);
    std::vector<Vec3> centers;
    std::vector<double> cos_radius;
    centers.reserve(spots.size());
    cos_radius.reserve(spots.size());
    for (const auto& spot : spots) {
        const double th = deg(spot.theta_deg), ph = deg(spot.phi_deg);
        centers.push_back({std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph),
                           std::cos(th)});
        cos_radius.push_back(std::cos(deg(spot.radius_deg)));
    }
    const int nlat = std::max(8, bands);
    for (int il = 0; il < nlat; ++il) {
        const double t0 = pi * il / nlat;
        const double t1 = pi * (il + 1) / nlat;
        const double tmid = 0.5 * (t0 + t1);
        const int azimuths = std::max(4, static_cast<int>(
            std::ceil(2.0 * nlat * std::sin(tmid))));
        const double tile_area = 2.0 * pi * (std::cos(t0) - std::cos(t1)) / azimuths;
        for (int ia = 0; ia < azimuths; ++ia) {
            const double ph = 2.0 * pi * (ia + 0.5) / azimuths;
            const Vec3 normal{std::sin(tmid) * std::cos(ph),
                              std::sin(tmid) * std::sin(ph), std::cos(tmid)};
            bool in_spot = false;
            for (std::size_t s = 0; s < centers.size(); ++s) {
                if (dot(normal, centers[s]) >= cos_radius[s]) { in_spot = true; break; }
            }
            if (in_spot) continue;
            samples.push_back({normal, tile_area, intensity});
        }
    }
    return samples;
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    cfg.spots.clear();
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + key);
            return argv[i];
        };
        if (key == "--mass") cfg.mass_solar = std::stod(value());
        else if (key == "--radius") cfg.radius_km = std::stod(value());
        else if (key == "--distance") cfg.distance_kpc = std::stod(value());
        else if (key == "--period") cfg.period_s = std::stod(value());
        else if (key == "--inclination") cfg.inclination_deg = std::stod(value());
        else if (key == "--position-angle") cfg.position_angle_deg = std::stod(value());
        else if (key == "--samples") cfg.phase_samples = std::stoi(value());
        else if (key == "--rings") cfg.surface_rings = std::stoi(value());
        else if (key == "--max-images") cfg.max_images = std::stoi(value());
        else if (key == "--render") cfg.render_image = true;
        else if (key == "--transfer") cfg.transfer_table = true;
        else if (key == "--spectral-grid") cfg.spectral_grid = true;
        else if (key == "--event-list") { cfg.event_list = true; cfg.spectral_grid = true; }
        else if (key == "--fit-worker") { cfg.fit_worker = true; cfg.spectral_grid = true; }
        else if (key == "--transfer-samples") cfg.transfer_samples = std::stoi(value());
        else if (key == "--energy-min") cfg.energy_min_kev = std::stod(value());
        else if (key == "--energy-max") cfg.energy_max_kev = std::stod(value());
        else if (key == "--energy-bins") cfg.energy_bins = std::stoi(value());
        else if (key == "--time-bins") cfg.time_bins = std::stoi(value());
        else if (key == "--instrument") cfg.instrument_id = value();
        else if (key == "--instrument-label") cfg.instrument_label = value();
        else if (key == "--instrument-profile") cfg.instrument_profile = value();
        else if (key == "--instrument-response") cfg.instrument_response = value();
        else if (key == "--time-resolution-us") cfg.instrument_time_resolution_us = std::stod(value());
        else if (key == "--dead-time-us") cfg.instrument_dead_time_us = std::stod(value());
        else if (key == "--nh") cfg.nh_1e22 = std::stod(value());
        else if (key == "--line-energy") cfg.line_energy_kev = std::stod(value());
        else if (key == "--line-cyclotron") cfg.line_cyclotron = true;
        else if (key == "--line-width") cfg.line_width_kev = std::stod(value());
        else if (key == "--line-depth") cfg.line_depth = std::stod(value());
        else if (key == "--beaming") cfg.beaming_a = std::stod(value());
        else if (key == "--beaming2") cfg.beaming_b = std::stod(value());
        else if (key == "--fit-line") cfg.fit_line = true;
        else if (key == "--line2-energy") cfg.line2_energy_kev = std::stod(value());
        else if (key == "--line2-width") cfg.line2_width_kev = std::stod(value());
        else if (key == "--line2-depth") cfg.line2_depth = std::stod(value());
        else if (key == "--fit-line2") cfg.fit_line2 = true;
        else if (key == "--blackbody-spots") cfg.blackbody_spots = true;
        else if (key == "--spot-overlay") cfg.spot_overlay = true;
        else if (key == "--no-spots") cfg.no_spots = true;
        else if (key == "--layered-atmosphere") { cfg.layered_atmosphere = true; cfg.atmosphere_fraction = 0.0; }
        else if (key == "--atmosphere-fraction") cfg.atmosphere_fraction = std::stod(value());
        else if (key == "--fit-atmosphere-fraction") cfg.fit_atmosphere_fraction = true;
        else if (key == "--fit-beaming") cfg.fit_beaming = true;
        else if (key == "--atmosphere") cfg.atmosphere_hardening = std::stod(value());
        else if (key == "--magnetic-field") cfg.magnetic_field_g = std::stod(value());
        else if (key == "--fit-log-field") cfg.fit_log_field = true;
        else if (key == "--base-temp-mk") cfg.base_temperature_mk = std::stod(value());
        else if (key == "--base-kt-kev") cfg.base_temperature_mk = std::stod(value()) / kt_kev_per_mk;
        else if (key == "--fit-base-temperature") cfg.fit_base_temperature = true;
        else if (key == "--temperature-peaking") cfg.temperature_peaking = std::stod(value());
        else if (key == "--temperature-min-frac") cfg.temperature_min_frac = std::stod(value());
        else if (key == "--fit-temperature-peaking") cfg.fit_temperature_peaking = true;
        else if (key == "--base-kt2-kev") cfg.base_temperature2_mk = std::stod(value()) / kt_kev_per_mk;
        else if (key == "--base-temp2-mk") cfg.base_temperature2_mk = std::stod(value());
        else if (key == "--temperature-peaking2") cfg.temperature_peaking2 = std::stod(value());
        else if (key == "--fit-base-temperature2") cfg.fit_base_temperature2 = true;
        else if (key == "--fit-temperature-peaking2") cfg.fit_temperature_peaking2 = true;
        else if (key == "--pole2-tilt") cfg.pole2_tilt_deg = std::stod(value());
        else if (key == "--fit-pole2-tilt") cfg.fit_pole2_tilt = true;
        else if (key == "--fit-magnetic-colatitude") cfg.fit_magnetic_colatitude = true;
        else if (key == "--fit-magnetic-azimuth") cfg.fit_magnetic_azimuth = true;
        else if (key == "--anisotropy-table") cfg.anisotropy_table = value();
        else if (key == "--nsmaxg-table") cfg.nsmaxg_table = value();
        else if (key == "--atmosphere-table") cfg.atmosphere_table = value();
        else if (key == "--magnetic-colatitude") cfg.magnetic_colatitude_deg = std::stod(value());
        else if (key == "--magnetic-azimuth") cfg.magnetic_azimuth_deg = std::stod(value());
        else if (key == "--fit-atmosphere") cfg.fit_atmosphere = true;
        else if (key == "--absorption") cfg.absorption_table = value();
        else if (key == "--exposure") cfg.exposure_s = std::stod(value());
        else if (key == "--seed") cfg.random_seed = std::stoull(value());
        else if (key == "--max-events") cfg.max_events = std::stoull(value());
        else if (key == "--width") cfg.image_width = std::stoi(value());
        else if (key == "--height") cfg.image_height = std::stoi(value());
        else if (key == "--phase") cfg.observed_phase = std::stod(value());
        else if (key == "--spot") {
            std::string spec = value();
            std::replace(spec.begin(), spec.end(), ',', ' ');
            std::istringstream in(spec);
            Spot s;
            if (!(in >> s.theta_deg >> s.phi_deg >> s.radius_deg >> s.temperature_mk)) {
                throw std::runtime_error("spot must be theta,phi,radius,temperature");
            }
            cfg.spots.push_back(s);
        } else if (key == "--help") {
            std::cout << "MAGNUS engine --mass M --radius R --period P --inclination i "
                         "--spot theta,phi,radius,temp [--spot ...] "
                         "[--spectral-grid --energy-min E --energy-max E --energy-bins N --time-bins N] "
                         "[--atmosphere-table T --magnetic-colatitude deg --magnetic-azimuth deg]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (cfg.spots.empty() && !cfg.no_spots) cfg.spots.push_back({30.0, 0.0, 10.0, 2.5});
    if (cfg.mass_solar <= 0 || cfg.radius_km <= 0 || cfg.distance_kpc <= 0 || cfg.period_s <= 0) {
        throw std::runtime_error("mass, radius, distance and period must be positive");
    }
    for (const auto& spot : cfg.spots) {
        if (spot.temperature_mk <= 0 || spot.radius_deg <= 0) {
            throw std::runtime_error("spot radius and temperature must be positive");
        }
    }
    if (!(cfg.energy_min_kev > 0.0 && cfg.energy_max_kev > cfg.energy_min_kev)) {
        throw std::runtime_error("energy range must satisfy 0 < energy-min < energy-max");
    }
    cfg.phase_samples = std::clamp(cfg.phase_samples, 32, 1000);
    cfg.surface_rings = std::clamp(cfg.surface_rings, 4, 30);
    cfg.max_images = std::clamp(cfg.max_images, 1, 4);
    cfg.image_width = std::clamp(cfg.image_width, 96, 1024);
    cfg.image_height = std::clamp(cfg.image_height, 96, 1024);
    cfg.transfer_samples = std::clamp(cfg.transfer_samples, 128, 4096);
    cfg.energy_bins = std::clamp(cfg.energy_bins, 8, 512);
    cfg.time_bins = std::clamp(cfg.time_bins, 16, 1000);
    if (cfg.instrument_time_resolution_us < 0.0 || cfg.instrument_dead_time_us < 0.0) {
        throw std::runtime_error("instrument time resolution and dead time cannot be negative");
    }
    if (!(cfg.exposure_s > 0.0) || cfg.exposure_s > 1.0e9) {
        throw std::runtime_error("exposure must satisfy 0 < exposure <= 1e9 s");
    }
    cfg.max_events = std::clamp<std::size_t>(cfg.max_events, 1, 2000000);
    return cfg;
}

void write_u16(std::ostream& out, std::uint16_t value) {
    const char bytes[2]{static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
    out.write(bytes, 2);
}

void write_u32(std::ostream& out, std::uint32_t value) {
    const char bytes[4]{static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff),
                        static_cast<char>((value >> 16) & 0xff), static_cast<char>((value >> 24) & 0xff)};
    out.write(bytes, 4);
}

struct Rgb {
    double r{}, g{}, b{};
};

Rgb spot_palette(std::size_t index) {
    constexpr Rgb palette[]{{1.00, 0.60, 0.13}, {1.00, 0.10, 0.38}, {0.12, 0.72, 1.00}};
    return palette[index % 3];
}

int image_order(double psi) {
    return std::max(0, static_cast<int>(std::floor((psi + 1.0e-9) / pi)));
}

void render_bmp(const Config& cfg, const RayTable& rays, double u) {
    const int width = cfg.image_width;
    const int height = cfg.image_height;
    const int row_stride = (width * 3 + 3) & ~3;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(row_stride) * height, 0);
    const double inclination = deg(cfg.inclination_deg);
    const double position_angle = deg(cfg.position_angle_deg);
    const Vec3 observer{std::sin(inclination), 0.0, std::cos(inclination)};
    const Vec3 base_right{0.0, 1.0, 0.0};
    const Vec3 base_up{-std::cos(inclination), 0.0, std::sin(inclination)};
    const Vec3 screen_x = std::cos(position_angle) * base_right + std::sin(position_angle) * base_up;
    const Vec3 screen_up = -std::sin(position_angle) * base_right + std::cos(position_angle) * base_up;
    const double sin_alpha_max = std::sin(rays.alpha_max());
    const double gravitational_shift = std::sqrt(1.0 - u);
    const double equatorial_beta = 2.0 * pi * cfg.radius_km /
        (cfg.period_s * c_km_s * gravitational_shift);
    double hottest = 0.1;
    for (const auto& spot : cfg.spots) hottest = std::max(hottest, spot.temperature_mk);
    const double hottest4 = std::pow(hottest, 4.0);
    constexpr int supersampling = 2;

    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            Rgb accumulated{};
            for (int syi = 0; syi < supersampling; ++syi) {
                for (int sxi = 0; sxi < supersampling; ++sxi) {
                    const double sx = (2.0 * (px + (sxi + 0.5) / supersampling) / width - 1.0);
                    const double sy = (2.0 * (py + (syi + 0.5) / supersampling) / height - 1.0);
                    const double q = std::sqrt(sx * sx + sy * sy);
                    if (q >= 1.0) continue;
                    const double alpha = std::asin(std::clamp(q * sin_alpha_max, 0.0, 1.0));
                    double psi{}, delay_integral{};
                    if (!rays.trace_alpha(alpha, psi, delay_integral)) continue;
                    const int order = image_order(psi);
                    if (order >= cfg.max_images) continue;
                    const double chi_x = q > 1.0e-12 ? sx / q : 0.0;
                    const double chi_y = q > 1.0e-12 ? -sy / q : 0.0;
                    const Vec3 transverse = chi_x * screen_x + chi_y * screen_up;
                    const Vec3 surface_normal = unit(std::sin(psi) * transverse + std::cos(psi) * observer);
                    const double delay_seconds = cfg.radius_km * delay_integral / c_km_s;
                    const double emission_phase = cfg.observed_phase - 2.0 * pi * delay_seconds / cfg.period_s;
                    const double sin_psi = std::sin(psi);
                    Vec3 tangent{};
                    if (std::abs(sin_psi) > 1.0e-8) {
                        tangent = (1.0 / sin_psi) * (observer + (-std::cos(psi)) * surface_normal);
                    }
                    const Vec3 emitted_direction = unit(std::cos(alpha) * surface_normal + std::sin(alpha) * tangent);
                    Vec3 velocity_dir{-surface_normal.y, surface_normal.x, 0.0};
                    const double sin_theta = norm(velocity_dir);
                    velocity_dir = unit(velocity_dir);
                    const double beta = std::min(0.95, equatorial_beta * sin_theta);
                    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
                    const double doppler = 1.0 / (gamma * (1.0 - beta * dot(emitted_direction, velocity_dir)));
                    const double transfer4 = std::pow(gravitational_shift * doppler, 4.0);

                    Rgb sample_color{};
                    double sample_intensity = 0.0;
                    for (std::size_t si = 0; si < cfg.spots.size(); ++si) {
                        const auto& spot = cfg.spots[si];
                        const double theta = deg(spot.theta_deg);
                        const double phi = deg(spot.phi_deg) + emission_phase;
                        const Vec3 center{std::sin(theta) * std::cos(phi),
                                          std::sin(theta) * std::sin(phi), std::cos(theta)};
                        if (dot(surface_normal, center) < std::cos(deg(spot.radius_deg))) continue;
                        const double intensity = std::pow(spot.temperature_mk, 4.0) * transfer4 / hottest4;
                        const Rgb palette = spot_palette(si);
                        sample_color.r += intensity * palette.r;
                        sample_color.g += intensity * palette.g;
                        sample_color.b += intensity * palette.b;
                        sample_intensity += intensity;
                    }

                    // A dim photosphere makes the apparent stellar boundary and higher-order
                    // image annuli visible without competing with the hot-spot emission.
                    const double limb = 0.35 + 0.65 * std::cos(alpha);
                    const double order_fade = 1.0 / (1.0 + 0.18 * order);
                    Rgb base{0.018 * limb, 0.032 * limb, 0.075 * limb};
                    if (sample_intensity > 0.0) {
                        const double tone = 1.0 - std::exp(-1.7 * sample_intensity * order_fade);
                        const double scale = tone / sample_intensity;
                        base.r += sample_color.r * scale;
                        base.g += sample_color.g * scale;
                        base.b += sample_color.b * scale;
                    }
                    accumulated.r += base.r;
                    accumulated.g += base.g;
                    accumulated.b += base.b;
                }
            }
            const double inv_samples = 1.0 / (supersampling * supersampling);
            auto encode = [inv_samples](double channel) -> std::uint8_t {
                const double mapped = std::pow(std::clamp(channel * inv_samples, 0.0, 1.0), 1.0 / 2.2);
                return static_cast<std::uint8_t>(std::lround(255.0 * mapped));
            };
            // BMP stores rows bottom-up and channels in BGR order.
            const std::size_t offset = static_cast<std::size_t>(height - 1 - py) * row_stride + 3 * px;
            pixels[offset] = encode(accumulated.b);
            pixels[offset + 1] = encode(accumulated.g);
            pixels[offset + 2] = encode(accumulated.r);
        }
    }

    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(pixels.size());
    std::cout.write("BM", 2);
    write_u32(std::cout, 54 + pixel_bytes);
    write_u16(std::cout, 0); write_u16(std::cout, 0); write_u32(std::cout, 54);
    write_u32(std::cout, 40); write_u32(std::cout, width); write_u32(std::cout, height);
    write_u16(std::cout, 1); write_u16(std::cout, 24); write_u32(std::cout, 0);
    write_u32(std::cout, pixel_bytes); write_u32(std::cout, 2835); write_u32(std::cout, 2835);
    write_u32(std::cout, 0); write_u32(std::cout, 0);
    std::cout.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

void write_transfer_table(const Config& cfg, const RayTable& rays, double u) {
    std::cout << "{\"status\":\"ok\",\"compactness\":" << number(u)
              << ",\"alpha_max\":" << number(rays.alpha_max())
              << ",\"psi_max\":" << number(rays.psi_max())
              << ",\"samples\":" << cfg.transfer_samples << ",\"transfer\":[";
    const double sin_alpha_max = std::sin(rays.alpha_max());
    for (int i = 0; i < cfg.transfer_samples; ++i) {
        if (i) std::cout << ',';
        const double q = static_cast<double>(i) / (cfg.transfer_samples - 1);
        const double safe_q = std::min(q, 1.0 - 2.0e-7);
        const double alpha = std::asin(std::clamp(safe_q * sin_alpha_max, 0.0, 1.0));
        double psi{}, delay{};
        if (!rays.trace_alpha(alpha, psi, delay)) {
            psi = rays.psi_max();
            delay = 0.0;
        }
        std::cout << number(psi) << ',' << number(delay);
    }
    std::cout << "]}\n";
}

Vec3 rotate_about_z(Vec3 n, double phase) {
    const double cp = std::cos(phase);
    const double sp = std::sin(phase);
    return {cp * n.x - sp * n.y, sp * n.x + cp * n.y, n.z};
}

// Perfil de absorção gaussiana em profundidade óptica, na forma que a
// literatura usa para XDINS: exp(-D * exp(-(E-E0)^2 / 2 sigma^2)). Aplica-se na
// energia **emitida**, no referencial comóvel, porque a feição se forma na
// superfície — aplicá-la na energia observada erraria pelo desvio para o
// vermelho gravitacional e pelo Doppler.
double line_transmission(double emitted_energy_kev, double centre_kev,
                         double width_kev, double depth) {
    if (depth <= 0.0 || width_kev <= 0.0) return 1.0;
    const double offset = (emitted_energy_kev - centre_kev) / width_kev;
    return std::exp(-depth * std::exp(-0.5 * offset * offset));
}

// Padrão de feixe I(mu) = (1 + a*mu + b*mu^2) / (1 + 2a/3 + b/2).
//
// A normalização não é cosmética: sem ela, mudar a ou b mudaria o fluxo
// bolométrico e os parâmetros ficariam degenerados com a temperatura e com a
// área do ponto. O denominador vem de exigir que a integral de I(mu)*mu sobre o
// hemisfério não mude — ela vale 1/2 + a/3 + b/4 contra 1/2 do caso isotrópico.
// Assim a e b mexem só na FORMA do feixe, que é o que distingue geometrias.
//
// O termo quadrático existe porque a forma linear não bastou: ajustando a
// RBS 1223 ela encostava no piso da priori nas três configurações testadas, ou
// seja os dados pediam um leque mais achatado do que 1 + a*mu consegue produzir.
// Com b, o padrão pode ter um nulo em um ângulo intermediário — por exemplo
// (1 - 1.5*mu)^2, que se anula em mu = 2/3 — que é o tipo de estrutura de dois
// lóbulos que atmosferas magnetizadas produzem.
//
// A positividade é imposta na priori, não aqui: cortar em zero neste ponto
// invalidaria em silêncio a integral de onde veio a normalização.
double beaming_factor(double cos_emission, double a, double b) {
    if (a == 0.0 && b == 0.0) return 1.0;
    const double normalisation = 1.0 + 2.0 * a / 3.0 + b / 2.0;
    if (normalisation <= 0.0) return 1.0;
    const double shape = 1.0 + a * cos_emission + b * cos_emission * cos_emission;
    return std::max(0.0, shape) / normalisation;
}

double blackbody_photon_intensity(double energy_kev, double temperature_mk) {
    if (!(energy_kev > 0.0 && temperature_mk > 0.0)) return 0.0;
    const double x = energy_kev / (kt_kev_per_mk * temperature_mk);
    if (x > 700.0) return 0.0;
    const double denominator = std::expm1(x);
    if (!(denominator > 0.0)) return 0.0;
    const double energy_erg = energy_kev * erg_per_kev;
    // Planck photon specific intensity per keV in the comoving frame:
    // photons s^-1 cm^-2 sr^-1 keV^-1.
    return 2.0 * energy_erg * energy_erg * erg_per_kev /
        (planck_erg_s * planck_erg_s * planck_erg_s * c_cm_s * c_cm_s * denominator);
}

// ---------------------------------------------------------------------------
// Atmosfera magnetizada cinza de dois modos.
//
// Num plasma fortemente magnetizado a radiação se propaga em dois modos normais
// com opacidades muito diferentes. O modo extraordinário tem seção de choque
// suprimida por cerca de (E/E_Be)^2, onde E_Be = 11,6 keV * B_12 é a energia de
// cíclotron do elétron — em 10^13 G isso são seis ordens de grandeza abaixo de
// 1 keV. O modo extraordinário escapa então de camadas muito mais fundas, e
// portanto mais quentes, que o ordinário.
//
// O cálculo usa a relação de Eddington-Barbier: a intensidade que emerge na
// direção mu vem da profundidade óptica tau = mu, medida na escala DAQUELE
// modo. Com a estrutura de temperatura cinza T^4(tau) = (3/4) T_ef^4 (tau + 2/3),
// o modo m emerge de tau_comum = mu / r_m, com r_m a razão de opacidades.
//
// ISTO NÃO É UM MODELO DE ATMOSFERA DE PRIMEIROS PRINCÍPIOS. Não há equação de
// estado de hidrogênio parcialmente ionizado, não há transições ligado-ligado,
// não há conversão de modo por polarização do vácuo. É a estrutura mínima que
// produz feixe a partir de transporte radiativo em vez de ajuste, e o
// parâmetro livre — a endurecimento de cor — tem significado físico.
// O artigo precisa dizer isso com estas palavras.

// O parâmetro livre é o ENDURECIMENTO, T_X/T_ef na normal e em 1 keV, e não a
// razão microscópica de opacidades. A escolha é deliberada.
//
// A razão verdadeira é minúscula: com E_Be = 11,6 keV * B_12, um campo de
// 8,6e13 G dá E_Be ~ 1 MeV e (E/E_Be)^2 ~ 1e-6 em 1 keV. Mas este modelo cinza
// não sobrevive a 1e-6 — o modo X sairia de tau_O = 1e6, onde a estrutura cinza
// diz T = 165 T_ef, o que é absurdo. O absurdo é do modelo, não da física: numa
// atmosfera de verdade o perfil de temperatura se reajusta, porque o modo X,
// escapando fácil, esfria as camadas de onde sai. Os cálculos que fazem esse
// reajuste — Ho & Lai (2001), van Adelsberg & Lai (2006), Suleimanov et al.
// (2009) — acham endurecimento de cor entre 1,5 e 2 para hidrogênio
// magnetizado, não de 165.
//
// Então o parâmetro é o endurecimento, com priori 1 a 3: cobre o intervalo que
// os cálculos completos produzem, com folga, e exclui a fuga. Rotulá-lo de
// "razão de opacidades" e depois limitá-lo quatro ordens de grandeza acima do
// valor microscópico seria mentir sobre o que está sendo ajustado. Assim, o
// número que sai do ajuste se compara direto com aqueles artigos.
//
// **ESTA FAMÍLIA NÃO ANINHA O CORPO NEGRO, e nenhum valor de h a faz aninhar.**
// Nem sequer em h = 1: ali os dois modos coincidem em 1 keV, mas a separação
// vai com E^2 e portanto sobrevive no resto da banda — 0,15 a 2 keV é quase
// toda abaixo de 1 keV. E mesmo sem separação alguma o modelo continuaria
// diferente, porque a atmosfera cinza de opacidade única já é escurecida no
// bordo, ao passo que o corpo negro é isotrópico: medido, 0,927 da taxa na
// banda. Comparar atmosfera com corpo negro é comparação NÃO ANINHADA — por
// AIC ou BIC, nunca por razão de verossimilhanças com um grau de liberdade.
// Uma versão anterior deste arquivo afirmava o contrário; o teste
// test_hardening_of_one_reproduces_the_blackbody derrubou a afirmação.

// ---------------------------------------------------------------------------
// Anisotropia da opacidade, de Potekhin & Chabrier (2003).
//
// **É o ingrediente que faltava.** O modelo cinza acima gera o feixe só pela
// estratificação em profundidade, e por isso dá escurecimento de bordo para
// QUALQUER valor dos seus parâmetros — medido, monotônico de 1,000 na normal a
// ~0,1 na rasante, enquanto os dados da RBS 1223 pediram o contrário. O que
// falta é a opacidade depender do ângulo em relação a B, e Potekhin & Chabrier
// publicam isso tabulado, de graça:
//
//     1/K(theta) = cos^2(theta)/K0 + sin^2(theta)/K1
//
// com K0 ao longo de B e K1 perpendicular. A relação é dos próprios autores.
// scripts/build_magnetic_anisotropy.py resolve a fotosfera (P = g/K0, com os
// dois lados da mesma linha da tabela) e tabela a razão a = K0/K1 em
// (lg T, lg B, lg g). Procedência, hash e o que citar estão em
// atmosphere_data/potekhin_magnetic_h/PROVENIENCIA.json.
//
// Num ponto quente sobre o polo magnético B é radial, então o ângulo com B é o
// próprio ângulo de emissão. A profundidade vertical de onde o fóton escapa
// sai de tau_inclinado = 1:
//
//     tau_escape(mu) = mu * K0/K(theta) = mu * (mu^2 + a (1 - mu^2))
//
// que devolve mu quando a = 1, ou seja o caso isotrópico continua lá dentro.
struct MagneticAnisotropy {
    struct Entry { double log_t, log_b, log_g, ratio; };
    std::vector<Entry> entries;

    bool loaded() const { return !entries.empty(); }

    void load(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open anisotropy table: " + path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#' || line[0] == 'l') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream fields(line);
            Entry entry{};
            if (fields >> entry.log_t >> entry.log_b >> entry.log_g >> entry.ratio) {
                entries.push_back(entry);
            }
        }
        if (entries.empty()) throw std::runtime_error("anisotropy table is empty: " + path);
    }

    //: Vizinho mais próximo na grade, com as três coordenadas normalizadas pelo
    //: passo de tabulação para que nenhuma domine a distância. Interpolação
    //: trilinear seria melhor, mas a grade é fina (0,05 em lg T, 0,1 em lg B,
    //: 0,2 em lg g) e a razão varia devagar dentro de uma célula.
    double ratio_at(double log_t, double log_b, double log_g) const {
        if (entries.empty()) return 1.0;
        double best = 1.0, best_distance = std::numeric_limits<double>::infinity();
        for (const Entry& entry : entries) {
            const double dt = (entry.log_t - log_t) / 0.05;
            const double db = (entry.log_b - log_b) / 0.10;
            const double dg = (entry.log_g - log_g) / 0.20;
            const double distance = dt * dt + db * db + dg * dg;
            if (distance < best_distance) {
                best_distance = distance;
                best = entry.ratio;
            }
        }
        return best;
    }
};

// Definidos mais abaixo, junto do leitor de resposta instrumental.
std::uint32_t read_le_u32(std::istream& input);
float read_le_f32(std::istream& input);

// ---------------------------------------------------------------------------
// Espectro NSMAXG interpolado: Ho, Potekhin & Chabrier (2008).
//
// Substitui o corpo negro pelo espectro REAL de hidrogênio parcialmente
// ionizado e magnetizado, com a linha de cíclotron do próton em
// E = 0,0063 B_12 keV incluída de origem — 0,20 keV para a RBS 1223, onde
// medimos a razão 0,155 contra corpo negro.
//
// O que está tabelado é lg(w), com w a razão para um corpo negro de mesma
// T_ef, os dois normalizados ao mesmo integral na banda. Logo **lg w = 0
// devolve o corpo negro exato**, e o modelo aninha. A razão é a mesma para
// fluxo de energia e de fótons, porque ambos se dividem por E, então ela
// multiplica direto a intensidade de fótons daqui.
//
// **E o que esta tabela NÃO tem é o feixe.** O fluxo é um momento da
// intensidade: F = 2 pi int I(mu) mu dmu, um número por energia contra uma
// função de ângulo. A família inteira (1 + a mu)/(1 + 2a/3) produz fluxo
// idêntico para qualquer a, do leque extremo ao lápis extremo — verificado até
// a última casa. Nenhuma interpolação recupera isso, porque a informação não
// está lá. O feixe continua vindo da forma angular do modelo de dois modos,
// com a anisotropia de opacidade de Potekhin & Chabrier.
struct NsmaxgTable {
    std::vector<float> log_b, log_t, log_g, energy, log_w;

    bool loaded() const { return !log_w.empty(); }

    static std::vector<float> read_axis(std::istream& input) {
        const std::uint32_t count = read_le_u32(input);
        std::vector<float> axis(count);
        for (float& value : axis) value = read_le_f32(input);
        return axis;
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open NSMAXG table: " + path);
        char magic[8];
        if (!input.read(magic, 8) || std::string(magic, 8) != "NSMAXG01") {
            throw std::runtime_error("not a NSMAXG table: " + path);
        }
        log_b = read_axis(input);
        log_t = read_axis(input);
        log_g = read_axis(input);
        energy = read_axis(input);
        const std::size_t total = log_b.size() * log_t.size() * log_g.size() * energy.size();
        log_w.resize(total);
        for (float& value : log_w) value = read_le_f32(input);
    }

    //: Índice inferior e peso, presos nas bordas — extrapolar uma equação de
    //: estado seria inventar física fora do que foi calculado.
    static void bracket(const std::vector<float>& axis, double value,
                        std::size_t& low, double& weight) {
        if (axis.size() < 2 || value <= axis.front()) { low = 0; weight = 0.0; return; }
        if (value >= axis.back()) { low = axis.size() - 2; weight = 1.0; return; }
        low = 0;
        while (low + 2 < axis.size() && axis[low + 1] <= value) ++low;
        const double span = axis[low + 1] - axis[low];
        weight = span > 0.0 ? (value - axis[low]) / span : 0.0;
    }

    double log_ratio(double energy_kev, double log_t_value, double log_b_value,
                     double log_g_value) const {
        if (!loaded()) return 0.0;
        std::size_t ib, it, ig, ie;
        double wb, wt, wg, we;
        bracket(log_b, log_b_value, ib, wb);
        bracket(log_t, log_t_value, it, wt);
        bracket(log_g, log_g_value, ig, wg);
        bracket(energy, energy_kev, ie, we);
        const std::size_t ng = log_g.size(), ne = energy.size();
        auto at = [&](std::size_t b, std::size_t t, std::size_t g, std::size_t e) {
            return static_cast<double>(log_w[(((b * log_t.size() + t) * ng + g) * ne) + e]);
        };
        double total = 0.0;
        for (int db = 0; db < 2; ++db) {
            for (int dt = 0; dt < 2; ++dt) {
                for (int dg = 0; dg < 2; ++dg) {
                    for (int de = 0; de < 2; ++de) {
                        const double weight =
                            (db ? wb : 1.0 - wb) * (dt ? wt : 1.0 - wt) *
                            (dg ? wg : 1.0 - wg) * (de ? we : 1.0 - we);
                        if (weight > 0.0) {
                            total += weight * at(ib + db, it + dt, ig + dg, ie + de);
                        }
                    }
                }
            }
        }
        return total;
    }
};

// ---------------------------------------------------------------------------
// Tabela de intensidade do MAGNUS: I(E, mu, theta_B).
//
// É a peça que o NSMAXG não dá. Aquelas tabelas trazem FLUXO, que é um momento
// da intensidade — a família inteira de feixes com o mesmo primeiro momento
// produz fluxo idêntico —, e por isso nenhuma interpolação recupera o feixe de
// lá. O feixe tem de ser calculado, e esta tabela é o formato em que ele chega.
//
// **O que fica gravado é lg w, com w = I / B_E(T_ef)**: a razão para a
// intensidade de corpo negro ISOTRÓPICA de mesma temperatura efetiva. Três
// consequências, e a terceira é a que faz o portão existir:
//
//  1. w é de ordem 1 e suave, ao passo que I varia dez ordens de grandeza ao
//     longo da grade de temperatura, e interpolar isso linearmente é ruim;
//  2. a escala T^4 sai analiticamente, e o motor já sabe fazer corpo negro;
//  3. **w = 1 em toda a tabela devolve o corpo negro isotrópico exato**, porque
//     um corpo negro isotrópico a T_ef emite exatamente sigma T_ef^4. O modelo
//     ANINHA, e uma tabela de zeros vira o portão do leitor: ela tem de
//     reproduzir o motor sem atmosfera nenhuma, dígito a dígito.
//
// E a interpolação é linear em lg w, ou seja LOGARÍTMICA na intensidade. Não é
// detalhe de implementação: interpolar intensidade linearmente em mu perto de
// mu -> 0 borra justamente a região rasante, que é onde lápis e leque se
// distinguem. O formato resolve isso ao guardar o logaritmo.
//
// Os cinco eixos, do mais lento ao mais rápido no arquivo: lg T, lg g, theta_B
// em graus, mu, lg E. O eixo theta_B é o que o formato de cinco colunas do
// X-PSI não tem — e sem ele dois pontos quentes em colatitudes diferentes
// usariam o mesmo feixe, o que num dipolo é falso por dezenas de graus.
struct AtmosphereTable {
    std::vector<float> log_b, log_t, log_g, theta_b_deg, mu, log_e, log_w;
    //: Fluxo médio-em-ângulo por (B,T,g,theta_B,E), como log10 da razão para
    //: corpo negro (sem o eixo mu). Serve ao modo de camadas: dividi-lo para fora
    //: deixa o contínuo em corpo negro e preserva só o feixe.
    std::vector<float> flux_log;

    bool loaded() const { return !log_w.empty(); }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open atmosphere table: " + path);
        char magic[8];
        input.read(magic, 8);
        const std::string tag(magic, 8);
        // MAGNUSI2 traz um eixo de B na frente (para ajustar o campo); MAGNUSI1
        // é o formato de um campo só, tratado como eixo de B de UM ponto — o
        // layout de memória é idêntico, e a interpolação 6D degenera nele.
        const bool has_field_axis = (tag == "MAGNUSI2");
        if (!has_field_axis && tag != "MAGNUSI1") {
            throw std::runtime_error("not a MAGNUS intensity table: " + path);
        }
        log_b = has_field_axis ? NsmaxgTable::read_axis(input)
                               : std::vector<float>{0.0f};
        log_t = NsmaxgTable::read_axis(input);
        log_g = NsmaxgTable::read_axis(input);
        theta_b_deg = NsmaxgTable::read_axis(input);
        mu = NsmaxgTable::read_axis(input);
        log_e = NsmaxgTable::read_axis(input);
        const std::size_t total = log_b.size() * log_t.size() * log_g.size() *
            theta_b_deg.size() * mu.size() * log_e.size();
        if (total == 0) throw std::runtime_error("atmosphere table has an empty axis: " + path);
        log_w.resize(total);
        // O leitor de float é o mesmo do perfil de instrumento e reclama com a
        // mensagem DELE. Aqui o arquivo é outro, e uma mensagem que aponta para
        // o lugar errado custa uma tarde a quem for depurar.
        try {
            for (float& value : log_w) value = read_le_f32(input);
        } catch (const std::exception&) {
            throw std::runtime_error("atmosphere table ends early: " + path);
        }
        if (!input) throw std::runtime_error("atmosphere table ends early: " + path);
        precompute_flux();
    }

    //: Integra o feixe em mu para o fluxo médio (razão para corpo negro) em cada
    //: (B,T,g,theta_B,E). Quadratura trapezoidal nos nós de mu, normalizada pela
    //: mesma quadratura aplicada a w=1 — assim w constante devolve razão 1.
    void precompute_flux() {
        const std::size_t nB = log_b.size(), nT = log_t.size(), nG = log_g.size(),
                          nb = theta_b_deg.size(), nm = mu.size(), ne = log_e.size();
        std::vector<double> wtrap(nm, 0.0);
        if (nm == 1) {
            wtrap[0] = 1.0;
        } else {
            wtrap[0] = 0.5 * (mu[1] - mu[0]);
            wtrap[nm - 1] = 0.5 * (mu[nm - 1] - mu[nm - 2]);
            for (std::size_t m = 1; m + 1 < nm; ++m) wtrap[m] = 0.5 * (mu[m + 1] - mu[m - 1]);
        }
        double denom = 0.0;
        for (std::size_t m = 0; m < nm; ++m) denom += mu[m] * wtrap[m];
        flux_log.assign(nB * nT * nG * nb * ne, 0.0f);
        for (std::size_t f = 0; f < nB; ++f)
        for (std::size_t t = 0; t < nT; ++t)
        for (std::size_t g = 0; g < nG; ++g)
        for (std::size_t b = 0; b < nb; ++b)
        for (std::size_t e = 0; e < ne; ++e) {
            double num = 0.0;
            for (std::size_t m = 0; m < nm; ++m) {
                const double w = std::pow(10.0, static_cast<double>(
                    log_w[((((((f * nT + t) * nG + g) * nb + b) * nm + m) * ne) + e)]));
                num += w * mu[m] * wtrap[m];
            }
            const double ratio = denom > 0.0 ? num / denom : 1.0;
            flux_log[((((f * nT + t) * nG + g) * nb + b) * ne) + e] =
                static_cast<float>(std::log10(std::max(1.0e-30, ratio)));
        }
    }

    //: log10 do fluxo médio-em-ângulo (razão para corpo negro), interpolado nos
    //: cinco eixos sem mu. É o que o modo de camadas subtrai do log_ratio.
    double flux_log_ratio(double energy_kev, double theta_b, double log_t_value,
                          double log_g_value, double log_b_value) const {
        if (flux_log.empty()) return 0.0;
        std::size_t ifld, it, ig, ib, ie;
        double wf, wt, wg, wb, we;
        NsmaxgTable::bracket(log_b, log_b_value, ifld, wf);
        NsmaxgTable::bracket(log_t, log_t_value, it, wt);
        NsmaxgTable::bracket(log_g, log_g_value, ig, wg);
        NsmaxgTable::bracket(theta_b_deg, theta_b, ib, wb);
        NsmaxgTable::bracket(log_e, std::log10(std::max(1.0e-12, energy_kev)), ie, we);
        const std::size_t nt = log_t.size(), ng = log_g.size(),
                          nb = theta_b_deg.size(), ne = log_e.size();
        auto at = [&](std::size_t f, std::size_t t, std::size_t g, std::size_t b,
                      std::size_t e) {
            return static_cast<double>(flux_log[((((f * nt + t) * ng + g) * nb + b) * ne) + e]);
        };
        double total = 0.0;
        for (int df = 0; df < 2; ++df) { const double ff = df ? wf : 1.0 - wf; if (ff == 0.0) continue;
        for (int dt = 0; dt < 2; ++dt) { const double ft = ff * (dt ? wt : 1.0 - wt); if (ft == 0.0) continue;
        for (int dg = 0; dg < 2; ++dg) { const double fg = ft * (dg ? wg : 1.0 - wg); if (fg == 0.0) continue;
        for (int db = 0; db < 2; ++db) { const double fb = fg * (db ? wb : 1.0 - wb); if (fb == 0.0) continue;
        for (int de = 0; de < 2; ++de) { const double weight = fb * (de ? we : 1.0 - we);
            if (weight != 0.0) total += weight * at(ifld + df, it + dt, ig + dg, ib + db, ie + de);
        }}}}}
        return total;
    }

    //: Interpolação nos cinco eixos, presa nas bordas em todos eles — a mesma
    //: disciplina do resto do arquivo: extrapolar uma atmosfera é inventar
    //: física fora do que foi calculado. Os pesos exatamente nulos são pulados,
    //: o que também é o que protege um eixo de um ponto só.
    double log_ratio(double energy_kev, double cos_emission, double theta_b,
                     double log_t_value, double log_g_value,
                     double log_b_value) const {
        if (!loaded()) return 0.0;
        std::size_t ifld, it, ig, ib, im, ie;
        double wf, wt, wg, wb, wm, we;
        NsmaxgTable::bracket(log_b, log_b_value, ifld, wf);
        NsmaxgTable::bracket(log_t, log_t_value, it, wt);
        NsmaxgTable::bracket(log_g, log_g_value, ig, wg);
        NsmaxgTable::bracket(theta_b_deg, theta_b, ib, wb);
        NsmaxgTable::bracket(mu, cos_emission, im, wm);
        NsmaxgTable::bracket(log_e, std::log10(std::max(1.0e-12, energy_kev)), ie, we);
        const std::size_t nt = log_t.size(), ng = log_g.size(),
                          nb = theta_b_deg.size(), nm = mu.size(), ne = log_e.size();
        auto at = [&](std::size_t f, std::size_t t, std::size_t g, std::size_t b,
                      std::size_t m, std::size_t e) {
            return static_cast<double>(
                log_w[((((((f * nt + t) * ng + g) * nb + b) * nm + m) * ne) + e)]);
        };
        double total = 0.0;
        for (int df = 0; df < 2; ++df) {
            const double ff = df ? wf : 1.0 - wf;
            if (ff == 0.0) continue;
            for (int dt = 0; dt < 2; ++dt) {
                const double ft = ff * (dt ? wt : 1.0 - wt);
                if (ft == 0.0) continue;
                for (int dg = 0; dg < 2; ++dg) {
                    const double fg = ft * (dg ? wg : 1.0 - wg);
                    if (fg == 0.0) continue;
                    for (int db = 0; db < 2; ++db) {
                        const double fb = fg * (db ? wb : 1.0 - wb);
                        if (fb == 0.0) continue;
                        for (int dm = 0; dm < 2; ++dm) {
                            const double fm = fb * (dm ? wm : 1.0 - wm);
                            if (fm == 0.0) continue;
                            for (int de = 0; de < 2; ++de) {
                                const double weight = fm * (de ? we : 1.0 - we);
                                if (weight != 0.0) {
                                    total += weight * at(ifld + df, it + dt, ig + dg,
                                                         ib + db, im + dm, ie + de);
                                }
                            }
                        }
                    }
                }
            }
        }
        return total;
    }
};

//: Ângulo entre B e a normal da superfície, num dipolo. Não é parâmetro livre:
//: com B = (B_p/2)(2 cos t r + sin t theta), sai cos(theta_B) = 2 cos t /
//: sqrt(1 + 3 cos^2 t). O que a atmosfera enxerga é o MÓDULO — os dois modos
//: normais não distinguem o sinal de B —, então o resultado vive em [0, 90].
//: Confere: colatitude magnética de 30 graus dá 16,10; a de 120 dá 40,89.
double dipole_theta_b_deg(double cos_magnetic_colatitude) {
    const double c = std::abs(clamp_unit(cos_magnetic_colatitude));
    const double cos_b = 2.0 * c / std::sqrt(1.0 + 3.0 * c * c);
    return std::acos(std::min(1.0, cos_b)) * 180.0 / pi;
}

//: Profundidade vertical de onde emerge um fóton que sai em mu, com a opacidade
//: anisotrópica. a = K0/K1 > 1 quer dizer perpendicular a B menos opaco.
double anisotropic_escape_depth(double mu, double anisotropy) {
    if (!(anisotropy > 0.0) || anisotropy == 1.0) return mu;
    return mu * (mu * mu + anisotropy * (1.0 - mu * mu));
}

//: Profundidade de onde o modo X emerge na normal, em 1 keV. O piso é a
//: profundidade de onde o modo ORDINÁRIO emerge, tau = 1, e não 0: assim
//: h = 1 quer dizer "os dois modos saem juntos", o endurecimento é medido
//: contra o modo ordinário, e a função é contínua na parede da priori. Um
//: degrau ali seria um penhasco para o amostrador justamente onde a cadeia
//: passa mais tempo.
//: De T_X/T_O = ((tau + 2/3)/(1 + 2/3))^(1/4) sai tau = (5/3) h^4 - 2/3.
double emergence_depth(double hardening) {
    if (!(hardening > 1.0)) return 1.0;
    const double h2 = hardening * hardening;
    return h2 * h2 * (5.0 / 3.0) - 2.0 / 3.0;
}

//: Razão de opacidades entre os modos extraordinário e ordinário. Cresce com
//: E^2, porque a supressão do modo X afrouxa com a energia.
double mode_opacity_ratio(double energy_kev, double hardening) {
    const double ratio = energy_kev * energy_kev / emergence_depth(hardening);
    return std::min(1.0, std::max(1.0e-12, ratio));
}

// Temperatura na profundidade óptica tau, aproximação de Eddington.
double grey_temperature_mk(double tau, double effective_mk) {
    return effective_mk * std::pow(0.75 * (tau + 2.0 / 3.0), 0.25);
}

// Intensidade crua que emerge em mu, somando os dois modos. Cada um carrega
// metade do fluxo em profundidade, onde estão em equilíbrio.
double two_mode_raw(double energy_kev, double effective_mk, double cos_emission,
                    double hardening, double anisotropy) {
    if (!(energy_kev > 0.0 && effective_mk > 0.0)) return 0.0;
    const double mu = std::max(1.0e-6, cos_emission);
    const double ratio = mode_opacity_ratio(energy_kev, hardening);
    // A profundidade de escape deixa de ser mu e passa a depender do ângulo com
    // B, pelas opacidades medidas de Potekhin & Chabrier. Com a = 1 volta a ser
    // mu, e todo o resto abaixo continua igual.
    const double depth = anisotropic_escape_depth(mu, anisotropy);
    // Modo ordinário: escala de referência.
    const double ordinary = blackbody_photon_intensity(
        energy_kev, grey_temperature_mk(depth, effective_mk));
    // Modo extraordinário: opacidade menor por "ratio", logo emerge mais fundo
    // e mais quente.
    const double extraordinary = blackbody_photon_intensity(
        energy_kev, grey_temperature_mk(depth / ratio, effective_mk));
    return 0.5 * (ordinary + extraordinary);
}

// Fator que devolve ao modelo o fluxo bolométrico que ele tem de ter.
//
// **A soma crua dos dois modos não conserva energia.** A estrutura cinza
// T^4(tau) = (3/4) T_ef^4 (tau + 2/3) foi deduzida para UMA opacidade, com a
// condição de que o fluxo que emerge seja sigma T_ef^4; o modo ordinário
// sozinho a cumpre exatamente, porque int (3/4)(mu + 2/3) mu dmu = 1/2, que é
// o mesmo do caso isotrópico. O modo X não: emergindo de tau_O = mu / r, para
// r pequeno ele vem de fundo demais e quente demais. Medido aqui, com
// endurecimento 23 e kT_ef = 112 eV, o modelo cru emite 447 vezes o fluxo de um corpo
// negro à mesma T_ef — "T_ef" deixaria de ser temperatura efetiva, e o ajuste
// esconderia o erro encolhendo kT e o raio do ponto.
//
// Numa atmosfera de verdade nada disso acontece porque o perfil de temperatura
// se reajusta: o modo X, escapando fácil, esfria as camadas de onde sai. Impor
// de volta sigma T_ef^4 é a forma barata de representar esse reajuste. A
// atmosfera fica então mexendo só na FORMA — do feixe e do espectro — que é a
// mesma disciplina que beaming_factor segue, e T_ef continua comparável ao
// ajuste de corpo negro.
double two_mode_normalisation(double effective_mk, double hardening,
                              double anisotropy) {
    // Um par (T, r) por avaliação de verossimilhança, e a integral abaixo é
    // desprezível ao lado do traçado de raios; ainda assim, repeti-la para cada
    // ponto da superfície e cada energia seria puro desperdício.
    static thread_local double cached_mk = -1.0, cached_ratio = -1.0,
                              cached_anisotropy = -1.0, cached = 1.0;
    if (effective_mk == cached_mk && hardening == cached_ratio &&
        anisotropy == cached_anisotropy) return cached;

    // Grade logarítmica em energia, larga o bastante para conter tanto a cauda
    // de Wien do modo quente quanto a subida Rayleigh-Jeans do frio.
    constexpr int energy_steps = 320, angle_steps = 24;
    constexpr double log_min = -9.2103403719761836, log_max = 4.6051701859880918;
    const double d_log = (log_max - log_min) / energy_steps;
    double model = 0.0, reference = 0.0;
    for (int i = 0; i < angle_steps; ++i) {
        const double mu = (i + 0.5) / angle_steps;
        for (int j = 0; j < energy_steps; ++j) {
            const double energy = std::exp(log_min + (j + 0.5) * d_log);
            // dE = E d(lnE); o fluxo de energia pesa a intensidade de fótons por E.
            const double weight = mu * energy * energy * d_log;
            model += two_mode_raw(energy, effective_mk, mu, hardening, anisotropy) * weight;
            reference += blackbody_photon_intensity(energy, effective_mk) * weight;
        }
    }
    cached_mk = effective_mk;
    cached_ratio = hardening;
    cached_anisotropy = anisotropy;
    cached = model > 0.0 ? reference / model : 1.0;
    return cached;
}

// A FORMA angular do modelo de dois modos, separada do espectro dele.
//
// Serve para o espectro vir do NSMAXG e o ângulo daqui, que é a única
// combinação possível: um traz o que o outro não tem. A normalização segue a
// mesma disciplina de beaming_factor — int shape(mu) mu dmu = 1/2, o valor do
// caso isotrópico —, então a forma muda o feixe e não mexe no fluxo.
double two_mode_angular_shape(double energy_kev, double effective_mk, double cos_emission,
                              double hardening, double anisotropy) {
    constexpr int nodes = 16;
    double flux = 0.0;
    for (int index = 0; index < nodes; ++index) {
        const double node = (index + 0.5) / nodes;
        flux += two_mode_raw(energy_kev, effective_mk, node, hardening, anisotropy) *
            node / nodes;
    }
    if (!(flux > 0.0)) return 1.0;
    return two_mode_raw(energy_kev, effective_mk, cos_emission, hardening, anisotropy) /
        (2.0 * flux);
}

double two_mode_intensity(double energy_kev, double effective_mk, double cos_emission,
                          double hardening, double anisotropy) {
    return two_mode_raw(energy_kev, effective_mk, cos_emission, hardening, anisotropy) *
        two_mode_normalisation(effective_mk, hardening, anisotropy);
}

bool trace_spectral_contribution(const Config& cfg, const RayTable& rays, double u,
                                 const SpectralSurfaceSample& sample, double arrival_phase,
                                 int image_order, RayContribution& result) {
    const double inclination = deg(cfg.inclination_deg);
    const Vec3 observer{std::sin(inclination), 0.0, std::cos(inclination)};
    const double omega = 2.0 * pi / cfg.period_s;
    const double gravitational_shift = std::sqrt(1.0 - u);
    const double equatorial_beta = 2.0 * pi * cfg.radius_km /
        (cfg.period_s * c_km_s * gravitational_shift);
    double emission_phase = arrival_phase;
    Vec3 normal{};
    double psi0{}, alpha{}, dalpha_dpsi{}, delay_s{};

    // Solve the retarded emission phase because the travel-time correction depends
    // on the ray selected by the rotating surface geometry.
    for (int iteration = 0; iteration < 7; ++iteration) {
        normal = rotate_about_z(sample.normal, emission_phase);
        const double cos_psi0 = clamp_unit(dot(normal, observer));
        psi0 = std::acos(cos_psi0);
        double psi_branch{};
        if (image_order == 0) psi_branch = psi0;
        else if (image_order % 2 == 1) psi_branch = (image_order + 1) * pi - psi0;
        else psi_branch = image_order * pi + psi0;
        if (!rays.invert(psi_branch, alpha, dalpha_dpsi)) return false;
        double traced_psi{}, delay_integral{};
        if (!rays.trace_alpha(alpha, traced_psi, delay_integral)) return false;
        delay_s = cfg.radius_km * delay_integral / c_km_s;
        const double updated_phase = arrival_phase - omega * delay_s;
        if (std::abs(updated_phase - emission_phase) < 1.0e-11) {
            emission_phase = updated_phase;
            break;
        }
        emission_phase = updated_phase;
    }

    // Re-evaluate the converged geometry.
    normal = rotate_about_z(sample.normal, emission_phase);
    const double cos_psi0 = clamp_unit(dot(normal, observer));
    psi0 = std::acos(cos_psi0);
    const double sin_psi0 = std::max(std::sin(psi0), 1.0e-12);
    double psi_branch{};
    if (image_order == 0) psi_branch = psi0;
    else if (image_order % 2 == 1) psi_branch = (image_order + 1) * pi - psi0;
    else psi_branch = image_order * pi + psi0;
    if (!rays.invert(psi_branch, alpha, dalpha_dpsi)) return false;
    double traced_psi{}, delay_integral{};
    if (!rays.trace_alpha(alpha, traced_psi, delay_integral)) return false;
    delay_s = cfg.radius_km * delay_integral / c_km_s;

    const Vec3 toward_observer = unit(observer + (-cos_psi0) * normal);
    const double direction_sign = image_order % 2 == 0 ? 1.0 : -1.0;
    const Vec3 emitted = std::cos(alpha) * normal +
        direction_sign * std::sin(alpha) * toward_observer;
    Vec3 velocity_dir{-normal.y, normal.x, 0.0};
    const double sin_theta = norm(velocity_dir);
    velocity_dir = unit(velocity_dir);
    const double beta = std::min(0.95, equatorial_beta * sin_theta);
    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    const double doppler = 1.0 / (gamma * (1.0 - beta * dot(emitted, velocity_dir)));
    const double jacobian = std::sin(alpha) * std::abs(dalpha_dpsi) / sin_psi0;
    const double radius_cm = cfg.radius_km * 1.0e5;
    const double distance_cm = cfg.distance_kpc * kpc_to_cm;
    const double solid_angle = sample.weight * (radius_cm * radius_cm) /
        (distance_cm * distance_cm) * std::max(0.0, std::cos(alpha)) * jacobian / (1.0 - u);

    result = {arrival_phase, emission_phase, jacobian, solid_angle,
              gravitational_shift * doppler, delay_s, sample.temperature_mk,
              sample.theta_b_deg, std::max(0.0, std::cos(alpha)), sample.spot_id,
              image_order};
    return result.solid_angle_sr > 0.0 && result.energy_shift_g > 0.0;
}

struct InstrumentPoint {
    double energy_kev{};
    double area_cm2{};
    double mean_kev{};
    double sigma_kev{};
    double efficiency{};
};

struct RedistributionEntry {
    std::int32_t channel{};
    float energy_min_kev{};
    float energy_max_kev{};
    float probability{};
};

struct RedistributionRow {
    double true_energy_kev{};
    double area_cm2{};
    std::vector<RedistributionEntry> entries;
};

struct ResponseKernel {
    double area_cm2{};
    std::vector<double> channel_probability;
};

struct EventChannel {
    std::int32_t pi_channel{};
    double detected_energy_kev{};
    double cumulative_probability{};
};

struct EventResponseKernel {
    double area_cm2{};
    double detection_efficiency{};
    std::vector<EventChannel> channels;
};

std::uint32_t read_le_u32(std::istream& input) {
    char bytes[4];
    if (!input.read(bytes, 4)) throw std::runtime_error("truncated instrument response");
    return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[0])) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[1])) << 8) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[2])) << 16) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[3])) << 24);
}

float read_le_f32(std::istream& input) {
    return std::bit_cast<float>(read_le_u32(input));
}

double read_le_f64(std::istream& input) {
    const std::uint64_t low = read_le_u32(input);
    const std::uint64_t high = read_le_u32(input);
    return std::bit_cast<double>(low | (high << 32));
}

// Transmissão do meio interestelar entre a estrela e o telescópio.
//
// A tabela de seção de choque vem de scripts/build_absorption_table.py, que a
// extrai do TBabs do XSPEC (Wilms, Allen & McCray 2000). Como o modelo é
// puramente multiplicativo, T(E) = exp(-sigma(E) N_H), uma única tabela serve
// para qualquer coluna de hidrogênio e N_H pode variar livremente no ajuste.
class InterstellarAbsorption {
  public:
    explicit InterstellarAbsorption(const std::string& path) {
        if (path.empty()) return;
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open absorption table: " + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream values(line);
            double energy{}, sigma{};
            if (values >> energy >> sigma && energy > 0.0 && sigma >= 0.0) {
                energies_.push_back(energy);
                sigmas_.push_back(sigma);
            }
        }
        if (energies_.size() < 2) {
            throw std::runtime_error("absorption table is too short: " + path);
        }
        loaded_ = true;
    }

    bool loaded() const { return loaded_; }

    // A tabela é carregada uma vez por processo; a coluna entra a cada avaliação,
    // porque N_H é parâmetro do ajuste e muda a cada amostra do MCMC.
    double transmission(double energy_kev, double column_cm2) const {
        if (!loaded_ || !(column_cm2 > 0.0)) return 1.0;
        return std::exp(-column_cm2 * sigma(energy_kev));
    }

  private:
    double sigma(double energy_kev) const {
        // Fora da tabela extrapolar seria inventar: abaixo do primeiro ponto a
        // absorção já é praticamente total, acima do último é desprezível.
        if (energy_kev <= energies_.front()) return sigmas_.front();
        if (energy_kev >= energies_.back()) return sigmas_.back();
        const auto upper = std::lower_bound(energies_.begin(), energies_.end(), energy_kev);
        const std::size_t hi = static_cast<std::size_t>(upper - energies_.begin());
        const std::size_t lo = hi - 1;
        const double e0 = energies_[lo], e1 = energies_[hi];
        const double s0 = sigmas_[lo], s1 = sigmas_[hi];
        // Entre bordas de absorção a seção de choque cai como lei de potência,
        // então interpola-se em log-log; linear subestimaria os vales.
        if (!(s0 > 0.0 && s1 > 0.0 && e1 > e0)) {
            const double f = (e1 > e0) ? (energy_kev - e0) / (e1 - e0) : 0.0;
            return s0 + f * (s1 - s0);
        }
        const double f = std::log(energy_kev / e0) / std::log(e1 / e0);
        return s0 * std::pow(s1 / s0, f);
    }

    bool loaded_{false};
    std::vector<double> energies_;
    std::vector<double> sigmas_;
};

class InstrumentResponse {
  public:
    explicit InstrumentResponse(const Config& cfg)
        : ideal_(cfg.instrument_profile.empty() && cfg.instrument_response.empty()) {
        if (ideal_) return;
        if (!cfg.instrument_response.empty()) {
            load_full_response(cfg.instrument_response);
            return;
        }
        std::ifstream input(cfg.instrument_profile);
        if (!input) throw std::runtime_error("cannot open instrument profile: " + cfg.instrument_profile);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream values(line);
            InstrumentPoint point;
            if (values >> point.energy_kev >> point.area_cm2 >> point.mean_kev
                       >> point.sigma_kev >> point.efficiency) {
                points_.push_back(point);
            }
        }
        if (points_.size() < 2) throw std::runtime_error("instrument profile has insufficient rows");
        std::sort(points_.begin(), points_.end(), [](const auto& a, const auto& b) {
            return a.energy_kev < b.energy_kev;
        });
    }

    bool ideal() const { return ideal_; }
    bool full_rmf() const { return !rows_.empty(); }
    std::size_t response_rows() const { return rows_.size(); }

    InstrumentPoint at(double energy) const {
        if (ideal_) return {energy, 1.0, energy, 0.0, 1.0};
        if (energy < points_.front().energy_kev || energy > points_.back().energy_kev) return {};
        const auto upper = std::lower_bound(points_.begin(), points_.end(), energy,
            [](const InstrumentPoint& point, double value) { return point.energy_kev < value; });
        if (upper == points_.begin()) return *upper;
        if (upper == points_.end()) return points_.back();
        const auto& hi = *upper;
        const auto& lo = *(upper - 1);
        const double f = (energy - lo.energy_kev) / (hi.energy_kev - lo.energy_kev);
        return {energy,
                lo.area_cm2 + f * (hi.area_cm2 - lo.area_cm2),
                lo.mean_kev + f * (hi.mean_kev - lo.mean_kev),
                std::max(1.0e-6, lo.sigma_kev + f * (hi.sigma_kev - lo.sigma_kev)),
                std::clamp(lo.efficiency + f * (hi.efficiency - lo.efficiency), 0.0, 1.0)};
    }

    ResponseKernel kernel(double true_energy, const std::vector<double>& output_edges) const {
        ResponseKernel result;
        result.channel_probability.assign(output_edges.size() - 1, 0.0);
        if (ideal_) {
            result.area_cm2 = 1.0;
            const auto bin = std::upper_bound(output_edges.begin(), output_edges.end(), true_energy);
            if (bin != output_edges.begin() && bin != output_edges.end()) {
                result.channel_probability[static_cast<std::size_t>(bin - output_edges.begin() - 1)] = 1.0;
            }
            return result;
        }
        if (!full_rmf()) {
            const InstrumentPoint point = at(true_energy);
            result.area_cm2 = point.area_cm2;
            if (!(point.area_cm2 > 0.0 && point.efficiency > 0.0)) return result;
            const double scale = std::sqrt(2.0) * point.sigma_kev;
            for (std::size_t channel = 0; channel + 1 < output_edges.size(); ++channel) {
                const double probability = 0.5 *
                    (std::erf((output_edges[channel + 1] - point.mean_kev) / scale) -
                     std::erf((output_edges[channel] - point.mean_kev) / scale));
                result.channel_probability[channel] =
                    point.efficiency * std::max(0.0, probability);
            }
            return result;
        }
        if (true_energy < rows_.front().true_energy_kev ||
            true_energy > rows_.back().true_energy_kev) return result;
        const auto upper = std::lower_bound(rows_.begin(), rows_.end(), true_energy,
            [](const RedistributionRow& row, double value) { return row.true_energy_kev < value; });
        std::size_t hi = static_cast<std::size_t>(upper - rows_.begin());
        if (hi == 0) hi = 1;
        if (hi >= rows_.size()) hi = rows_.size() - 1;
        const std::size_t lo = hi - 1;
        const double span = rows_[hi].true_energy_kev - rows_[lo].true_energy_kev;
        const double fraction = span > 0.0 ?
            std::clamp((true_energy - rows_[lo].true_energy_kev) / span, 0.0, 1.0) : 0.0;
        result.area_cm2 = rows_[lo].area_cm2 + fraction *
            (rows_[hi].area_cm2 - rows_[lo].area_cm2);
        auto add_row = [&](const RedistributionRow& row, double weight) {
            for (const auto& entry : row.entries) {
                const double e_lo = static_cast<double>(entry.energy_min_kev);
                const double e_hi = static_cast<double>(entry.energy_max_kev);
                const double span_ch = e_hi - e_lo;
                if (!(span_ch > 0.0)) {
                    // Canal nativo degenerado: cai no bin do centro.
                    const double measured_energy = 0.5 * (e_lo + e_hi);
                    const auto bin = std::upper_bound(output_edges.begin(), output_edges.end(), measured_energy);
                    if (bin == output_edges.begin() || bin == output_edges.end()) continue;
                    const std::size_t index = static_cast<std::size_t>(bin - output_edges.begin() - 1);
                    result.channel_probability[index] += weight * entry.probability;
                    continue;
                }
                // Reparte a probabilidade do canal nativo entre os bins de saída
                // pela fração de sobreposição de [e_lo,e_hi] com cada bin. Jogar
                // tudo no bin do centro (o que se fazia) quantiza a redistribuição
                // e serrilha o espectro dobrado quando a largura do canal nativo
                // não casa com a do bin de saída; repartir por sobreposição
                // conserva a probabilidade e alisa o degrau.
                auto start = std::upper_bound(output_edges.begin(), output_edges.end(), e_lo);
                if (start != output_edges.begin()) --start;
                for (auto edge = start; edge + 1 < output_edges.end(); ++edge) {
                    const double b_lo = *edge;
                    const double b_hi = *(edge + 1);
                    if (b_lo >= e_hi) break;
                    const double lo = std::max(e_lo, b_lo);
                    const double hi = std::min(e_hi, b_hi);
                    if (hi <= lo) continue;
                    const std::size_t index = static_cast<std::size_t>(edge - output_edges.begin());
                    result.channel_probability[index] +=
                        weight * entry.probability * (hi - lo) / span_ch;
                }
            }
        };
        add_row(rows_[lo], 1.0 - fraction);
        add_row(rows_[hi], fraction);
        return result;
    }

    EventResponseKernel event_kernel(double true_energy) const {
        EventResponseKernel result;
        if (ideal_) {
            result.area_cm2 = 1.0;
            result.detection_efficiency = 1.0;
            result.channels.push_back({-1, true_energy, 1.0});
            return result;
        }
        if (!full_rmf()) {
            const InstrumentPoint point = at(true_energy);
            result.area_cm2 = point.area_cm2;
            result.detection_efficiency = point.efficiency;
            if (point.area_cm2 > 0.0 && point.efficiency > 0.0) {
                result.channels.push_back({-1, point.mean_kev, point.efficiency});
            }
            return result;
        }
        if (true_energy < rows_.front().true_energy_kev ||
            true_energy > rows_.back().true_energy_kev) return result;
        const auto upper = std::lower_bound(rows_.begin(), rows_.end(), true_energy,
            [](const RedistributionRow& row, double value) { return row.true_energy_kev < value; });
        std::size_t hi = static_cast<std::size_t>(upper - rows_.begin());
        if (hi == 0) hi = 1;
        if (hi >= rows_.size()) hi = rows_.size() - 1;
        const std::size_t lo = hi - 1;
        const double span = rows_[hi].true_energy_kev - rows_[lo].true_energy_kev;
        const double fraction = span > 0.0 ?
            std::clamp((true_energy - rows_[lo].true_energy_kev) / span, 0.0, 1.0) : 0.0;
        result.area_cm2 = rows_[lo].area_cm2 + fraction *
            (rows_[hi].area_cm2 - rows_[lo].area_cm2);
        double cumulative = 0.0;
        auto append_row = [&](const RedistributionRow& row, double weight) {
            for (const auto& entry : row.entries) {
                const double probability = weight * entry.probability;
                if (!(probability > 0.0)) continue;
                cumulative += probability;
                result.channels.push_back({entry.channel,
                    0.5 * (static_cast<double>(entry.energy_min_kev) + entry.energy_max_kev),
                    cumulative});
            }
        };
        append_row(rows_[lo], 1.0 - fraction);
        append_row(rows_[hi], fraction);
        result.detection_efficiency = cumulative;
        return result;
    }

  private:
    void load_full_response(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open full instrument response: " + path);
        char magic[8];
        if (!input.read(magic, 8) || std::string(magic, 8) != std::string("PLSRMF3\0", 8)) {
            throw std::runtime_error("invalid PULSARIS sparse RMF: " + path);
        }
        const std::uint32_t version = read_le_u32(input);
        const std::uint32_t row_count = read_le_u32(input);
        const std::uint32_t channel_count = read_le_u32(input);
        if (version != 3 || row_count < 2 || row_count > 1000000 ||
            channel_count == 0 || channel_count > 1000000) {
            throw std::runtime_error("unsupported or unsafe PULSARIS sparse RMF");
        }
        std::unordered_map<std::int32_t, std::pair<float, float>> channel_bounds;
        channel_bounds.reserve(channel_count);
        for (std::uint32_t channel_index = 0; channel_index < channel_count; ++channel_index) {
            const auto channel = static_cast<std::int32_t>(read_le_u32(input));
            channel_bounds[channel] = {read_le_f32(input), read_le_f32(input)};
        }
        rows_.reserve(row_count);
        std::size_t total_entries = 0;
        for (std::uint32_t row_index = 0; row_index < row_count; ++row_index) {
            RedistributionRow row;
            row.true_energy_kev = read_le_f64(input);
            row.area_cm2 = read_le_f64(input);
            const std::uint32_t entry_count = read_le_u32(input);
            total_entries += entry_count;
            if (entry_count > 1000000 || total_entries > 100000000) {
                throw std::runtime_error("instrument response exceeds safety limits");
            }
            row.entries.reserve(entry_count);
            for (std::uint32_t entry_index = 0; entry_index < entry_count; ++entry_index) {
                RedistributionEntry entry;
                entry.channel = static_cast<std::int32_t>(read_le_u32(input));
                entry.probability = read_le_f32(input);
                const auto bounds = channel_bounds.find(entry.channel);
                if (bounds == channel_bounds.end()) {
                    throw std::runtime_error("RMF entry references an unknown detector channel");
                }
                entry.energy_min_kev = bounds->second.first;
                entry.energy_max_kev = bounds->second.second;
                if (entry.probability > 0.0f && entry.energy_max_kev >= entry.energy_min_kev) {
                    row.entries.push_back(entry);
                }
            }
            rows_.push_back(std::move(row));
        }
        if (!std::is_sorted(rows_.begin(), rows_.end(), [](const auto& a, const auto& b) {
                return a.true_energy_kev < b.true_energy_kev;
            })) throw std::runtime_error("instrument response energies are not monotonic");
    }

    bool ideal_{};
    std::vector<InstrumentPoint> points_;
    std::vector<RedistributionRow> rows_;
};

void apply_temporal_resolution(std::vector<double>& rate, int time_bins, int energy_bins,
                               double period_s, double resolution_us) {
    const double bin_seconds = period_s / time_bins;
    const double resolution_seconds = resolution_us * 1.0e-6;
    if (resolution_seconds <= bin_seconds) return;
    int window = std::max(1, static_cast<int>(std::lround(resolution_seconds / bin_seconds)));
    window = std::min(window, time_bins);
    std::vector<double> smoothed(rate.size(), 0.0);
    const int left = window / 2;
    for (int it = 0; it < time_bins; ++it) {
        for (int offset = 0; offset < window; ++offset) {
            const int source_time = (it + offset - left + time_bins) % time_bins;
            for (int ie = 0; ie < energy_bins; ++ie) {
                smoothed[static_cast<std::size_t>(it) * energy_bins + ie] +=
                    rate[static_cast<std::size_t>(source_time) * energy_bins + ie] / window;
            }
        }
    }
    rate.swap(smoothed);
}

struct SyntheticEvent {
    double arrival_time_s{};
    double recorded_time_s{};
    double true_energy_kev{};
    std::int32_t pi_channel{};
    double detected_energy_kev{};
};

void write_event_list(const Config& cfg, const std::vector<double>& photon_flux,
                      const std::vector<double>& energy_edges,
                      const InstrumentResponse& response) {
    const int energy_bins = static_cast<int>(energy_edges.size()) - 1;
    const double energy_width = energy_edges[1] - energy_edges[0];
    const double phase_bin_s = cfg.period_s / cfg.time_bins;
    const auto full_cycles = static_cast<std::uint64_t>(std::floor(cfg.exposure_s / cfg.period_s));
    const double partial_s = cfg.exposure_s - static_cast<double>(full_cycles) * cfg.period_s;
    std::vector<EventResponseKernel> kernels;
    kernels.reserve(energy_bins);
    for (int ie = 0; ie < energy_bins; ++ie) {
        kernels.push_back(response.event_kernel(0.5 * (energy_edges[ie] + energy_edges[ie + 1])));
    }

    double expected_candidates = 0.0;
    for (int it = 0; it < cfg.time_bins; ++it) {
        const double partial_overlap = std::max(0.0, std::min(phase_bin_s,
            partial_s - it * phase_bin_s));
        const double represented_time = static_cast<double>(full_cycles) * phase_bin_s + partial_overlap;
        for (int ie = 0; ie < energy_bins; ++ie) {
            const auto& kernel = kernels[ie];
            const double incident_density = photon_flux[static_cast<std::size_t>(it) * energy_bins + ie];
            expected_candidates += incident_density * energy_width * kernel.area_cm2 *
                kernel.detection_efficiency * represented_time;
        }
    }
    if (!std::isfinite(expected_candidates) || expected_candidates > cfg.max_events) {
        throw std::runtime_error("expected event count " + number(expected_candidates) +
            " exceeds max-events=" + std::to_string(cfg.max_events) +
            "; reduce exposure, distance/temperature, or raise the limit");
    }

    std::mt19937_64 rng(cfg.random_seed);
    std::uniform_real_distribution<double> unit_random(0.0, 1.0);
    std::vector<SyntheticEvent> candidates;
    candidates.reserve(static_cast<std::size_t>(std::ceil(expected_candidates * 1.1 + 32.0)));
    auto draw_events = [&](int it, int ie, double duration, auto&& draw_time) {
        if (!(duration > 0.0)) return;
        const auto& kernel = kernels[ie];
        if (!(kernel.area_cm2 > 0.0 && kernel.detection_efficiency > 0.0) ||
            kernel.channels.empty()) return;
        const double incident_density = photon_flux[static_cast<std::size_t>(it) * energy_bins + ie];
        const double expected = incident_density * energy_width * kernel.area_cm2 *
            kernel.detection_efficiency * duration;
        if (!(expected > 0.0)) return;
        std::poisson_distribution<std::uint64_t> poisson(expected);
        const std::uint64_t count = poisson(rng);
        if (candidates.size() + count > cfg.max_events) {
            throw std::runtime_error("sampled event count exceeds max-events safety limit");
        }
        for (std::uint64_t event_index = 0; event_index < count; ++event_index) {
            const double selector = unit_random(rng) * kernel.detection_efficiency;
            const auto selected = std::lower_bound(kernel.channels.begin(), kernel.channels.end(), selector,
                [](const EventChannel& channel, double value) {
                    return channel.cumulative_probability < value;
                });
            const EventChannel& channel = selected == kernel.channels.end() ?
                kernel.channels.back() : *selected;
            const double true_energy = energy_edges[ie] + unit_random(rng) * energy_width;
            const double arrival_time = draw_time();
            candidates.push_back({arrival_time, arrival_time, true_energy,
                                  channel.pi_channel, channel.detected_energy_kev});
        }
    };

    if (full_cycles > 0) {
        std::uniform_int_distribution<std::uint64_t> cycle_draw(0, full_cycles - 1);
        for (int it = 0; it < cfg.time_bins; ++it) {
            for (int ie = 0; ie < energy_bins; ++ie) {
                draw_events(it, ie, static_cast<double>(full_cycles) * phase_bin_s, [&]() {
                    return static_cast<double>(cycle_draw(rng)) * cfg.period_s +
                        (it + unit_random(rng)) * phase_bin_s;
                });
            }
        }
    }
    for (int it = 0; it < cfg.time_bins; ++it) {
        const double overlap = std::max(0.0, std::min(phase_bin_s, partial_s - it * phase_bin_s));
        if (!(overlap > 0.0)) continue;
        for (int ie = 0; ie < energy_bins; ++ie) {
            draw_events(it, ie, overlap, [&]() {
                return static_cast<double>(full_cycles) * cfg.period_s + it * phase_bin_s +
                    unit_random(rng) * overlap;
            });
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.arrival_time_s < b.arrival_time_s;
    });
    const double dead_time_s = cfg.instrument_dead_time_us * 1.0e-6;
    const double resolution_s = cfg.instrument_time_resolution_us * 1.0e-6;
    std::vector<SyntheticEvent> detected;
    detected.reserve(candidates.size());
    double next_live_time = -INFINITY;
    for (auto event : candidates) {
        if (event.arrival_time_s < next_live_time) continue;
        next_live_time = event.arrival_time_s + dead_time_s;
        if (resolution_s > 0.0) {
            event.recorded_time_s = std::round(event.arrival_time_s / resolution_s) * resolution_s;
            event.recorded_time_s = std::clamp(event.recorded_time_s, 0.0, cfg.exposure_s);
        }
        detected.push_back(event);
    }
    std::stable_sort(detected.begin(), detected.end(), [](const auto& a, const auto& b) {
        return a.recorded_time_s < b.recorded_time_s;
    });

    std::cout << "# PULSARIS_SYNTHETIC_EVENTS_V1\n"
              << "# folded_in_phase=false\n"
              << "# instrument=" << cfg.instrument_id << "\n"
              << "# exposure_s=" << number(cfg.exposure_s) << "\n"
              << "# period_s=" << number(cfg.period_s) << "\n"
              << "# mass_solar=" << number(cfg.mass_solar) << "\n"
              << "# radius_km=" << number(cfg.radius_km) << "\n"
              << "# distance_kpc=" << number(cfg.distance_kpc) << "\n"
              << "# inclination_deg=" << number(cfg.inclination_deg) << "\n"
              << "# max_images=" << cfg.max_images << "\n"
              << "# spot_count=" << cfg.spots.size() << "\n"
              << "# phase_reference_s=0\n"
              << "# seed=" << cfg.random_seed << "\n"
              << "# expected_before_dead_time=" << number(expected_candidates) << "\n"
              << "# generated_before_dead_time=" << candidates.size() << "\n"
              << "# detected_after_dead_time=" << detected.size() << "\n"
              << "# time_resolution_us=" << number(cfg.instrument_time_resolution_us) << "\n"
              << "# dead_time_us=" << number(cfg.instrument_dead_time_us) << "\n";
    for (std::size_t index = 0; index < cfg.spots.size(); ++index) {
        const auto& spot = cfg.spots[index];
        std::cout << "# spot" << index + 1 << "_theta_deg=" << number(spot.theta_deg) << "\n"
                  << "# spot" << index + 1 << "_phi_deg=" << number(spot.phi_deg) << "\n"
                  << "# spot" << index + 1 << "_radius_deg=" << number(spot.radius_deg) << "\n"
                  << "# spot" << index + 1 << "_kt_keV="
                  << number(spot.temperature_mk * kt_kev_per_mk) << "\n";
    }
    std::cout << "TIME,TRUE_ENERGY_KEV,PI,DETECTED_ENERGY_KEV\n";
    std::cout << std::setprecision(12);
    for (const auto& event : detected) {
        std::cout << event.recorded_time_s << ',' << event.true_energy_kev << ','
                  << event.pi_channel << ',' << event.detected_energy_kev << '\n';
    }
}

void write_spectral_grid(const Config& cfg, const RayTable& rays, double u,
                         const InstrumentResponse* prepared_response = nullptr,
                         const InterstellarAbsorption* prepared_absorption = nullptr) {
    const auto started = std::chrono::steady_clock::now();
    // Eixo do dipolo, no referencial do corpo. O ângulo entre B e a normal sai
    // dele ponto a ponto, e não muda com a fase: normal e eixo giram juntos.
    const double magnetic_colatitude = deg(cfg.magnetic_colatitude_deg);
    const double magnetic_azimuth = deg(cfg.magnetic_azimuth_deg);
    const Vec3 magnetic_axis{std::sin(magnetic_colatitude) * std::cos(magnetic_azimuth),
                             std::sin(magnetic_colatitude) * std::sin(magnetic_azimuth),
                             std::cos(magnetic_colatitude)};
    std::vector<SpectralSurfaceSample> surface;
    // O fundo de estrela inteira entra primeiro, com spot_id = -1, e os spots
    // vêm por cima: os ladrilhos do fundo que caem dentro de um spot já foram
    // descartados por sample_full_sphere, então cada pedaço de superfície emite
    // uma vez só — o fundo onde não há spot, o spot onde há.
    //
    // Em modo OVERLAY (--spot-overlay) o fundo NÃO é subtraído sob os spots: a
    // atmosfera cobre a estrela inteira e o corpo negro do spot é somado POR
    // CIMA, uma componente extra localizada em vez de substituir a atmosfera.
    if (cfg.base_temperature_mk > 0.0) {
        const int base_bands = std::clamp(2 * cfg.surface_rings, 16, 60);
        const std::vector<Spot> subtract = cfg.spot_overlay ? std::vector<Spot>{} : cfg.spots;
        const auto base = sample_full_sphere(cfg.base_temperature_mk, base_bands, subtract);
        // Lei T(theta): a>0 liga a distribuição dipolar (polos quentes, equador
        // frio); a=0 devolve o fundo uniforme. cos(theta_mag)=normal·eixo_B.
        // Lei de Perez-Azorin por polo. Um polo (base_temperature2<=0): a lei
        // simetrica de sempre. DOIS polos (Hambaryan): polo 1 no eixo do dipolo
        // (+B), polo 2 no eixo -B GIRADO por beta (pole2_tilt) no plano
        // magneto-rotacional. Cada polo contribui so no seu hemisferio proximo
        // (c>0), somando em T^4; piso T_min unico. beta=0 => polos antipodais
        // (cada ponto sob um so polo, exatamente como a lei de um dipolo).
        const double a1 = cfg.temperature_peaking;
        const double t_pole1_4 = std::pow(cfg.base_temperature_mk, 4.0);
        const bool two_pole = cfg.base_temperature2_mk > 0.0;
        const double a2 = two_pole ? cfg.temperature_peaking2 : a1;
        const double t_pole2_4 = std::pow(two_pole ? cfg.base_temperature2_mk
                                                    : cfg.base_temperature_mk, 4.0);
        const double t_min4 = std::pow(std::max(0.0, cfg.temperature_min_frac)
                                       * cfg.base_temperature_mk, 4.0);
        // Eixo do polo 2: -B girado por beta em torno de (B x z), no plano de B e
        // do eixo de rotacao z. beta=0 => -B (antipodal). B x z_hat = (By,-Bx,0).
        Vec3 axis2{-magnetic_axis.x, -magnetic_axis.y, -magnetic_axis.z};
        if (two_pole && std::abs(cfg.pole2_tilt_deg) > 1.0e-9) {
            Vec3 k{magnetic_axis.y, -magnetic_axis.x, 0.0};
            const double kn = std::sqrt(dot(k, k));
            if (kn > 1.0e-12) {
                k = (1.0 / kn) * k;
                const double b = deg(cfg.pole2_tilt_deg), cb = std::cos(b), sb = std::sin(b);
                const Vec3 v = axis2;
                const Vec3 kxv{k.y * v.z - k.z * v.y, k.z * v.x - k.x * v.z,
                               k.x * v.y - k.y * v.x};
                const double kdv = dot(k, v);
                axis2 = {v.x * cb + kxv.x * sb + k.x * kdv * (1.0 - cb),
                         v.y * cb + kxv.y * sb + k.y * kdv * (1.0 - cb),
                         v.z * cb + kxv.z * sb + k.z * kdv * (1.0 - cb)};
            }
        }
        auto lobe = [](double c, double a, double tp4) -> double {
            if (c <= 0.0) return 0.0;
            const double c2 = c * c, s2 = std::max(0.0, 1.0 - c2);
            return (a > 0.0) ? tp4 * c2 / (c2 + a * s2) : tp4;
        };
        for (const auto& sample : base) {
            const double cmag = dot(sample.normal, magnetic_axis);
            double temperature = cfg.base_temperature_mk;
            if (a1 > 0.0 || two_pole) {
                const double c1 = cmag;
                const double c2 = two_pole ? dot(sample.normal, axis2) : -cmag;
                const double t4 = t_min4 + lobe(c1, a1, t_pole1_4)
                                  + lobe(c2, a2, t_pole2_4);
                temperature = std::pow(std::max(1.0e-8, t4), 0.25);
            }
            surface.push_back({sample.normal, sample.weight, temperature,
                               dipole_theta_b_deg(cmag), -1});
        }
    }
    for (std::size_t spot_id = 0; spot_id < cfg.spots.size(); ++spot_id) {
        const auto samples = sample_spot(cfg.spots[spot_id], cfg.surface_rings);
        for (const auto& sample : samples) {
            surface.push_back({sample.normal, sample.weight, cfg.spots[spot_id].temperature_mk,
                               dipole_theta_b_deg(dot(sample.normal, magnetic_axis)),
                               static_cast<int>(spot_id)});
        }
    }

    const double energy_width = (cfg.energy_max_kev - cfg.energy_min_kev) / cfg.energy_bins;
    std::vector<double> energy_edges(static_cast<std::size_t>(cfg.energy_bins) + 1);
    std::vector<double> energy_centers(cfg.energy_bins);
    for (int ie = 0; ie <= cfg.energy_bins; ++ie) {
        energy_edges[ie] = cfg.energy_min_kev + ie * energy_width;
        if (ie < cfg.energy_bins) energy_centers[ie] = energy_edges[ie] + 0.5 * energy_width;
    }

    std::vector<double> times(cfg.time_bins);
    std::vector<double> flux(static_cast<std::size_t>(cfg.time_bins) * cfg.energy_bins, 0.0);
    std::vector<double> band_rate(cfg.time_bins, 0.0);
    std::vector<double> mean_spectrum(cfg.energy_bins, 0.0);
    int images_used = 1;

    // Anisotropia da opacidade na fotosfera, de Potekhin & Chabrier (2003).
    // Sem tabela ou sem campo o valor é 1, e a atmosfera volta a ser a cinza
    // isotrópica — que é o modelo incapaz de fazer leque.
    //
    // A gravidade usa g = GM(1+z)/R^2, que é a definição com que aquelas
    // tabelas foram indexadas; ignorar o redshift daria a linha errada.
    const bool emits = !cfg.spots.empty() || cfg.base_temperature_mk > 0.0;
    // Temperatura representativa da estrela para a anisotropia (um escalar por
    // estrela, simplificação já existente): o fundo se houver, senão o spot.
    const double representative_mk = cfg.base_temperature_mk > 0.0
        ? cfg.base_temperature_mk
        : (cfg.spots.empty() ? 0.0 : cfg.spots.front().temperature_mk);
    double anisotropy = 1.0;
    if (!cfg.anisotropy_table.empty() && cfg.magnetic_field_g > 0.0 && emits) {
        static thread_local MagneticAnisotropy table;
        static thread_local std::string loaded_from;
        if (loaded_from != cfg.anisotropy_table) {
            table.load(cfg.anisotropy_table);
            loaded_from = cfg.anisotropy_table;
        }
        const double mass_cm = cfg.mass_solar * 1.4766250385e5;    // GM/c^2, em cm
        const double radius_cm_surface = cfg.radius_km * 1.0e5;
        const double redshift = 1.0 /
            std::sqrt(std::max(1.0e-6, 1.0 - 2.0 * mass_cm / radius_cm_surface));
        const double gravity = 8.98755178736817e20 * mass_cm * redshift /
            (radius_cm_surface * radius_cm_surface);
        anisotropy = table.ratio_at(
            std::log10(std::max(1.0, representative_mk * 1.0e6)),
            std::log10(cfg.magnetic_field_g),
            std::log10(std::max(1.0, gravity)));
    }

    // O espectro real, se houver tabela e campo. Ele entra NO LUGAR do corpo
    // negro; o feixe continua sendo o do modelo de dois modos, porque o
    // espectro não carrega ângulo nenhum.
    static thread_local NsmaxgTable nsmaxg;
    static thread_local std::string nsmaxg_loaded_from;
    const bool use_nsmaxg = !cfg.nsmaxg_table.empty() && cfg.magnetic_field_g > 0.0 &&
        emits;

    // A tabela do MAGNUS. Ela é a atmosfera CALCULADA, e substitui espectro e
    // feixe de uma vez porque traz os dois; por isso vem antes das outras na
    // cadeia, e nenhuma forma angular se multiplica por cima dela — isso seria
    // contar o feixe duas vezes.
    static thread_local AtmosphereTable atmosphere;
    static thread_local std::string atmosphere_loaded_from;
    const bool use_atmosphere_table = !cfg.atmosphere_table.empty() && emits;
    if (use_atmosphere_table && atmosphere_loaded_from != cfg.atmosphere_table) {
        atmosphere.load(cfg.atmosphere_table);
        atmosphere_loaded_from = cfg.atmosphere_table;
    }

    double log_b_value = 0.0, log_g_value = 0.0;
    if (use_nsmaxg || use_atmosphere_table) {
        if (use_nsmaxg && nsmaxg_loaded_from != cfg.nsmaxg_table) {
            nsmaxg.load(cfg.nsmaxg_table);
            nsmaxg_loaded_from = cfg.nsmaxg_table;
        }
        const double mass_cm = cfg.mass_solar * 1.4766250385e5;
        const double radius_cm_surface = cfg.radius_km * 1.0e5;
        const double redshift = 1.0 /
            std::sqrt(std::max(1.0e-6, 1.0 - 2.0 * mass_cm / radius_cm_surface));
        log_b_value = cfg.magnetic_field_g > 0.0 ? std::log10(cfg.magnetic_field_g) : 0.0;
        log_g_value = std::log10(std::max(1.0, 8.98755178736817e20 * mass_cm *
            redshift / (radius_cm_surface * radius_cm_surface)));
    }

    for (int it = 0; it < cfg.time_bins; ++it) {
        times[it] = (it + 0.5) * cfg.period_s / cfg.time_bins;
        const double arrival_phase = 2.0 * pi * times[it] / cfg.period_s;
        std::vector<RayContribution> contributions;
        contributions.reserve(surface.size() * static_cast<std::size_t>(cfg.max_images));
        for (const auto& sample : surface) {
            for (int image_order = 0; image_order < cfg.max_images; ++image_order) {
                RayContribution contribution;
                if (trace_spectral_contribution(cfg, rays, u, sample, arrival_phase,
                                                image_order, contribution)) {
                    contributions.push_back(contribution);
                    images_used = std::max(images_used, image_order + 1);
                }
            }
        }
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            const double observed_energy = energy_centers[ie];
            double value = 0.0;
            for (const auto& contribution : contributions) {
                const double emitted_energy = observed_energy / contribution.energy_shift_g;
                // A atmosfera substitui corpo negro E feixe de uma vez: ela
                // produz os dois. A linha continua por cima, porque é uma
                // feição espectral que o modelo cinza não tem como gerar.
                double base;
                if (cfg.blackbody_spots && contribution.spot_id >= 0) {
                    // Spot de superfície condensada: corpo negro puro (mais mole
                    // que a atmosfera), com o feixe empírico (isotrópico por
                    // padrão). A área já saiu do fundo, então não há dupla conta.
                    base = blackbody_photon_intensity(emitted_energy,
                                                      contribution.temperature_mk) *
                        beaming_factor(contribution.cos_emission, cfg.beaming_a,
                                       cfg.beaming_b);
                } else if (use_atmosphere_table) {
                    // Atmosfera de espessura efetiva f: mantém o feixe inteiro e
                    // escala o endurecimento espectral (o fluxo médio-em-ângulo)
                    // por f. f=1 é a atmosfera cheia (log w exato do portão);
                    // f=0 é corpo negro + feixe (camadas). O dado escolhe f.
                    const double lt = std::log10(std::max(1.0,
                        contribution.temperature_mk * 1.0e6));
                    double logw = atmosphere.log_ratio(
                        emitted_energy, contribution.cos_emission,
                        contribution.theta_b_deg, lt, log_g_value, log_b_value);
                    if (cfg.atmosphere_fraction < 1.0) {
                        logw -= (1.0 - cfg.atmosphere_fraction) *
                            atmosphere.flux_log_ratio(emitted_energy,
                                contribution.theta_b_deg, lt, log_g_value, log_b_value);
                    }
                    base = blackbody_photon_intensity(emitted_energy,
                                                      contribution.temperature_mk) *
                        std::pow(10.0, logw);
                } else if (use_nsmaxg) {
                    // Espectro do NSMAXG, feixe do modelo de dois modos. Cada
                    // um traz o que o outro não tem, e a forma angular é
                    // normalizada para não mexer no fluxo que a tabela fixou.
                    const double hardening = std::max(1.0, cfg.atmosphere_hardening);
                    base = blackbody_photon_intensity(emitted_energy,
                                                      contribution.temperature_mk) *
                        std::pow(10.0, nsmaxg.log_ratio(
                            emitted_energy,
                            std::log10(std::max(1.0, contribution.temperature_mk * 1.0e6)),
                            log_b_value, log_g_value)) *
                        two_mode_angular_shape(emitted_energy, contribution.temperature_mk,
                                               contribution.cos_emission, hardening,
                                               anisotropy);
                } else if (cfg.atmosphere_hardening >= 1.0) {
                    base = two_mode_intensity(emitted_energy, contribution.temperature_mk,
                                              contribution.cos_emission,
                                              cfg.atmosphere_hardening, anisotropy);
                } else {
                    base = blackbody_photon_intensity(emitted_energy,
                                                      contribution.temperature_mk) *
                        beaming_factor(contribution.cos_emission, cfg.beaming_a,
                                       cfg.beaming_b);
                }
                // Modo cíclotron: a energia de REPOUSO da linha vem do campo
                // (E_cp = 0.63*(B/1e14) keV, próton) em vez de ser livre. O
                // redshift entra sozinho pelo g-shift (emitted = observado*(1+z)),
                // então a feição observada fica em E_cp/(1+z) e vincula B e z.
                const double line_E = cfg.line_cyclotron
                    ? 0.63 * cfg.magnetic_field_g / 1.0e14 : cfg.line_energy_kev;
                const double emitted_intensity = base *
                    line_transmission(emitted_energy, line_E,
                                      cfg.line_width_kev, cfg.line_depth) *
                    line_transmission(emitted_energy, cfg.line2_energy_kev,
                                      cfg.line2_width_kev, cfg.line2_depth);
                // I_N(E)/E^2 is invariant, hence the g^2 factor for photon intensity.
                value += contribution.energy_shift_g * contribution.energy_shift_g *
                    emitted_intensity * contribution.solid_angle_sr;
            }
            flux[static_cast<std::size_t>(it) * cfg.energy_bins + ie] = value;
            band_rate[it] += value * energy_width;
            mean_spectrum[ie] += value / cfg.time_bins;
        }
    }

    // A absorção acontece no caminho, entre a estrela e o telescópio: aplica-se
    // ao fluxo incidente, de modo que tanto a lista de eventos sintética quanto
    // a grade ajustada a herdem de um único ponto.
    std::unique_ptr<InterstellarAbsorption> owned_absorption;
    if (!prepared_absorption && !cfg.absorption_table.empty()) {
        owned_absorption = std::make_unique<InterstellarAbsorption>(cfg.absorption_table);
    }
    const InterstellarAbsorption* absorption =
        prepared_absorption ? prepared_absorption : owned_absorption.get();
    const double column_cm2 = cfg.nh_1e22 * 1.0e22;
    const bool absorbed = absorption && absorption->loaded() && column_cm2 > 0.0;
    if (absorbed) {
        for (int it = 0; it < cfg.time_bins; ++it) {
            for (int ie = 0; ie < cfg.energy_bins; ++ie) {
                const double factor = absorption->transmission(energy_centers[ie], column_cm2);
                flux[static_cast<std::size_t>(it) * cfg.energy_bins + ie] *= factor;
            }
        }
        std::fill(band_rate.begin(), band_rate.end(), 0.0);
        std::fill(mean_spectrum.begin(), mean_spectrum.end(), 0.0);
        for (int it = 0; it < cfg.time_bins; ++it) {
            for (int ie = 0; ie < cfg.energy_bins; ++ie) {
                const double value = flux[static_cast<std::size_t>(it) * cfg.energy_bins + ie];
                band_rate[it] += value * energy_width;
                mean_spectrum[ie] += value / cfg.time_bins;
            }
        }
    }

    std::unique_ptr<InstrumentResponse> owned_response;
    if (!prepared_response) owned_response = std::make_unique<InstrumentResponse>(cfg);
    const InstrumentResponse& response = prepared_response ? *prepared_response : *owned_response;
    if (cfg.event_list) {
        write_event_list(cfg, flux, energy_edges, response);
        return;
    }
    // LIMITAÇÃO CONHECIDA, medida e deixada como está. A grade de energia
    // VERDADEIRA é a mesma da banda ajustada, então fótons emitidos fora dela
    // nunca entram como fonte, e o espalhamento PARA DENTRO da banda não é
    // representado. O espalhamento para fora está correto: a soma abaixo
    // simplesmente descarta os canais que caem fora, sem renormalizar — a
    // renormalização é que seria o erro, porque inventaria conservação.
    //
    // Medido com o ARF real desta observação (0844140101, EPIC-pn, 2067
    // pontos de 0,05 a 16 keV) contra um espectro térmico absorvido de
    // kT = 112 eV: 1,02% dos fótons colhidos nascem abaixo de 0,15 keV e
    // 0,0003% acima de 2 keV — e só a fração DESSES que a redistribuição
    // empurra para dentro da banda é que falta. Sub-percentual aqui. Para uma
    // fonte mais dura, ou banda mais estreita, deixa de ser.
    std::vector<double> detected_rate(flux.size(), 0.0);
    std::vector<ResponseKernel> response_kernels;
    response_kernels.reserve(cfg.energy_bins);
    for (double true_energy : energy_centers) {
        response_kernels.push_back(response.kernel(true_energy, energy_edges));
    }
    for (int it = 0; it < cfg.time_bins; ++it) {
        for (int true_bin = 0; true_bin < cfg.energy_bins; ++true_bin) {
            const double incident_density = flux[static_cast<std::size_t>(it) * cfg.energy_bins + true_bin];
            const ResponseKernel& kernel = response_kernels[true_bin];
            if (!(incident_density > 0.0 && kernel.area_cm2 > 0.0)) continue;
            if (response.ideal()) {
                detected_rate[static_cast<std::size_t>(it) * cfg.energy_bins + true_bin] += incident_density;
                continue;
            }
            const double accepted_rate = incident_density * energy_width * kernel.area_cm2;
            for (int channel = 0; channel < cfg.energy_bins; ++channel) {
                detected_rate[static_cast<std::size_t>(it) * cfg.energy_bins + channel] +=
                    accepted_rate * kernel.channel_probability[channel] / energy_width;
            }
        }
    }
    apply_temporal_resolution(detected_rate, cfg.time_bins, cfg.energy_bins, cfg.period_s,
                              cfg.instrument_time_resolution_us);
    std::vector<double> detected_band_rate(cfg.time_bins, 0.0);
    std::vector<double> mean_detected_spectrum(cfg.energy_bins, 0.0);
    const double dead_time_s = cfg.instrument_dead_time_us * 1.0e-6;
    for (int it = 0; it < cfg.time_bins; ++it) {
        double raw_rate = 0.0;
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            raw_rate += detected_rate[static_cast<std::size_t>(it) * cfg.energy_bins + ie] * energy_width;
        }
        const double live_fraction = 1.0 / (1.0 + raw_rate * dead_time_s);
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            double& value = detected_rate[static_cast<std::size_t>(it) * cfg.energy_bins + ie];
            value *= live_fraction;
            detected_band_rate[it] += value * energy_width;
            mean_detected_spectrum[ie] += value / cfg.time_bins;
        }
    }

    const double mean_band_rate = std::accumulate(band_rate.begin(), band_rate.end(), 0.0) /
        band_rate.size();
    const double mean_detected_rate = std::accumulate(detected_band_rate.begin(),
        detected_band_rate.end(), 0.0) / detected_band_rate.size();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    auto write_vector = [](const std::vector<double>& values) {
        std::cout << '[';
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << number(values[i]);
        }
        std::cout << ']';
    };

    std::cout << "{\"status\":\"ok\",\"product\":\"spectral_grid\""
              << ",\"model\":\"Schwarzschild+Doppler+blackbody"
              << (cfg.line_depth > 0.0 ? "+gaussian_line" : "")
              << (use_atmosphere_table ? "+magnus_intensity_table"
                  : use_nsmaxg ? "+nsmaxg_spectrum_two_mode_beam"
                  : cfg.atmosphere_hardening >= 1.0 ? "+two_mode_grey_atmosphere"
                  : cfg.beaming_b != 0.0 ? "+quadratic_beaming"
                  : cfg.beaming_a != 0.0 ? "+linear_beaming" : "+isotropic")
              << "\""
              << ",\"compactness\":" << number(u)
              << ",\"redshift\":" << number(1.0 / std::sqrt(1.0 - u) - 1.0)
              << ",\"base_kt_keV\":" << number(cfg.base_temperature_mk * kt_kev_per_mk)
              << ",\"temperature_peaking\":" << number(cfg.temperature_peaking)
              << ",\"atmosphere_fraction\":" << number(cfg.atmosphere_fraction)
              << ",\"images_used\":" << images_used
              << ",\"period_s\":" << number(cfg.period_s)
              << ",\"distance_kpc\":" << number(cfg.distance_kpc)
              << ",\"energy_unit\":\"keV\""
              << ",\"time_unit\":\"s\""
              << ",\"flux_unit\":\"ph cm^-2 s^-1 keV^-1\""
              << ",\"rate_unit\":\"ph cm^-2 s^-1\""
              << ",\"mean_band_rate\":" << number(mean_band_rate)
              << ",\"instrument\":{\"id\":\"" << cfg.instrument_id
              << "\",\"label\":\"" << cfg.instrument_label
              << "\",\"response_applied\":" << (response.ideal() ? "false" : "true")
              << ",\"redistribution\":\"" << (response.ideal() ? "ideal" :
                    response.full_rmf() ? "full_sparse_OGIP_RMF" : "Gaussian_from_RMF_moments") << "\""
              << ",\"response_rows\":" << response.response_rows()
              << ",\"time_resolution_us\":" << number(cfg.instrument_time_resolution_us)
              << ",\"dead_time_us\":" << number(cfg.instrument_dead_time_us) << "}"
              << ",\"absorption\":{\"applied\":" << (absorbed ? "true" : "false")
              << ",\"nh_1e22\":" << number(cfg.nh_1e22)
              << ",\"model\":\"" << (absorbed ? "TBabs_wilm" : "none") << "\"}"
              << ",\"atmosphere\":{\"applied\":" << (use_atmosphere_table ? "true" : "false")
              << ",\"table\":\"" << cfg.atmosphere_table
              << "\",\"magnetic_colatitude_deg\":" << number(cfg.magnetic_colatitude_deg)
              << ",\"theta_b_deg_by_spot\":[";
    // O ângulo entre B e a normal no CENTRO de cada ponto quente. Sai relatado
    // porque é a diferença que o eixo theta_B existe para carregar: num dipolo
    // alinhado, pontos em colatitude 30 e 120 graus não veem o mesmo feixe.
    for (std::size_t spot_index = 0; spot_index < cfg.spots.size(); ++spot_index) {
        if (spot_index) std::cout << ',';
        const double spot_theta = deg(cfg.spots[spot_index].theta_deg);
        const double spot_phi = deg(cfg.spots[spot_index].phi_deg);
        const Vec3 centre{std::sin(spot_theta) * std::cos(spot_phi),
                          std::sin(spot_theta) * std::sin(spot_phi),
                          std::cos(spot_theta)};
        std::cout << number(dipole_theta_b_deg(dot(centre, magnetic_axis)));
    }
    std::cout << "]}"
              << ",\"detected_rate_unit\":\"count s^-1 keV^-1\""
              << ",\"mean_detected_rate\":" << number(mean_detected_rate)
              << ",\"execution_ms\":" << number(elapsed_ms)
              << ",\"energy_edges_keV\":";
    write_vector(energy_edges);
    std::cout << ",\"energy_centers_keV\":";
    write_vector(energy_centers);
    std::cout << ",\"time_s\":";
    write_vector(times);
    std::cout << ",\"band_rate\":";
    write_vector(band_rate);
    std::cout << ",\"mean_spectrum\":";
    write_vector(mean_spectrum);
    std::cout << ",\"detected_band_rate\":";
    write_vector(detected_band_rate);
    std::cout << ",\"mean_detected_spectrum\":";
    write_vector(mean_detected_spectrum);
    std::cout << ",\"photon_flux\":[";
    for (int it = 0; it < cfg.time_bins; ++it) {
        if (it) std::cout << ',';
        std::cout << '[';
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            if (ie) std::cout << ',';
            std::cout << number(flux[static_cast<std::size_t>(it) * cfg.energy_bins + ie]);
        }
        std::cout << ']';
    }
    std::cout << "],\"detected_count_rate\":[";
    for (int it = 0; it < cfg.time_bins; ++it) {
        if (it) std::cout << ',';
        std::cout << '[';
        for (int ie = 0; ie < cfg.energy_bins; ++ie) {
            if (ie) std::cout << ',';
            std::cout << number(detected_rate[static_cast<std::size_t>(it) * cfg.energy_bins + ie]);
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
}

int run_fit_worker(const Config& base) {
    // Long-lived protocol used by the Bayesian fitting backend.  Instrument
    // calibration is immutable during a fit and is therefore loaded once.
    const InstrumentResponse response(base);
    // Calibração e tabela de absorção são imutáveis durante o ajuste; só a
    // coluna N_H varia, e ela chega no fim de cada linha de parâmetros.
    std::unique_ptr<InterstellarAbsorption> absorption;
    if (!base.absorption_table.empty()) {
        absorption = std::make_unique<InterstellarAbsorption>(base.absorption_table);
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        try {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream input(line);
            Config cfg = base;
            double phase_offset{};
            if (!(input >> cfg.mass_solar >> cfg.radius_km >> cfg.inclination_deg >> phase_offset)) {
                throw std::runtime_error("fit worker expected mass,radius,inclination,phase");
            }
            for (auto& spot : cfg.spots) {
                double relative_phi{}, temperature_mk{};
                if (!(input >> spot.theta_deg >> relative_phi >> spot.radius_deg >> temperature_mk)) {
                    throw std::runtime_error("fit worker received an incomplete spot vector");
                }
                spot.phi_deg = phase_offset + relative_phi;
                spot.temperature_mk = temperature_mk;
            }
            // A ordem dos opcionais é fixa: linha, feixe, N_H. Cada bloco só
            // aparece quando o worker foi iniciado com o --fit- correspondente,
            // e é por isso que essas flags existem — o protocolo é posicional e
            // não distinguiria um bloco ausente de um valor deslocado.
            if (base.fit_line) {
                if (!(input >> cfg.line_energy_kev >> cfg.line_width_kev >> cfg.line_depth)) {
                    throw std::runtime_error("fit worker expected the gaussian line triplet");
                }
                if (!(cfg.line_width_kev > 0.0 && cfg.line_depth >= 0.0)) {
                    throw std::runtime_error("fit worker received an invalid gaussian line");
                }
            }
            if (base.fit_line2) {
                if (!(input >> cfg.line2_energy_kev >> cfg.line2_width_kev >> cfg.line2_depth)) {
                    throw std::runtime_error("fit worker expected the second gaussian line triplet");
                }
                if (!(cfg.line2_width_kev > 0.0 && cfg.line2_depth >= 0.0)) {
                    throw std::runtime_error("fit worker received an invalid second gaussian line");
                }
            }
            if (base.fit_atmosphere) {
                if (!(input >> cfg.atmosphere_hardening)) {
                    throw std::runtime_error("fit worker expected the atmosphere hardening");
                }
                if (!(cfg.atmosphere_hardening >= 1.0)) {
                    throw std::runtime_error("fit worker received a non-positive "
                                             "opacity ratio");
                }
            }
            if (base.fit_log_field) {
                double log_field{};
                if (!(input >> log_field)) {
                    throw std::runtime_error("fit worker expected lg B");
                }
                cfg.magnetic_field_g = std::pow(10.0, log_field);
            }
            if (base.fit_base_temperature) {
                double base_kt_kev{};
                if (!(input >> base_kt_kev)) {
                    throw std::runtime_error("fit worker expected the base kT");
                }
                if (!(base_kt_kev > 0.0)) {
                    throw std::runtime_error("fit worker received a non-positive base kT");
                }
                cfg.base_temperature_mk = base_kt_kev / kt_kev_per_mk;
            }
            if (base.fit_temperature_peaking) {
                double peaking{};
                if (!(input >> peaking)) {
                    throw std::runtime_error("fit worker expected the temperature peaking a");
                }
                if (!(peaking >= 0.0)) {
                    throw std::runtime_error("fit worker received a negative peaking a");
                }
                cfg.temperature_peaking = peaking;
            }
            if (base.fit_base_temperature2) {
                double base_kt2_kev{};
                if (!(input >> base_kt2_kev)) {
                    throw std::runtime_error("fit worker expected the second-pole kT");
                }
                if (!(base_kt2_kev > 0.0)) {
                    throw std::runtime_error("fit worker received a non-positive second-pole kT");
                }
                cfg.base_temperature2_mk = base_kt2_kev / kt_kev_per_mk;
            }
            if (base.fit_temperature_peaking2) {
                double peaking2{};
                if (!(input >> peaking2)) {
                    throw std::runtime_error("fit worker expected the second-pole peaking a2");
                }
                if (!(peaking2 >= 0.0)) {
                    throw std::runtime_error("fit worker received a negative second-pole a2");
                }
                cfg.temperature_peaking2 = peaking2;
            }
            if (base.fit_pole2_tilt) {
                double tilt{};
                if (!(input >> tilt)) {
                    throw std::runtime_error("fit worker expected the second-pole tilt beta");
                }
                cfg.pole2_tilt_deg = tilt;
            }
            if (base.fit_atmosphere_fraction) {
                double frac{};
                if (!(input >> frac)) {
                    throw std::runtime_error("fit worker expected the atmosphere fraction f");
                }
                if (!(frac >= 0.0 && frac <= 1.0)) {
                    throw std::runtime_error("fit worker received an atmosphere fraction "
                                             "outside [0,1]");
                }
                cfg.atmosphere_fraction = frac;
            }
            if (base.fit_magnetic_colatitude) {
                double colat{};
                if (!(input >> colat)) {
                    throw std::runtime_error("fit worker expected the magnetic colatitude");
                }
                if (!(colat >= 0.0 && colat <= 180.0)) {
                    throw std::runtime_error("fit worker received a magnetic colatitude "
                                             "outside [0,180]");
                }
                cfg.magnetic_colatitude_deg = colat;
            }
            if (base.fit_magnetic_azimuth) {
                double azim{};
                if (!(input >> azim)) {
                    throw std::runtime_error("fit worker expected the magnetic azimuth");
                }
                cfg.magnetic_azimuth_deg = azim;
            }
            if (base.fit_beaming) {
                if (!(input >> cfg.beaming_a >> cfg.beaming_b)) {
                    throw std::runtime_error("fit worker expected the beaming pair a,b");
                }
                // A positividade de 1 + a*mu + b*mu^2 em [0,1] é responsabilidade
                // da priori; aqui só se recusa o que zeraria a normalização.
                if (!(1.0 + 2.0 * cfg.beaming_a / 3.0 + cfg.beaming_b / 2.0 > 0.0)) {
                    throw std::runtime_error("fit worker received a beaming pair whose "
                                             "normalisation is not positive");
                }
            }

            // N_H é opcional e vem por último: linhas antigas, sem ele, seguem
            // válidas e mantêm a coluna da configuração base.
            double nh{};
            if (input >> nh) {
                if (!(nh >= 0.0)) throw std::runtime_error("fit worker received a negative nH");
                cfg.nh_1e22 = nh;
            }
            if (!(cfg.mass_solar > 0.0 && cfg.radius_km > 0.0) ||
                !(cfg.inclination_deg >= 0.0 && cfg.inclination_deg <= 180.0)) {
                throw std::runtime_error("fit worker parameters are outside physical bounds");
            }
            const double u = 2.0 * gm_sun_over_c2_km * cfg.mass_solar / cfg.radius_km;
            if (!(u > 0.0 && u < 0.985)) {
                throw std::runtime_error("fit worker compactness is outside physical bounds");
            }
            RayTable rays(u);
            write_spectral_grid(cfg, rays, u, &response, absorption.get());
        } catch (const std::exception& error) {
            std::string message = error.what();
            std::replace(message.begin(), message.end(), '"', '\'');
            std::cout << "{\"status\":\"error\",\"message\":\"" << message << "\"}\n";
        }
        std::cout.flush();
    }
    return 0;
}

int run(const Config& cfg) {
    if (cfg.fit_worker) return run_fit_worker(cfg);
    const auto started = std::chrono::steady_clock::now();
    const double u = 2.0 * gm_sun_over_c2_km * cfg.mass_solar / cfg.radius_km;
    if (u >= 0.985) throw std::runtime_error("radius is too close to or inside the Schwarzschild radius");
    RayTable rays(u);
    if (cfg.transfer_table) {
        write_transfer_table(cfg, rays, u);
        return 0;
    }
    if (cfg.spectral_grid) {
        write_spectral_grid(cfg, rays, u);
        return 0;
    }
    if (cfg.render_image) {
        render_bmp(cfg, rays, u);
        return 0;
    }
    std::vector<SurfaceSample> surface;
    for (const auto& spot : cfg.spots) {
        auto part = sample_spot(spot, cfg.surface_rings);
        surface.insert(surface.end(), part.begin(), part.end());
    }

    const double inclination = deg(cfg.inclination_deg);
    const Vec3 observer{std::sin(inclination), 0.0, std::cos(inclination)};
    const double equatorial_beta = 2.0 * pi * cfg.radius_km / (cfg.period_s * c_km_s * std::sqrt(1.0 - u));
    std::vector<double> flux(cfg.phase_samples, 0.0);
    int images_used = 1;

    for (int ip = 0; ip < cfg.phase_samples; ++ip) {
        const double phase = 2.0 * pi * ip / (cfg.phase_samples - 1);
        const double cp = std::cos(phase), sp = std::sin(phase);
        double total = 0.0;
        for (const auto& sample : surface) {
            const Vec3 n{cp * sample.normal.x - sp * sample.normal.y,
                         sp * sample.normal.x + cp * sample.normal.y,
                         sample.normal.z};
            const double cos_psi0 = clamp_unit(dot(n, observer));
            const double psi0 = std::acos(cos_psi0);
            const double sin_psi0 = std::max(std::sin(psi0), 1.0e-10);
            const Vec3 toward_observer = unit(observer + (-cos_psi0) * n);
            Vec3 velocity_dir{-n.y, n.x, 0.0};
            const double sin_theta = norm(velocity_dir);
            velocity_dir = unit(velocity_dir);
            const double beta = std::min(0.95, equatorial_beta * sin_theta);
            const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);

            for (int image_index = 0; image_index < cfg.max_images; ++image_index) {
                double psi_branch;
                if (image_index == 0) psi_branch = psi0;
                else if (image_index % 2 == 1) psi_branch = (image_index + 1) * pi - psi0;
                else psi_branch = image_index * pi + psi0;
                double alpha{}, dalpha_dpsi{};
                if (!rays.invert(psi_branch, alpha, dalpha_dpsi)) continue;
                images_used = std::max(images_used, image_index + 1);
                const double direction_sign = image_index % 2 == 0 ? 1.0 : -1.0;
                const Vec3 emitted = std::cos(alpha) * n + direction_sign * std::sin(alpha) * toward_observer;
                const double cos_xi = dot(emitted, velocity_dir);
                const double doppler = 1.0 / (gamma * (1.0 - beta * cos_xi));
                const double jacobian = std::sin(alpha) * std::abs(dalpha_dpsi) / sin_psi0;
                total += sample.weight * sample.intensity * std::pow(doppler, 5.0) *
                         std::max(0.0, std::cos(alpha)) * jacobian;
            }
        }
        flux[ip] = total;
    }

    const double mean = std::accumulate(flux.begin(), flux.end(), 0.0) / flux.size();
    if (!(mean > 0.0)) throw std::runtime_error("no observable flux for this configuration");
    for (double& f : flux) f /= mean;
    const auto [minimum, maximum] = std::minmax_element(flux.begin(), flux.end());
    double rms_sum = 0.0;
    for (double f : flux) rms_sum += (f - 1.0) * (f - 1.0);
    const double rms = std::sqrt(rms_sum / flux.size());
    const double pulsed_fraction = (*maximum - *minimum) / (*maximum + *minimum);
    const double redshift = 1.0 / std::sqrt(1.0 - u) - 1.0;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    std::cout << "{\"status\":\"ok\",\"model\":\"Schwarzschild+Doppler\","
              << "\"compactness\":" << number(u)
              << ",\"redshift\":" << number(redshift)
              << ",\"psi_max\":" << number(rays.psi_max())
              << ",\"multiple_images\":" << (rays.psi_max() > pi ? "true" : "false")
              << ",\"images_used\":" << images_used
              << ",\"frequency_hz\":" << number(1.0 / cfg.period_s)
              << ",\"execution_ms\":" << number(elapsed_ms)
              << ",\"metrics\":{\"max_min_ratio\":" << number(*maximum / *minimum)
              << ",\"mean\":1.0,\"rms\":" << number(rms)
              << ",\"pulsed_fraction\":" << number(pulsed_fraction) << "},\"curve\":[";
    for (std::size_t i = 0; i < flux.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "[" << number(static_cast<double>(i) / (flux.size() - 1)) << ',' << number(flux[i]) << "]";
    }
    std::cout << "]}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "MAGNUS engine error: " << error.what() << '\n';
        return 2;
    }
}
