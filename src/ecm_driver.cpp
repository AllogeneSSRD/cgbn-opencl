#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <fstream>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#else
#include <unistd.h>
#endif

#include <gmp.h>

#include "opencl_ecm_entry.h"
#include "ecm.h"
#include "cgbn_stage1.h"

static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}

class ExprParser {
public:
    explicit ExprParser(const std::string &input) : text(input), pos(0), error(false) {}

    bool parse(mpz_t out) {
        skip_ws();
        parse_expr(out);
        skip_ws();
        if (!error && pos != text.size()) {
            set_error("unexpected trailing characters");
        }
        return !error;
    }

    const std::string &message() const { return message_text; }

private:
    const std::string &text;
    size_t pos;
    bool error;
    std::string message_text;

    void set_error(const std::string &msg) {
        if (!error) {
            error = true;
            message_text = msg;
        }
    }

    void skip_ws() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
    }

    bool match(char ch) {
        skip_ws();
        if (pos < text.size() && text[pos] == ch) {
            ++pos;
            return true;
        }
        return false;
    }

    bool peek(char ch) {
        skip_ws();
        return pos < text.size() && text[pos] == ch;
    }

    void parse_expr(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_term(lhs);
        while (!error) {
            if (match('+')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_add(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('-')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_sub(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_term(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_power(lhs);
        while (!error) {
            if (match('*')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                mpz_mul(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('/')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                if (mpz_sgn(rhs) == 0) {
                    set_error("division by zero");
                    mpz_clear(rhs);
                    break;
                }
                if (!mpz_divisible_p(lhs, rhs)) {
                    set_error("division is not exact (result would not be an integer)");
                    mpz_clear(rhs);
                    break;
                }
                mpz_divexact(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_power(mpz_t out) {
        mpz_t base;
        mpz_init(base);
        parse_unary(base);
        if (error) {
            mpz_clear(base);
            return;
        }

        if (match('^')) {
            mpz_t exponent;
            mpz_init(exponent);
            parse_power(exponent);
            if (error) {
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            if (mpz_sgn(exponent) < 0 || !mpz_fits_ulong_p(exponent)) {
                set_error("exponent must be a non-negative integer that fits in unsigned long");
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            unsigned long exp = mpz_get_ui(exponent);
            mpz_pow_ui(out, base, exp);
            mpz_clear(base);
            mpz_clear(exponent);
            return;
        }

        mpz_set(out, base);
        mpz_clear(base);
    }

    void parse_unary(mpz_t out) {
        skip_ws();
        if (match('+')) {
            parse_unary(out);
            return;
        }
        if (match('-')) {
            parse_unary(out);
            mpz_neg(out, out);
            return;
        }
        parse_primary(out);
    }

    void parse_primary(mpz_t out) {
        skip_ws();
        if (match('(')) {
            parse_expr(out);
            if (!match(')')) {
                set_error("missing closing parenthesis");
            }
            return;
        }

        size_t start = pos;
        bool saw_digit = false;
        while (pos < text.size()) {
            unsigned char ch = static_cast<unsigned char>(text[pos]);
            if (std::isalnum(ch) || ch == 'x' || ch == 'X') {
                saw_digit = true;
                ++pos;
            } else {
                break;
            }
        }
        if (!saw_digit) {
            set_error("expected number or parenthesized expression");
            return;
        }

        std::string token = text.substr(start, pos - start);
        if (mpz_set_str(out, token.c_str(), 0) != 0) {
            set_error(std::string("invalid integer token: ") + token);
        }
    }
};

// Compute batch product s = prod_{p<=B1} p^{floor(log_p(B1))}
static bool compute_batch_s(mpz_t s, double B1){
    static const unsigned MAX_HEIGHT = 32;

    if(B1 < 2.0) {
        mpz_set_ui(s, 1);
        return true;
    }

    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }

    const uint32_t limit = (uint32_t)limit64;

    std::vector<char> sieve((size_t)limit + 1u, 1);
    sieve[0] = sieve[1] = 0;
    for (uint32_t p = 2; (uint64_t)p * (uint64_t)p <= limit; ++p) {
        if (!sieve[p]) {
            continue;
        }
        for (uint64_t q = (uint64_t)p * (uint64_t)p; q <= limit; q += p) {
            sieve[(size_t)q] = 0;
        }
    }

    mpz_t acc[MAX_HEIGHT];
    mpz_t ppz;
    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_init(acc[j]);
    }
    mpz_init(ppz);

    unsigned i = 0;
    for (uint32_t pi = 2; pi <= limit; ++pi) {
        if (!sieve[pi]) {
            continue;
        }

        uint64_t pp = pi;
        const uint64_t maxpp = limit / pi;
        while (pp <= maxpp) {
            pp *= pi;
        }

        mpz_import(ppz, 1, -1, sizeof(pp), 0, 0, &pp);

        if ((i & 1u) == 0u) {
            mpz_set(acc[0], ppz);
        } else {
            mpz_mul(acc[0], acc[0], ppz);
        }

        unsigned j = 0;
        while ((i & (1u << j)) != 0u) {
            if (j + 1 >= MAX_HEIGHT - 1) {
                for (unsigned k = 0; k < MAX_HEIGHT; ++k) {
                    mpz_clear(acc[k]);
                }
                mpz_clear(ppz);
                return false;
            }

            if ((i & (1u << (j + 1))) == 0u) {
                mpz_swap(acc[j + 1], acc[j]);
            } else {
                mpz_mul(acc[j + 1], acc[j + 1], acc[j]);
            }
            mpz_set_ui(acc[j], 1);
            ++j;
        }

        ++i;
    }

    if (i == 0) {
        mpz_set_ui(s, 1);
    } else {
        mpz_set(s, acc[0]);
        for (unsigned j = 1; j < MAX_HEIGHT && mpz_cmp_ui(acc[j], 0) != 0; ++j) {
            mpz_mul(s, s, acc[j]);
        }
    }

    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_clear(acc[j]);
    }
    mpz_clear(ppz);
    return true;
}

static bool parse_sigma_arg(const std::string &arg, uint32_t *sigma_out) {
    std::string s = arg;
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
        s = s.substr(colon + 1);
    }
    try {
        unsigned long long v = std::stoull(s);
        if (v == 0 || v > 0xFFFFFFFFull) {
            return false;
        }
        *sigma_out = (uint32_t)v;
        return true;
    } catch (...) {
        return false;
    }
}

static constexpr unsigned long CHKSUMMOD = 4294967291UL;

static bool check_save_file_writable(const std::string &savefilename, bool saveappend) {
    if (!saveappend && access(savefilename.c_str(), 0) == 0) {
        std::cerr << "Save file " << savefilename << " already exists, will not overwrite" << std::endl;
        return false;
    }
    FILE *savefile = fopen(savefilename.c_str(), "a");
    if (savefile == nullptr) {
        std::cerr << "Could not open file " << savefilename << " for writing" << std::endl;
        return false;
    }
    fclose(savefile);
    if (!saveappend) {
        struct stat st {};
        if (stat(savefilename.c_str(), &st) != 0 || st.st_size != 0) {
            std::cerr << "Save file " << savefilename << " initialization failed" << std::endl;
            return false;
        }
        if (remove(savefilename.c_str()) != 0) {
            std::cerr << "Save file " << savefilename << " could not be cleaned up" << std::endl;
            return false;
        }
    }
    return true;
}

static std::string mpz_to_dec_string(const mpz_t v) {
    char *s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
}

static std::string build_saved_n_expr(const std::string &original_expr, const mpz_t N,
                                      uint32_t curves, mpz_t *factors, int *array_found) {
    std::string expr = original_expr.empty() ? mpz_to_dec_string(N) : original_expr;
    mpz_t remaining;
    mpz_init_set(remaining, N);

    std::vector<std::string> uniq_factors_dec;
    uniq_factors_dec.reserve(curves);
    for (uint32_t i = 0; i < curves; ++i) {
        if (array_found[i] == ECM_NO_FACTOR_FOUND) {
            continue;
        }
        if (mpz_cmp_ui(factors[i], 1) <= 0 || mpz_cmp(factors[i], N) >= 0) {
            continue;
        }
        std::string dec = mpz_to_dec_string(factors[i]);
        if (std::find(uniq_factors_dec.begin(), uniq_factors_dec.end(), dec) == uniq_factors_dec.end()) {
            uniq_factors_dec.push_back(dec);
        }
    }

    std::sort(uniq_factors_dec.begin(), uniq_factors_dec.end(),
              [](const std::string &a, const std::string &b) {
                  if (a.size() != b.size()) {
                      return a.size() < b.size();
                  }
                  return a < b;
              });

    mpz_t f;
    mpz_init(f);
    for (const std::string &dec : uniq_factors_dec) {
        if (mpz_set_str(f, dec.c_str(), 10) != 0) {
            continue;
        }
        if (!mpz_divisible_p(remaining, f)) {
            continue;
        }
        expr = "(" + expr + ")/" + dec;
        mpz_divexact(remaining, remaining, f);
    }
    mpz_clear(f);
    mpz_clear(remaining);
    return expr;
}

static std::string build_who_field() {
    const char *uname = std::getenv("LOGNAME");
    if (!uname || !*uname) {
        uname = std::getenv("USERNAME");
    }
    std::string user = (uname && *uname) ? uname : "";

    std::string host;
#ifdef _WIN32
    char hbuf[MAX_COMPUTERNAME_LENGTH + 2] = {0};
    DWORD sz = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(hbuf, &sz) && sz > 0) {
        host.assign(hbuf, hbuf + sz);
    }
#else
    char hbuf[64] = {0};
    if (gethostname(hbuf, sizeof(hbuf) - 1) == 0) {
        host = hbuf;
    }
#endif

    if (user.empty() && host.empty()) {
        return "";
    }
    return user + "@" + host;
}

static bool append_opencl_save_lines(const std::string &savefilename, const mpz_t N, double B1,
                                     uint32_t firstsigma, uint32_t curves, mpz_t *factors,
                                     const std::string &n_expr_save) {
    std::ofstream out(savefilename, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Could not open file " << savefilename << " for appending" << std::endl;
        return false;
    }

    mpz_t sigma_mpz, checksum;
    mpz_init(sigma_mpz);
    mpz_init(checksum);

    const time_t t = std::time(nullptr);
    char timebuf[128] = {0};
    const tm *lt = std::localtime(&t);
    if (lt) {
        std::strftime(timebuf, sizeof(timebuf), "%a %b %d %H:%M:%S %Y", lt);
    }
    const std::string who = build_who_field();

    for (uint32_t i = 0; i < curves; ++i) {
        mpz_set_ui(sigma_mpz, firstsigma + i);
        mpz_set_d(checksum, B1);
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(sigma_mpz, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(N, CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, mpz_fdiv_ui(factors[i], CHKSUMMOD));
        mpz_mul_ui(checksum, checksum, (ECM_PARAM_BATCH_32BITS_D + 1) % CHKSUMMOD);
        const unsigned long csum = mpz_fdiv_ui(checksum, CHKSUMMOD);

        char *sigma_dec = mpz_get_str(nullptr, 10, sigma_mpz);
        char *x_hex = mpz_get_str(nullptr, 16, factors[i]);

        out << "METHOD=ECM; PARAM=" << ECM_PARAM_BATCH_32BITS_D
            << "; SIGMA=" << sigma_dec
            << "; B1=" << std::llround(B1)
            << "; N=" << n_expr_save
            << "; X=0x" << x_hex
            << "; CHECKSUM=" << csum
            << "; PROGRAM=GMP-ECM 7.0.6;"
            << " X0=0x0; Y0=0x0;"
            << (who.empty() ? "" : (" WHO=" + who + ";"))
            << " TIME=" << timebuf << ";"
            << "\n";

        free(sigma_dec);
        free(x_hex);
    }

    mpz_clear(sigma_mpz);
    mpz_clear(checksum);
    return true;
}

int main(int argc, char **argv){
    bool verbose = false;
    bool use_gpu = false;
    uint32_t gpucurves = 0;
    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;
    std::string savefilename;
    bool saveappend = false;
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "-v") { verbose = true; continue; }
        if(a == "-gpu") { use_gpu = true; continue; }
        if(a == "-gpucurves" && i+1<argc){ gpucurves = (uint32_t)std::stoul(argv[++i]); continue; }
        if(a == "-gpuckpt" && i+1<argc){ gpuckpt_ms = (unsigned long) std::stoul(argv[++i]); continue; }
        if(a == "-sigma" && i+1<argc){
            if(!parse_sigma_arg(argv[++i], &fixed_sigma)){
                std::cerr << "Invalid -sigma value (need 1..2^32-1, optional param:3: prefix)" << std::endl;
                return 1;
            }
            sigma_fixed = true;
            continue;
        }
        if(a == "-save" && i+1<argc) {
            savefilename = argv[++i];
            saveappend = false;
            continue;
        }
        if(a == "-savea" && i+1<argc) {
            savefilename = argv[++i];
            saveappend = true;
            continue;
        }
        pos.push_back(a);
    }

    std::cout << "ecm driver starting" << std::endl;
    std::cout << "  mode: " << (use_gpu ? "gpu" : "cpu-stub")
              << ", gpucurves=" << gpucurves
              << ", gpuckpt_ms=" << gpuckpt_ms << std::endl;
    if(!pos.empty()){
        std::cout << "  B1=" << pos[0];
        if(pos.size() >= 2){
            std::cout << ", B2=" << pos[1];
        }
        std::cout << std::endl;
    }

#ifdef _WIN32
    if(_isatty(_fileno(stdin))) {
        std::cout << "This driver expects N on stdin. Example:" << std::endl;
        std::cout << "  echo '(2^991-1)' | .\\build\\Debug\\ecm.exe -v -gpu -gpucurves 384 1e6 0" << std::endl;
        return 1;
    }
#endif

    // read N from stdin
    std::string nline;
    {
        std::ostringstream oss;
        std::string line;
        while(std::getline(std::cin, line)){
            oss << line;
        }
        nline = oss.str();
        trim(nline);
    }
    if(nline.empty()){
        std::cerr << "No input number on stdin" << std::endl;
        return 1;
    }

    // positional B1 and B2
    double B1 = 0.0; double B2 = 0.0;
    if(pos.size() >= 1) {
        B1 = strtod(pos[0].c_str(), nullptr);
    }
    if(pos.size() >= 2) {
        B2 = strtod(pos[1].c_str(), nullptr);
    }

    mpz_t N; mpz_init(N);
    ExprParser parser(nline);
    if(!parser.parse(N)){
        std::cerr << "Failed to parse N: '"<< nline <<"'" << std::endl;
        std::cerr << "Parse error: " << parser.message() << std::endl;
        return 1;
    }

    std::cout << "Parsed N bit-size: " << mpz_sizeinbase(N, 2) << std::endl;
    if (verbose) {
        std::cout << "Parsed N = ";
        mpz_out_str(stdout, 10, N);
        std::cout << std::endl;
    }

    // set up ecm params
    ecm_params params;
    ecm_init(params);
    params->gpu = use_gpu ? 1 : 0;
    params->gpu_number_of_curves = gpucurves;
    params->gpu_checkpoint_interval_ms = gpuckpt_ms;
    params->verbose = verbose ? 1 : 0;
    params->param = ECM_PARAM_BATCH_32BITS_D; // GPU expects batch 32bits d

    // compute batch_s from B1
    mpz_t batch_s; mpz_init(batch_s);
    if(!compute_batch_s(batch_s, B1)){
        std::cerr << "Failed to compute batch_s"<<std::endl;
        return 1;
    }
    std::cout << "batch_s bit-size: " << mpz_sizeinbase(batch_s, 2) << std::endl;
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;

    // allocate factors
    uint32_t curves = gpucurves;
    if(curves == 0){
        std::cerr << "gpucurves must be > 0"<<std::endl;
        return 1;
    }

    if (use_gpu) {
        int prep = gpu_prepare_opencl((size_t)mpz_sizeinbase(N, 2), params->verbose);
        if (prep != 0) {
            std::cerr << "GPU: OpenCL prepare failed" << std::endl;
            mpz_clear(N);
            mpz_clear(batch_s);
            ecm_clear(params);
            return 1;
        }
    }

    if (!savefilename.empty()) {
        if (!check_save_file_writable(savefilename, saveappend)) {
            mpz_clear(N);
            mpz_clear(batch_s);
            ecm_clear(params);
            return 1;
        }
    }

    mpz_t *factors = (mpz_t*) malloc(sizeof(mpz_t)*curves);
    int *array_found = (int*) malloc(sizeof(int)*curves);
    for(uint32_t i=0;i<curves;i++){ mpz_init(factors[i]); array_found[i]=ECM_NO_FACTOR_FOUND; }

    uint32_t firstsigma = sigma_fixed ? fixed_sigma : gpu_pick_random_sigma(curves);
    if ((uint64_t)firstsigma + curves > 0x100000000ull) {
        std::cerr << "sigma range overflows uint32 (sigma + curves > 2^32)" << std::endl;
        return 1;
    }
    uint32_t lastsigma = firstsigma + curves - 1;

    mpz_t batch_d;
    mpz_init(batch_d);
    gpu_compute_batch_d(batch_d, firstsigma, N);

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << ", sigma=" << ECM_PARAM_BATCH_32BITS_D << ":" << firstsigma
              << "-" << lastsigma << " (" << curves << " curves)" << std::endl;

    {
        unsigned long k_blocks = params->k;
        std::cout << "dF=0, k=" << k_blocks << ", d=";
        mpz_out_str(stdout, 10, batch_d);
        std::cout << ", d2=0, i0=0" << std::endl;
    }

    float gputime = 0.0f;

    int ret = opencl_ecm_stage1(factors, array_found, N, params->batch_s, curves, &firstsigma,
                                params->gpu_checkpoint_interval_ms, &gputime, params->verbose);

    std::cout << "opencl_ecm_stage1 returned: "<< ret <<" gputime="<< gputime <<" ms\n";
    for(uint32_t i=0;i<curves;i++){
        if(array_found[i] != ECM_NO_FACTOR_FOUND){
            char *s = mpz_get_str(NULL,10,factors[i]);
            std::cout << "factor["<<i<<"]="<< s <<"\n";
            free(s);
        }
    }
    std::string n_expr_save = build_saved_n_expr(nline, N, curves, factors, array_found);

    if (ret != ECM_ERROR && !savefilename.empty()) {
        if (!append_opencl_save_lines(savefilename, N, B1, firstsigma, curves, factors, n_expr_save)) {
            std::cerr << "Failed to append OpenCL save lines into " << savefilename << std::endl;
            return 1;
        }
    }
    for(uint32_t i=0;i<curves;i++) mpz_clear(factors[i]);
    free(factors); free(array_found);
    mpz_clear(batch_d);
    mpz_clear(N); mpz_clear(batch_s);
    ecm_clear(params);
    return 0;
}
