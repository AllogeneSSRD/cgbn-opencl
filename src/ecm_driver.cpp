#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>

#ifdef _WIN32
#include <io.h>
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

int main(int argc, char **argv){
    bool verbose = false;
    bool use_gpu = false;
    uint32_t gpucurves = 0;
    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "-v") { verbose = true; continue; }
        if(a == "-gpu") { use_gpu = true; continue; }
        if(a == "-gpucurves" && i+1<argc){ gpucurves = (uint32_t)std::stoul(argv[++i]); continue; }
        if(a == "-gpuckpt" && i+1<argc){ gpuckpt_ms = (unsigned long) std::stoul(argv[++i]); continue; }
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

    mpz_t *factors = (mpz_t*) malloc(sizeof(mpz_t)*curves);
    int *array_found = (int*) malloc(sizeof(int)*curves);
    for(uint32_t i=0;i<curves;i++){ mpz_init(factors[i]); array_found[i]=ECM_NO_FACTOR_FOUND; }

    uint32_t firstsigma = gpu_pick_random_sigma(curves);
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

    int ret = opencl_ecm_stage1(factors, array_found, N, params->batch_s, curves, &firstsigma, params->gpu_checkpoint_interval_ms, &gputime, params->verbose);

    std::cout << "opencl_ecm_stage1 returned: "<< ret <<" gputime="<< gputime <<" ms\n";
    for(uint32_t i=0;i<curves;i++){
        if(array_found[i] != ECM_NO_FACTOR_FOUND){
            char *s = mpz_get_str(NULL,10,factors[i]);
            std::cout << "factor["<<i<<"]="<< s <<"\n";
            free(s);
        }
    }

    for(uint32_t i=0;i<curves;i++) mpz_clear(factors[i]);
    free(factors); free(array_found);
    mpz_clear(batch_d);
    mpz_clear(N); mpz_clear(batch_s);
    ecm_clear(params);
    return 0;
}
