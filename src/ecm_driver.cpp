#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#endif

#include <gmp.h>

#include "opencl_ecm_entry.h"
#include "ecm.h"

static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}

// Minimal parser to handle inputs like (2^991-1) or decimal/hex strings
static bool parse_N_string(const std::string &in, mpz_t out){
    std::string s = in;
    trim(s);
    if(s.empty()) return false;
    // remove surrounding parentheses
    if(s.size() >= 2 && s.front() == '(' && s.back() == ')'){
        s = s.substr(1, s.size()-2);
        trim(s);
    }
    // Look for pattern base ^ exp [+- offset]
    size_t pos_pow = s.find('^');
    if(pos_pow != std::string::npos){
        // get base left of ^ (possibly number)
        std::string base_s = s.substr(0, pos_pow);
        trim(base_s);
        // find + or - after exponent
        size_t pos_sign = s.find_first_of("+-", pos_pow+1);
        std::string exp_s, tail;
        if(pos_sign != std::string::npos){
            exp_s = s.substr(pos_pow+1, pos_sign - (pos_pow+1));
            tail = s.substr(pos_sign);
        } else {
            exp_s = s.substr(pos_pow+1);
        }
        trim(exp_s); trim(tail);
        unsigned long base = 0;
        if(base_s == "2") base = 2;
        else {
            // try parse base decimal
            base = std::stoul(base_s);
        }
        unsigned long exp = std::stoul(exp_s);
        mpz_t tmp;
        mpz_init(tmp);
        mpz_ui_pow_ui(tmp, base, exp);
        if(!tail.empty()){
            char sign = tail[0];
            mpz_t offs; mpz_init(offs);
            std::string offs_s = tail.substr(1);
            trim(offs_s);
            mpz_set_str(offs, offs_s.c_str(), 10);
            if(sign == '+') mpz_add(tmp, tmp, offs);
            else mpz_sub(tmp, tmp, offs);
            mpz_clear(offs);
        }
        mpz_set(out, tmp);
        mpz_clear(tmp);
        return true;
    }
    // else fallback to mpz_set_str base 0 (supports 0x, decimals)
    if(mpz_set_str(out, s.c_str(), 0) == 0) return true;
    return false;
}

// Compute batch product s = prod_{p<=B1} p^{floor(log_p(B1))}
static bool compute_batch_s(mpz_t s, double B1){
    if(B1 < 2.0) { mpz_set_ui(s, 1); return true; }
    uint32_t limit = (uint32_t) std::floor(B1 + 0.0001);
    std::vector<char> sieve(limit+1, 1);
    sieve[0]=sieve[1]=0;
    for(uint32_t p=2;p*(uint64_t)p<=limit;++p){
        if(sieve[p]){
            for(uint64_t q=(uint64_t)p*p;q<=limit;q+=p) sieve[(size_t)q]=0;
        }
    }
    mpz_set_ui(s,1);
    mpz_t term; mpz_init(term);
    for(uint32_t p=2;p<=limit;++p){
        if(!sieve[p]) continue;
        // compute p^e <= limit
        uint64_t pp = p;
        while(pp <= limit){
            mpz_mul_ui(s, s, p);
            if(pp > (uint64_t)limit / p) break;
            pp *= p;
        }
    }
    mpz_clear(term);
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
    if(!parse_N_string(nline, N)){
        std::cerr << "Failed to parse N: '"<< nline <<"'"<<std::endl;
        return 1;
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
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;

    // allocate factors
    uint32_t curves = gpucurves;
    if(curves == 0){
        std::cerr << "gpucurves must be > 0"<<std::endl;
        return 1;
    }
    mpz_t *factors = (mpz_t*) malloc(sizeof(mpz_t)*curves);
    int *array_found = (int*) malloc(sizeof(int)*curves);
    for(uint32_t i=0;i<curves;i++){ mpz_init(factors[i]); array_found[i]=ECM_NO_FACTOR_FOUND; }

    uint32_t firstsigma = 2u; // initial sigma seed
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
    mpz_clear(N); mpz_clear(batch_s);
    ecm_clear(params);
    return 0;
}
