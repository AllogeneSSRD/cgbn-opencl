#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <gmp.h>

#include "opencl_ecm_entry.h"
#include "ecm.h"
#include "cgbn_stage1.h"
#include "opencl_ecm_log.h"
#include "cl_probe.h"

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

struct PrimePowerBound {
    uint32_t p;
    uint32_t exp;
};

static bool build_primes_up_to_B1(double B1, std::vector<uint32_t> &primes) {
    primes.clear();
    if (B1 < 2.0) {
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
        if (!sieve[p]) continue;
        for (uint64_t q = (uint64_t)p * (uint64_t)p; q <= limit; q += p) {
            sieve[(size_t)q] = 0;
        }
    }

    for (uint32_t p = 2; p <= limit; ++p) {
        if (!sieve[p]) continue;
        primes.push_back(p);
    }
    return true;
}

static std::vector<PrimePowerBound> factor_by_small_primes(
    const mpz_t n, const std::vector<uint32_t> &primes) {
    std::vector<PrimePowerBound> out;
    mpz_t work;
    mpz_init_set(work, n);
    for (uint32_t p : primes) {
        uint32_t exp = 0;
        while (mpz_divisible_ui_p(work, p) != 0) {
            mpz_fdiv_q_ui(work, work, p);
            ++exp;
        }
        if (exp > 0) {
            out.push_back({p, exp});
        }
    }
    mpz_clear(work);
    return out;
}

static std::string format_group_order_smooth(const std::vector<PrimePowerBound> &parts) {
    std::ostringstream oss;
    oss << "[ ";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << "<" << parts[i].p << ", " << parts[i].exp << ">";
    }
    oss << " ]";
    return oss.str();
}

static std::string get_gp_executable() {
    auto normalize_cmd_path = [](std::string s) -> std::string {
        trim(s);
        // Remove any outer single/double quotes repeatedly.
        while (s.size() >= 2 &&
               ((s.front() == '"' && s.back() == '"') ||
                (s.front() == '\'' && s.back() == '\''))) {
            s = s.substr(1, s.size() - 2);
            trim(s);
        }
        // Also strip stray quote characters at either side.
        while (!s.empty() && (s.front() == '"' || s.front() == '\'')) {
            s.erase(s.begin());
        }
        while (!s.empty() && (s.back() == '"' || s.back() == '\'')) {
            s.pop_back();
        }
        // gp path should not contain quote characters; strip any residual ones.
        s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
        s.erase(std::remove(s.begin(), s.end(), '\''), s.end());
        return s;
    };

    const char *ecm_gp = std::getenv("ECM_GP_BIN");
    if (ecm_gp && *ecm_gp) {
        return normalize_cmd_path(std::string(ecm_gp));
    }
    const char *pari_gp = std::getenv("PARI_GP_BIN");
    if (pari_gp && *pari_gp) {
        return normalize_cmd_path(std::string(pari_gp));
    }
    return "gp";
}

static bool compute_group_order_pari_for_sigma3(mpz_t order_out, const mpz_t p,
                                                uint32_t sigma, std::string *err) {
    char tmp_file[L_tmpnam];
    if (std::tmpnam(tmp_file) == nullptr) {
        if (err) *err = "failed to create temporary script path";
        return false;
    }
    // tmpnam on Windows can return paths that are awkward for cmd parsing.
    // Put the temporary script in the current workspace with a simple filename.
    std::string base(tmp_file);
    for (char &c : base) {
        if (c == '\\' || c == '/' || c == ':' || c == '.' || c == ' ') {
            c = '_';
        }
    }
    const std::string script_path = "ecm_go_tmp_" + base + ".gp";
    std::ofstream gpfile(script_path, std::ios::out | std::ios::trunc);
    if (!gpfile.is_open()) {
        if (err) *err = "failed to open temporary gp script";
        return false;
    }

    const std::string p_dec = mpz_to_dec_string(p);
    gpfile << "p = " << p_dec << ";\n";
    gpfile << "s = " << sigma << ";\n";
    gpfile << "A = Mod(4*s, p) / Mod(2^32, p) - 2;\n";
    gpfile << "b = 4*A + 10;\n";
    gpfile << "E = ellinit([0, b*A, 0, b^2, 0]);\n";
    gpfile << "print(lift(ellcard(E)));\n";
    gpfile << "quit();\n";
    gpfile.close();

    std::string output;
    const std::string gp_exe = get_gp_executable();
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0)) {
        remove(script_path.c_str());
        if (err) *err = "CreatePipe failed for gp output";
        return false;
    }
    if (!SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(child_stdout_read);
        CloseHandle(child_stdout_write);
        remove(script_path.c_str());
        if (err) *err = "SetHandleInformation failed for gp output";
        return false;
    }

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = child_stdout_write;
    si.hStdError = child_stdout_write;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::string cmdline = "\"" + gp_exe + "\" -q -f \"" + script_path + "\"";
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    BOOL ok = CreateProcessA(
        gp_exe.c_str(),
        cmdline_buf.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi);
    CloseHandle(child_stdout_write);
    if (!ok) {
        DWORD code = GetLastError();
        CloseHandle(child_stdout_read);
        remove(script_path.c_str());
        if (err) *err = "CreateProcess(gp) failed, code=" + std::to_string((unsigned long)code) +
                        ", exe=" + gp_exe;
        return false;
    }

    char buffer[256];
    DWORD nread = 0;
    while (ReadFile(child_stdout_read, buffer, sizeof(buffer) - 1, &nread, nullptr) && nread > 0) {
        buffer[nread] = '\0';
        output += buffer;
    }
    CloseHandle(child_stdout_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 0;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    const std::string cmd = "\"" + gp_exe + "\" -q -f \"" + script_path + "\"";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        remove(script_path.c_str());
        if (err) *err = "failed to launch gp executable: " + gp_exe;
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int rc = pclose(pipe);
#endif
    remove(script_path.c_str());
    if ((long)rc != 0) {
        if (err) *err = "gp execution failed: " + gp_exe;
        return false;
    }

    std::istringstream iss(output);
    std::string line;
    std::string last_int;
    while (std::getline(iss, line)) {
        trim(line);
        if (line.empty()) continue;
        bool ok = true;
        size_t start = (line[0] == '-') ? 1 : 0;
        if (start == line.size()) ok = false;
        for (size_t i = start; ok && i < line.size(); ++i) {
            if (!std::isdigit((unsigned char)line[i])) ok = false;
        }
        if (ok) last_int = line;
    }
    if (last_int.empty()) {
        if (err) *err = "gp returned no integer ellcard output";
        return false;
    }
    if (mpz_set_str(order_out, last_int.c_str(), 10) != 0) {
        if (err) *err = "failed to parse gp ellcard integer";
        return false;
    }
    return true;
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
    ecm_install_timestamped_iostreams();
    bool verbose = false;
    bool use_gpu = false;
    uint32_t gpucurves = 0;
    double gpuckpt_seconds = -1.0;
    bool gpuckpt_set = false;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;
    int gpu_device_index = 0;
    bool print_group_order = false;
    std::string savefilename;
    bool saveappend = false;
    std::string gpu_mul_path;
    std::string gpu_sqr_path;
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "-v") { verbose = true; continue; }
        if(a == "-gpu") { use_gpu = true; continue; }
        if(a == "-gpucurves" && i+1<argc){ gpucurves = (uint32_t)std::stoul(argv[++i]); continue; }
        if(a == "-gpuckpt" && i+1<argc){
            try {
                gpuckpt_seconds = std::stod(argv[++i]);
                gpuckpt_set = true;
            } catch (...) {
                std::cerr << "Invalid -gpuckpt value, expected number of seconds" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "-d" && i+1<argc){
            try {
                gpu_device_index = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid -d value, expected integer device index" << std::endl;
                return 1;
            }
            if (gpu_device_index < 0) {
                std::cerr << "Invalid -d value, expected >= 0" << std::endl;
                return 1;
            }
            continue;
        }
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
        if(a == "--go") {
            print_group_order = true;
            continue;
        }
        if(a == "--mul" && i+1<argc) {
            gpu_mul_path = argv[++i];
            continue;
        }
        if(a == "--sqr" && i+1<argc) {
            gpu_sqr_path = argv[++i];
            continue;
        }
        pos.push_back(a);
    }

    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    if (gpuckpt_set) {
        const double val_ms = gpuckpt_seconds * 1000.0;
        if (val_ms != val_ms) {
            std::cerr << "Error, invalid -gpuckpt value (NaN)" << std::endl;
            return 1;
        }
        if (val_ms <= 0.0) {
            gpuckpt_ms = 0;
        } else if (val_ms >= (double)ULONG_MAX) {
            gpuckpt_ms = ULONG_MAX;
        } else {
            gpuckpt_ms = (unsigned long)std::llround(val_ms);
        }
    }

    std::cout << "ecm driver starting" << std::endl;
    std::cout << "  mode: " << (use_gpu ? "gpu" : "cpu-stub")
              << ", gpucurves=" << gpucurves
              << ", gpuckpt=" << (gpuckpt_ms == 0 ? 0.0 : gpuckpt_ms / 1000.0) << "s"
              << ", device=" << gpu_device_index
              << ", group_order=" << (print_group_order ? "on" : "off");
    if (!gpu_mul_path.empty()) {
        std::cout << ", mul=" << gpu_mul_path;
    }
    if (!gpu_sqr_path.empty()) {
        std::cout << ", sqr=" << gpu_sqr_path;
    }
    std::cout << std::endl;
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
        std::cout << "  echo '(2^991-1)' | .\\build\\Debug\\ecm.exe -v --go -gpu -gpucurves 384 1e6 0" << std::endl;
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
    // if (verbose) {
    //     std::cout << "Parsed N = ";
    //     mpz_out_str(stdout, 10, N);
    //     std::cout << std::endl;
    // }

    // set up ecm params
    ecm_params params;
    ecm_init(params);
    params->gpu = use_gpu ? 1 : 0;
    params->gpu_number_of_curves = gpucurves;
    params->gpu_checkpoint_interval_ms = gpuckpt_ms;
    if (!gpu_mul_path.empty()) {
        strncpy(params->gpu_mul_path, gpu_mul_path.c_str(), sizeof(params->gpu_mul_path) - 1u);
        params->gpu_mul_path[sizeof(params->gpu_mul_path) - 1u] = '\0';
    }
    if (!gpu_sqr_path.empty()) {
        strncpy(params->gpu_sqr_path, gpu_sqr_path.c_str(), sizeof(params->gpu_sqr_path) - 1u);
        params->gpu_sqr_path[sizeof(params->gpu_sqr_path) - 1u] = '\0';
    }
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
        if (!configureOpenclDeviceIndex(gpu_device_index, true)) {
            mpz_clear(N);
            mpz_clear(batch_s);
            ecm_clear(params);
            return 1;
        }
        int prep = gpu_prepare_opencl((size_t)mpz_sizeinbase(N, 2), params->verbose,
                                      params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                      params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr);
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

    std::vector<uint32_t> go_primes;
    if (print_group_order) {
        if (!build_primes_up_to_B1(B1, go_primes)) {
            std::cerr << "Failed to build primes for --go (invalid B1 range)" << std::endl;
            return 1;
        }
    }

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << ", sigma=" << ECM_PARAM_BATCH_32BITS_D << ":" << firstsigma
              << "-" << lastsigma << " (" << curves << " curves)" << std::endl;

    // {
    //     unsigned long k_blocks = params->k;
    //     std::cout << "dF=0, k=" << k_blocks << ", d=";
    //     mpz_out_str(stdout, 10, batch_d);
    //     std::cout << ", d2=0, i0=0" << std::endl;
    // }

    float gputime = 0.0f;

    int ret = opencl_ecm_stage1(factors, array_found, N, params->batch_s, curves, &firstsigma,
                                params->gpu_checkpoint_interval_ms, &gputime, params->verbose,
                                params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr);

    std::cout << "opencl_ecm_stage1 returned: "<< ret <<" gputime="<< gputime <<" ms\n";
    for(uint32_t i=0;i<curves;i++){
        if(array_found[i] != ECM_NO_FACTOR_FOUND){
            char *s = mpz_get_str(NULL,10,factors[i]);
            std::cout << "factor["<<i<<"]="<< s <<"\n";
            free(s);
            if (print_group_order) {
                uint32_t sigma_curve = firstsigma + i;
                if (mpz_probab_prime_p(factors[i], 25) <= 0) {
                    std::cout << "  go_factor[" << i << "]=[ ] (factor is not prime, skip #E(F_p))\n";
                    continue;
                }
                mpz_t go;
                mpz_init(go);
                std::string err;
                if (!compute_group_order_pari_for_sigma3(go, factors[i], sigma_curve, &err)) {
                    std::cout << "  go_factor[" << i << "]=[ ] (gp error: " << err << ")\n";
                    mpz_clear(go);
                    continue;
                }
                auto go_parts = factor_by_small_primes(go, go_primes);
                std::cout << "  go[" << i << "]=" << mpz_to_dec_string(go) << "\n";
                std::cout << "  go_factor[" << i << "]="
                          << format_group_order_smooth(go_parts) << "\n";
                mpz_clear(go);
            }
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
