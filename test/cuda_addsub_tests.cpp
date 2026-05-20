#include <cstdint>
#include <gmp.h>
#include <iostream>
#include <vector>
#include <iomanip>

// CPU version of CUDA add/sub benchmark
// Demonstrates the same operations as both CUDA and OpenCL implementations
int main() {
    constexpr int BITS = 1024;
    const int WORDS = BITS / 32;

    std::cout << "=== CPU Simulation of CUDA add/sub (matching OpenCL test) ===" << std::endl;
    std::cout << "Bits: " << BITS << ", Words: " << WORDS << std::endl << std::endl;

    // Prepare GMP numbers with same values as OpenCL test
    mpz_t a_gmp, b_gmp, res_add, res_sub;
    mpz_init(a_gmp);
    mpz_init(b_gmp);
    mpz_init(res_add);
    mpz_init(res_sub);

    // a = 2^991, b = 8218291649
    mpz_ui_pow_ui(a_gmp, 2, 991);
    mpz_set_ui(b_gmp, 8218291649u);

    // Helper to print hex (big-endian, omit leading zeros)
    auto print_hex = [&](const char *name, const mpz_t val) {
        size_t len = 0;
        mpz_t tmp;
        mpz_init_set(tmp, val);
        
        // Handle negative numbers
        if (mpz_sgn(tmp) < 0) {
            mpz_t mod, adjusted;
            mpz_init(mod);
            mpz_init(adjusted);
            mpz_ui_pow_ui(mod, 2, BITS);
            mpz_add(adjusted, tmp, mod);
            mpz_swap(tmp, adjusted);
            mpz_clear(mod);
            mpz_clear(adjusted);
        }

        // Export to words
        std::vector<uint32_t> words(WORDS);
        mpz_export(words.data(), &len, -1, sizeof(uint32_t), 0, 0, tmp);
        
        // Print as hex (big-endian)
        std::cout << name << ": 0x";
        bool leading = true;
        for (int i = WORDS - 1; i >= 0; --i) {
            if (leading && words[i] == 0) continue;
            leading = false;
            std::cout << std::hex << std::setw(8) << std::setfill('0') << words[i];
        }
        if (leading) std::cout << "0";
        std::cout << std::dec << std::endl;
        
        mpz_clear(tmp);
    };

    // Print inputs
    std::cout << "--- Inputs ---" << std::endl;
    print_hex("a (2^991)", a_gmp);
    print_hex("b (8218291649)", b_gmp);
    std::cout << std::endl;

    // Compute ADD: r = a + b; r = r + a (double op to prevent optimization)
    std::cout << "--- ADD operation: r = a+b; r = r+a ---" << std::endl;
    mpz_add(res_add, a_gmp, b_gmp);
    std::cout << "After r = a + b:" << std::endl;
    print_hex("  r", res_add);
    
    mpz_add(res_add, res_add, a_gmp);
    std::cout << "After r = r + a:" << std::endl;
    print_hex("  r", res_add);
    std::cout << std::endl;

    // Compute SUB: r = a - b; r = r - a (double op to prevent optimization)
    std::cout << "--- SUB operation: r = a-b; r = r-a ---" << std::endl;
    mpz_sub(res_sub, a_gmp, b_gmp);
    std::cout << "After r = a - b:" << std::endl;
    print_hex("  r", res_sub);
    
    mpz_sub(res_sub, res_sub, a_gmp);
    std::cout << "After r = r - a:" << std::endl;
    print_hex("  r", res_sub);
    std::cout << std::endl;

    // Export final results as word arrays for comparison
    std::vector<uint32_t> add_words(WORDS), sub_words(WORDS);
    
    size_t count = 0;
    mpz_export(add_words.data(), &count, -1, sizeof(uint32_t), 0, 0, res_add);
    for (size_t i = count; i < (size_t)WORDS; ++i) add_words[i] = 0u;
    
    // Handle negative SUB result
    if (mpz_sgn(res_sub) >= 0) {
        count = 0;
        mpz_export(sub_words.data(), &count, -1, sizeof(uint32_t), 0, 0, res_sub);
        for (size_t i = count; i < (size_t)WORDS; ++i) sub_words[i] = 0u;
    } else {
        mpz_t mod, tmp;
        mpz_init(mod);
        mpz_init(tmp);
        mpz_ui_pow_ui(mod, 2, BITS);
        mpz_add(tmp, res_sub, mod);
        count = 0;
        mpz_export(sub_words.data(), &count, -1, sizeof(uint32_t), 0, 0, tmp);
        for (size_t i = count; i < (size_t)WORDS; ++i) sub_words[i] = 0u;
        mpz_clear(mod);
        mpz_clear(tmp);
    }

    // Print as hex dump (for verification)
    std::cout << "--- Final results as hex dump (limbs little-endian) ---" << std::endl;
    std::cout << "ADD result (first 8 limbs): ";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << add_words[i] << " ";
    }
    std::cout << "..." << std::dec << std::endl;

    std::cout << "SUB result (first 8 limbs): ";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << sub_words[i] << " ";
    }
    std::cout << "..." << std::dec << std::endl;

    // Cleanup
    mpz_clear(a_gmp);
    mpz_clear(b_gmp);
    mpz_clear(res_add);
    mpz_clear(res_sub);

    std::cout << "\nExpected behavior:" << std::endl;
    std::cout << "- Both CUDA and OpenCL should produce these same results" << std::endl;
    std::cout << "- Double operations (r=a+b; r=r+a) prevent compiler optimization" << std::endl;

    return 0;
}
