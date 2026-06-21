#pragma once
// ============================================================================
// cpu_info.h — portable CPU topology & ISA detection.
//
// Provides:
//   CpuIsa  – bit-set of detected instruction-set groups
//   get_cpu_name()  – brand string (model name)
//   get_cpu_cores() – logical / physical core count
//   detect_cpu_isa()– fill CpuIsa from CPUID
//   print_cpu_info()– pretty-print all info (guard by verbose bool)
//
// Cross-platform (Windows MSVC / Linux GCC+Clang).
// ============================================================================

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

// ── ISA flag bits ─────────────────────────────────────────────────────

enum : uint64_t {
    ISA_SSE2       = 1ull << 0,
    ISA_AVX        = 1ull << 1,
    ISA_FMA        = 1ull << 2,
    ISA_AES        = 1ull << 3,
    ISA_AVX2       = 1ull << 4,
    ISA_BMI2       = 1ull << 5,
    ISA_ADX        = 1ull << 6,
    ISA_AVX512F    = 1ull << 7,
    ISA_AVX512DQ   = 1ull << 8,
    ISA_AVX512BW   = 1ull << 9,
    ISA_AVX512VL   = 1ull << 10,
    ISA_SHA        = 1ull << 11,
};

struct CpuIsa {
    uint64_t flags = 0;
    bool has(uint64_t f) const { return (flags & f) != 0; }
};

// ── Detection routines ────────────────────────────────────────────────

inline std::string get_cpu_name() {
    char name[49] = {0};
#ifdef _WIN32
    int data[4];
    __cpuid(data, 0x80000000);
    if ((unsigned)data[0] >= 0x80000004) {
        for (unsigned i = 0; i < 3; ++i) {
            __cpuid(data, (int)(0x80000002 + i));
            std::memcpy(name + i * 16, data, 16);
        }
    }
#else
    unsigned regs[4];
    if (__get_cpuid(0x80000000, &regs[0], &regs[1], &regs[2], &regs[3]) && regs[0] >= 0x80000004) {
        for (unsigned i = 0; i < 3; ++i) {
            __get_cpuid(0x80000002 + i, &regs[0], &regs[1], &regs[2], &regs[3]);
            std::memcpy(name + i * 16, regs, 16);
        }
    }
#endif
    while (name[0] == ' ') std::memmove(name, name + 1, 48);
    return name[0] ? std::string(name) : "(unknown)";
}

inline void get_cpu_cores(int &physical, int &logical) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si);
    logical = (int)si.dwNumberOfProcessors;
    physical = logical;
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len > 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
            len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buf.data(), &len)) {
            physical = 0;
            for (auto &i : buf)
                if (i.Relationship == RelationProcessorCore) physical++;
        }
    }
#else
    logical  = (int)sysconf(_SC_NPROCESSORS_CONF);
    physical = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (physical <= 0) physical = logical;
#endif
}

inline CpuIsa detect_cpu_isa() {
    CpuIsa isa;
#ifdef _WIN32
    int data[4];
    __cpuid(data, 1);
    if ((data[3] >> 26) & 1) isa.flags |= ISA_SSE2;
    if ((data[2] >> 28) & 1) isa.flags |= ISA_AVX;
    if ((data[2] >> 12) & 1) isa.flags |= ISA_FMA;
    if ((data[2] >> 25) & 1) isa.flags |= ISA_AES;
    __cpuid(data, 7);
    if ((data[1] >>  5) & 1) isa.flags |= ISA_AVX2;
    if ((data[1] >>  8) & 1) isa.flags |= ISA_BMI2;
    if ((data[1] >> 19) & 1) isa.flags |= ISA_ADX;
    if ((data[1] >> 16) & 1) isa.flags |= ISA_AVX512F;
    if ((data[1] >> 17) & 1) isa.flags |= ISA_AVX512DQ;
    if ((data[1] >> 30) & 1) isa.flags |= ISA_AVX512BW;
    if ((data[1] >> 31) & 1) isa.flags |= ISA_AVX512VL;
    if ((data[1] >> 29) & 1) isa.flags |= ISA_SHA;
#else
    unsigned regs[4];
    __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
    if ((regs[3] >> 26) & 1) isa.flags |= ISA_SSE2;
    if ((regs[2] >> 28) & 1) isa.flags |= ISA_AVX;
    if ((regs[2] >> 12) & 1) isa.flags |= ISA_FMA;
    if ((regs[2] >> 25) & 1) isa.flags |= ISA_AES;
    if (__get_cpuid_max(7, nullptr) >= 7) {
        __get_cpuid(7, &regs[0], &regs[1], &regs[2], &regs[3]);
        if ((regs[1] >>  5) & 1) isa.flags |= ISA_AVX2;
        if ((regs[1] >>  8) & 1) isa.flags |= ISA_BMI2;
        if ((regs[1] >> 19) & 1) isa.flags |= ISA_ADX;
        if ((regs[1] >> 16) & 1) isa.flags |= ISA_AVX512F;
        if ((regs[1] >> 17) & 1) isa.flags |= ISA_AVX512DQ;
        if ((regs[1] >> 30) & 1) isa.flags |= ISA_AVX512BW;
        if ((regs[1] >> 31) & 1) isa.flags |= ISA_AVX512VL;
        if ((regs[1] >> 29) & 1) isa.flags |= ISA_SHA;
    }
#endif
    return isa;
}

// ── Pretty-printer ────────────────────────────────────────────────────

/// Print CPU model, core count, ISA groups and optional bench-specific
/// ISA requirement lines.  \p verbose guards the entire function.
inline void print_cpu_info(bool verbose, const std::string &extra = {}) {
    if (!verbose) return;

    CpuIsa isa = detect_cpu_isa();
    std::string cpu_name = get_cpu_name();
    int physical = 0, logical = 0;
    get_cpu_cores(physical, logical);

    std::cout << "\n=== CPU Info ===\n"
              << "  Model:         " << cpu_name << "\n"
              << "  Cores:         " << physical << " physical / " << logical << " logical\n"
              << "  ISA groups:\n"
              << "    sse2:        " << (isa.has(ISA_SSE2) ? "SSE2 (always available)" : "(none)") << "\n"
              << "    avx:        " << (isa.has(ISA_AVX) ? "AVX" : "(none)")
              << (isa.has(ISA_AVX) && isa.has(ISA_FMA) ? " FMA" : "")
              << (isa.has(ISA_AVX) && isa.has(ISA_AES) ? " AES-NI" : "") << "\n"
              << "    avx2:        " << (isa.has(ISA_AVX2) ? "AVX2 FMA BMI2" : "(none)")
              << (isa.has(ISA_AVX2) && isa.has(ISA_ADX) ? " ADX" : "") << "\n"
              << "    avx512:      ";
    if (isa.has(ISA_AVX512F)) {
        std::cout << "AVX512F";
        if (isa.has(ISA_AVX512DQ))  std::cout << " AVX512DQ";
        if (isa.has(ISA_AVX512BW))  std::cout << " AVX512BW";
        if (isa.has(ISA_AVX512VL))  std::cout << " AVX512VL";
    } else { std::cout << "(none)"; }
    std::cout << "\n"
              << "    crypto:      " << (isa.has(ISA_AES) ? "AES-NI" : "");
    if (isa.has(ISA_AES) && isa.has(ISA_SHA)) std::cout << " SHA";
    if (!isa.has(ISA_AES) && !isa.has(ISA_SHA)) std::cout << "(none)";

    if (!extra.empty())
        std::cout << "\n\n" << extra;
    std::cout << "\n" << std::endl;
}
