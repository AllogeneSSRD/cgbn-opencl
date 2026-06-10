#include "opencl_ecm_log.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string* g_capture = nullptr;

} // namespace

void android_ecm_log_begin(std::string* out) {
    g_capture = out;
}

void android_ecm_log_end() {
    g_capture = nullptr;
}

bool ecm_log_timestamp_enabled() {
    return false;
}

void ecm_install_timestamped_iostreams() {}

int ecm_ts_vfprintf(FILE* stream, const char* fmt, va_list ap) {
    char buf[8192];
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    if (n <= 0) {
        return n;
    }
    const int clipped = (n < static_cast<int>(sizeof(buf))) ? n : static_cast<int>(sizeof(buf) - 1);
    if (g_capture != nullptr) {
        g_capture->append(buf, static_cast<size_t>(clipped));
        return clipped;
    }
    return vfprintf(stream, fmt, ap);
}

int ecm_ts_fprintf(FILE* stream, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int n = ecm_ts_vfprintf(stream, fmt, ap);
    va_end(ap);
    return n;
}
