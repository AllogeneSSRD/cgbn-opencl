#include "opencl_ecm_log.h"
#include "opencl_ecm_runtime_config.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>

namespace {

std::mutex g_log_mutex;

std::string timestamp_prefix() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream oss;
    oss << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "] ";
    return oss.str();
}

class timestamped_streambuf : public std::streambuf {
public:
    explicit timestamped_streambuf(std::streambuf *target) : target_(target) {}

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) {
            return target_->sputc(ch);
        }
        std::lock_guard<std::mutex> lk(g_log_mutex);
        if (at_line_start_) {
            std::string p = timestamp_prefix();
            target_->sputn(p.data(), static_cast<std::streamsize>(p.size()));
            at_line_start_ = false;
        }
        target_->sputc(static_cast<char>(ch));
        if (ch == '\n') {
            at_line_start_ = true;
        }
        return ch;
    }

    int sync() override {
        return target_->pubsync();
    }

private:
    std::streambuf *target_;
    bool at_line_start_ = true;
};

timestamped_streambuf *g_cout_buf = nullptr;
timestamped_streambuf *g_cerr_buf = nullptr;
bool g_installed = false;

} // namespace

bool ecm_log_timestamp_enabled() {
    return ecm_runtime_config().log_timestamp;  // default ON; CLI --no-log-timestamp 关闭
}

void ecm_install_timestamped_iostreams() {
    if (!ecm_log_timestamp_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (g_installed) {
        return;
    }
    g_cout_buf = new timestamped_streambuf(std::cout.rdbuf());
    g_cerr_buf = new timestamped_streambuf(std::cerr.rdbuf());
    std::cout.rdbuf(g_cout_buf);
    std::cerr.rdbuf(g_cerr_buf);
    g_installed = true;
}

int ecm_ts_vfprintf(FILE *stream, const char *fmt, va_list ap) {
    if (!ecm_log_timestamp_enabled()) {
        int rc = std::vfprintf(stream, fmt, ap);
        std::fflush(stream);
        return rc;
    }
    std::lock_guard<std::mutex> lk(g_log_mutex);
    std::string p = timestamp_prefix();
    std::fputs(p.c_str(), stream);
    int rc = std::vfprintf(stream, fmt, ap);
    std::fflush(stream);
    return rc;
}

int ecm_ts_fprintf(FILE *stream, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int rc = ecm_ts_vfprintf(stream, fmt, ap);
    va_end(ap);
    return rc;
}
