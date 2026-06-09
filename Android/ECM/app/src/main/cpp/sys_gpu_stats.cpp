#include <android/api-level.h>
#include <android/log.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include <sys/system_properties.h>

#define LOG_TAG "ECM-GPUStats"

namespace {

constexpr int kMaxWalkDepth = 5;

enum class Source : int {
    None = 0,
    Property = 1,
    Sysfs = 2,
};

struct GpuReadResult {
    int busy_percent = -1;
    long long freq_hz = -1;
    long long max_freq_hz = -1;
    Source busy_src = Source::None;
    Source freq_src = Source::None;
    std::string busy_detail;
    std::string freq_detail;
};

bool read_first_line(const char* path, std::string& out) {
    FILE* fp = std::fopen(path, "re");
    if (fp == nullptr) {
        return false;
    }
    char buf[256] = {0};
    if (std::fgets(buf, sizeof(buf), fp) == nullptr) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    out = buf;
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return !out.empty();
}

bool get_system_property(const char* name, char* out, size_t out_len) {
    if (name == nullptr || out == nullptr || out_len == 0) {
        return false;
    }
    out[0] = '\0';
    const int len = __system_property_get(name, out);
    return len > 0 && out[0] != '\0';
}

bool contains_ci(const char* hay, const char* needle) {
    if (hay == nullptr || needle == nullptr) {
        return false;
    }
    std::string h(hay);
    std::string n(needle);
    for (auto& c : h) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (auto& c : n) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return h.find(n) != std::string::npos;
}

bool is_property_blacklisted(const char* name) {
    if (name == nullptr) {
        return true;
    }
    static const char* kBlacklist[] = {
        "computility",
        "gpulevel",
        "partition",
        "profiler",
        "driver",
        "support",
        "enable",
        "hdr",
        "turbo",
        "qspa",
        "gfx.driver",
        "gpubw",
    };
    for (const char* bad : kBlacklist) {
        if (contains_ci(name, bad)) {
            return true;
        }
    }
    return false;
}

int parse_busy_percent(const std::string& raw, bool gpubusy_pair) {
    if (raw.empty()) {
        return -1;
    }
    if (gpubusy_pair) {
        unsigned long long busy = 0;
        unsigned long long total = 0;
        if (std::sscanf(raw.c_str(), "%llu %llu", &busy, &total) == 2 && total > 0) {
            const unsigned long long pct = (busy * 100ULL) / total;
            return static_cast<int>(pct > 100 ? 100 : pct);
        }
        return -1;
    }
    char* end = nullptr;
    const long v = std::strtol(raw.c_str(), &end, 10);
    if (end == raw.c_str()) {
        return -1;
    }
    if (v < 0) {
        return 0;
    }
    if (v <= 100) {
        return static_cast<int>(v);
    }
    if (v <= 1000) {
        return static_cast<int>(v / 10);
    }
    return 100;
}

long long parse_freq_hz_from_digits(long long v) {
    if (v <= 0) {
        return -1;
    }
    // Qualcomm vendor.gpu.freq is typically MHz (e.g. 1100).
    if (v < 10'000LL) {
        return v * 1'000'000LL;
    }
    // Already Hz or KHz.
    if (v < 10'000'000LL) {
        return v * 1'000LL;
    }
    return v;
}

long long parse_freq_hz(const std::string& raw) {
    if (raw.empty()) {
        return -1;
    }
    char* end = nullptr;
    long long v = std::strtoll(raw.c_str(), &end, 10);
    if (end == raw.c_str() || v <= 0) {
        return -1;
    }
    return parse_freq_hz_from_digits(v);
}

void try_property_freq(const char* name, const char* value, GpuReadResult& result) {
    if (is_property_blacklisted(name)) {
        return;
    }
    if (result.freq_hz > 0 || value == nullptr || value[0] == '\0') {
        return;
    }
    const long long hz = parse_freq_hz(value);
    if (hz > 0) {
        result.freq_hz = hz;
        result.freq_src = Source::Property;
        result.freq_detail = name;
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "freq_hz=%lld prop=%s val=%s", hz, name, value);
    }
}

void try_property_busy(const char* name, const char* value, GpuReadResult& result) {
    if (is_property_blacklisted(name)) {
        return;
    }
    if (result.busy_percent >= 0 || value == nullptr || value[0] == '\0') {
        return;
    }
    const int pct = parse_busy_percent(value, false);
    if (pct >= 0) {
        result.busy_percent = pct;
        result.busy_src = Source::Property;
        result.busy_detail = name;
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "busy=%d prop=%s val=%s", pct, name, value);
    }
}

void read_known_gpu_properties(GpuReadResult& result) {
    static const char* kFreqProps[] = {
        "vendor.gpu.freq",
        "vendor.gpu.clock",
        "vendor.gpu.current_freq",
        "vendor.gpu.cur_freq",
        "vendor.qti.gpu.freq",
        "vendor.qti.gfx.gpu.freq",
        "vendor.postboot.parsed.freq",
        "ro.vendor.gpu.freq",
        "persist.vendor.gpu.freq",
        "debug.egl.gpu.freq",
        "vendor.gpu.ddr.freq",
        "vendor.gpu.gfx.freq",
    };
    static const char* kBusyProps[] = {
        "vendor.gpu.busy",
        "vendor.gpu.load",
        "vendor.gpu.utilization",
        "vendor.gpu.usage",
        "vendor.qti.gpu.busy",
        "vendor.qti.gpu.load",
        "debug.egl.gpu.busy",
    };

    char value[PROP_VALUE_MAX] = {0};
    for (const char* name : kFreqProps) {
        if (get_system_property(name, value, sizeof(value))) {
            try_property_freq(name, value, result);
        }
    }
    for (const char* name : kBusyProps) {
        if (get_system_property(name, value, sizeof(value))) {
            try_property_busy(name, value, result);
        }
    }
}

bool name_matches_gpu(const char* name) {
    if (name == nullptr) {
        return false;
    }
    return std::strstr(name, "kgsl") != nullptr || std::strstr(name, "gpu") != nullptr ||
           std::strstr(name, "mali") != nullptr || std::strstr(name, "G3D") != nullptr;
}

bool leaf_is_busy(const char* name) {
    return std::strcmp(name, "gpu_busy_percentage") == 0 || std::strcmp(name, "busy_percentage") == 0 ||
           std::strcmp(name, "gpubusy") == 0 || std::strcmp(name, "utilization") == 0 ||
           std::strcmp(name, "gpu_busy") == 0;
}

bool leaf_is_freq(const char* name) {
    return std::strcmp(name, "cur_freq") == 0 || std::strcmp(name, "gpuclk") == 0 ||
           std::strcmp(name, "clock") == 0 || std::strcmp(name, "gpu_clock") == 0;
}

void walk_dir(const char* dir_path, int depth, bool gpu_context,
              std::vector<std::string>& busy_paths, std::vector<std::string>& freq_paths) {
    if (depth > kMaxWalkDepth) {
        return;
    }
    DIR* dir = opendir(dir_path);
    if (dir == nullptr) {
        return;
    }
    const bool in_gpu_tree = gpu_context || name_matches_gpu(dir_path);
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        std::string child = std::string(dir_path) + "/" + ent->d_name;
        if (leaf_is_busy(ent->d_name) && in_gpu_tree) {
            busy_paths.push_back(child);
        }
        if (leaf_is_freq(ent->d_name) && in_gpu_tree) {
            freq_paths.push_back(child);
        }
        if (ent->d_type == DT_DIR || ent->d_type == DT_UNKNOWN) {
            const bool child_gpu = in_gpu_tree || name_matches_gpu(ent->d_name);
            walk_dir(child.c_str(), depth + 1, child_gpu, busy_paths, freq_paths);
        }
    }
    closedir(dir);
}

void collect_sysfs_paths(std::vector<std::string>& busy_paths, std::vector<std::string>& freq_paths) {
    static const char* kFixedBusy[] = {
        // 高通骁龙 Adreno
        "/sys/class/kgsl/kgsl-3d0/gpubusy", // Adreno 642Lv1
        "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
        "/sys/class/kgsl/kgsl-3d0/busy_percentage",
        "/sys/devices/virtual/kgsl/kgsl-3d0/gpu_busy_percentage",

        // 联发科平台
        "/sys/class/misc/mali0/device/utilization",
        "/sys/class/misc/mali0/device/gpu_utilization",

        // 三星Exynos
        "/sys/kernel/gpu/gpu_busy_percentage",
        "/sys/kernel/gpu/gpu_clock",
        
        // 华为麒麟
        "/sys/devices/platform/*.gpu/clock",
        
        // 通用备用路径
        "/sys/class/drm/card0/device/clock",
        "/sys/class/drm/card0/device/gpuclk",
    };
    static const char* kFixedFreq[] = {
        // 高通骁龙 Adreno
        "/sys/class/kgsl/kgsl-3d0/gpuclk",
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
        "/sys/class/kgsl/kgsl-3d0/clk_freq",
        // 联发科平台
        "/sys/class/misc/mali0/device/clock",
        "/proc/gpufreq/gpufreq_opp_dump",        // 联发科频率表，包含当前频率
        // 三星Exynos
        "/sys/kernel/gpu/gpu_clock",
        // 华为麒麟
        "/sys/devices/platform/*.gpu/clock",

        "/sys/class/devfreq/fde60000.gpu/available_frequencies", // 可用频率列表
        "/sys/class/devfreq/fde60000.gpu/cur_freq",              // 当前频率
    };
    for (const char* p : kFixedBusy) {
        busy_paths.emplace_back(p);
    }
    for (const char* p : kFixedFreq) {
        freq_paths.emplace_back(p);
    }
    static const char* kRoots[] = {
        "/sys/class/kgsl",
        "/sys/class/devfreq",
        "/sys/devices/virtual/kgsl",
        "/sys/devices/platform",
    };
    for (const char* root : kRoots) {
        walk_dir(root, 0, std::strcmp(root, "/sys/class/devfreq") == 0, busy_paths, freq_paths);
    }
}

int read_sysfs_busy_once(const std::vector<std::string>& busy_paths) {
    for (const std::string& path : busy_paths) {
        std::string line;
        if (!read_first_line(path.c_str(), line)) {
            continue;
        }
        const bool gpubusy_pair = path.find("gpubusy") != std::string::npos;
        const int pct = parse_busy_percent(line, gpubusy_pair);
        if (pct >= 0) {
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "busy=%d sysfs=%s", pct, path.c_str());
            return pct;
        }
    }
    return -1;
}

void read_sysfs_gpu(GpuReadResult& result) {
    std::vector<std::string> busy_paths;
    std::vector<std::string> freq_paths;
    collect_sysfs_paths(busy_paths, freq_paths);

    for (int attempt = 0; attempt < 3 && result.busy_percent < 0; ++attempt) {
        const int pct = read_sysfs_busy_once(busy_paths);
        if (pct >= 0) {
            result.busy_percent = pct;
            result.busy_src = Source::Sysfs;
            result.busy_detail = "kgsl";
            break;
        }
    }

    for (const std::string& path : freq_paths) {
        std::string line;
        if (!read_first_line(path.c_str(), line)) {
            continue;
        }
        const long long hz = parse_freq_hz(line);
        if (hz > 0) {
            result.freq_hz = hz;
            result.freq_src = Source::Sysfs;
            result.freq_detail = path;
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "freq_hz=%lld sysfs=%s", hz, path.c_str());
            break;
        }
    }

    if (result.max_freq_hz <= 0) {
        static const char* kMaxFreq[] = {
            "/sys/class/kgsl/kgsl-3d0/max_gpuclk",
            "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq",
        };
        for (const char* path : kMaxFreq) {
            std::string line;
            if (!read_first_line(path, line)) {
                continue;
            }
            const long long hz = parse_freq_hz(line);
            if (hz > 0) {
                result.max_freq_hz = hz;
                __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "max_freq_hz=%lld sysfs=%s", hz, path);
                break;
            }
        }
    }
}

} // namespace

// out: [busy%, cur_freq_hz, busy_source, freq_source, max_freq_hz]
extern "C" void ecm_read_gpu_stats_ex(int* out_busy_percent, long long* out_freq_hz, int* out_busy_src,
                                      int* out_freq_src, long long* out_max_freq_hz) {
    GpuReadResult result;

    // 1) sysfs first (Adreno 642 gpubusy; most accurate when accessible).
    read_sysfs_gpu(result);

    // 2) Explicit vendor property whitelist only (no broad foreach — avoids MIUI gpulevel false positive).
    if (result.busy_percent < 0 || result.freq_hz <= 0) {
        read_known_gpu_properties(result);
    }

    if (out_busy_percent != nullptr) {
        *out_busy_percent = result.busy_percent;
    }
    if (out_freq_hz != nullptr) {
        *out_freq_hz = result.freq_hz;
    }
    if (out_busy_src != nullptr) {
        *out_busy_src = static_cast<int>(result.busy_src);
    }
    if (out_freq_src != nullptr) {
        *out_freq_src = static_cast<int>(result.freq_src);
    }
    if (out_max_freq_hz != nullptr) {
        *out_max_freq_hz = result.max_freq_hz;
    }

    if (result.busy_percent < 0 && result.freq_hz <= 0 && result.max_freq_hz <= 0) {
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "GPU stats unavailable (sysfs+whitelist props). 830 may lack kgsl sysfs access.");
    }
}

extern "C" void ecm_read_gpu_stats(int* out_busy_percent, long long* out_freq_hz) {
    ecm_read_gpu_stats_ex(out_busy_percent, out_freq_hz, nullptr, nullptr, nullptr);
}
