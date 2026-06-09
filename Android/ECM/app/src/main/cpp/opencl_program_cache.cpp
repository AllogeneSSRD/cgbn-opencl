#include "opencl_program_cache.h"

#include "jni_utf8.h"

#include <android/log.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {

std::string g_cache_root;
std::unordered_map<std::string, std::vector<unsigned char>> g_ram_bin_cache;
std::unordered_map<std::string, cl_program> g_live_programs;

struct OpenCLCacheSession {
    bool active = false;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_device_id dev = nullptr;
};

OpenCLCacheSession g_session;

uint64_t fnv1a64_update(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t fnv1a64_string(uint64_t h, const std::string& s) {
    h = fnv1a64_update(h, s.data(), s.size());
    const char sep = '\n';
    h = fnv1a64_update(h, &sep, 1);
    return h;
}

bool cache_enabled() {
    return !g_cache_root.empty();
}

bool ensure_dir(const std::string& dir) {
    if (dir.empty()) {
        return false;
    }
    if (mkdir(dir.c_str(), 0755) == 0) {
        return true;
    }
    return errno == EEXIST;
}

std::string cache_dir_path() {
    if (g_cache_root.empty()) {
        return {};
    }
    std::string dir = g_cache_root;
    if (dir.back() != '/') {
        dir += '/';
    }
    dir += "opencl_cache";
    return dir;
}

std::string make_cache_path(
        OpenCLApi& api,
        cl_device_id dev,
        const char* source,
        size_t source_len,
        const char* options) {
    const std::string dev_name = query_device_string(api, dev, CL_DEVICE_NAME);
    const std::string dev_vendor = query_device_string(api, dev, CL_DEVICE_VENDOR);
    const std::string dev_ver = query_device_string(api, dev, CL_DEVICE_VERSION);
    const std::string drv_ver = query_device_string(api, dev, CL_DRIVER_VERSION);
    const std::string opts = options ? options : "";

    uint64_t h = 1469598103934665603ull;
    h = fnv1a64_string(h, dev_name);
    h = fnv1a64_string(h, dev_vendor);
    h = fnv1a64_string(h, dev_ver);
    h = fnv1a64_string(h, drv_ver);
    h = fnv1a64_string(h, opts);
    h = fnv1a64_update(h, source, source_len);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    std::string dir = cache_dir_path();
    if (!dir.empty() && dir.back() != '/') {
        dir += '/';
    }
    return dir + "opencl_" + oss.str() + ".bin";
}

bool load_program_binary(const std::string& path, std::vector<unsigned char>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    if (end <= 0) {
        return false;
    }
    bytes.resize(static_cast<size_t>(end));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return in.good();
}

std::string program_build_log(OpenCLApi& api, cl_program prog, cl_device_id dev) {
    size_t need = 0;
    if (api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &need) != CL_SUCCESS ||
        need == 0) {
        return {};
    }
    std::vector<char> buf(need);
    api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, need, buf.data(), nullptr);
    return sanitize_modified_utf8(std::string(buf.data()));
}

struct SaveResult {
    bool ok = false;
    std::string detail;
    std::vector<unsigned char> bytes;
};

SaveResult extract_program_binary(OpenCLApi& api, cl_program program) {
    // OpenCL spec: probe param sizes with param_value=NULL, then fetch arrays.
    size_t sizes_bytes = 0;
    cl_int err = api.clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, 0, nullptr, &sizes_bytes);
    if (err != CL_SUCCESS || sizes_bytes == 0) {
        size_t single_sz = 0;
        err = api.clGetProgramInfo(
            program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &single_sz, nullptr);
        if (err != CL_SUCCESS) {
            return {false,
                    "clGetProgramInfo(BINARY_SIZES) err=" + std::to_string(err) + " (" +
                        cl_err_str(err) + ")",
                    {}};
        }
        if (single_sz == 0) {
            return {false, "driver returned binary size 0 (program binaries not exported?)", {}};
        }
        sizes_bytes = sizeof(size_t);
    }
    if ((sizes_bytes % sizeof(size_t)) != 0) {
        return {false, "BINARY_SIZES param size misaligned: " + std::to_string(sizes_bytes), {}};
    }

    const size_t num_devices = sizes_bytes / sizeof(size_t);
    std::vector<size_t> sizes(num_devices);
    err = api.clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, sizes_bytes, sizes.data(), nullptr);
    if (err != CL_SUCCESS) {
        return {false,
                "clGetProgramInfo(BINARY_SIZES[]) err=" + std::to_string(err) + " (" +
                    cl_err_str(err) + ")",
                {}};
    }

    size_t bin_idx = 0;
    size_t bin_sz = 0;
    for (size_t i = 0; i < num_devices; ++i) {
        if (sizes[i] > bin_sz) {
            bin_idx = i;
            bin_sz = sizes[i];
        }
    }
    if (bin_sz == 0) {
        return {false, "all program binary sizes are 0", {}};
    }

    std::vector<std::vector<unsigned char>> device_bins(num_devices);
    std::vector<unsigned char*> ptrs(num_devices, nullptr);
    for (size_t i = 0; i < num_devices; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        device_bins[i].resize(sizes[i]);
        ptrs[i] = device_bins[i].data();
    }

    // Spec: param size is num_devices * sizeof(unsigned char*). Some mobile drivers
    // return the binary byte count (e.g. 2902) from a NULL probe — ignore that.
    const size_t bins_param_bytes = num_devices * sizeof(unsigned char*);

    err = api.clGetProgramInfo(
        program, CL_PROGRAM_BINARIES, bins_param_bytes, ptrs.data(), nullptr);
    if (err != CL_SUCCESS && num_devices == 1 && ptrs[0] != nullptr) {
        unsigned char* single_ptr = ptrs[0];
        err = api.clGetProgramInfo(
            program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &single_ptr, nullptr);
    }
    if (err != CL_SUCCESS) {
        std::ostringstream detail;
        detail << "clGetProgramInfo(BINARIES[]) err=" << err << " (" << cl_err_str(err)
               << ") num_devices=" << num_devices << " bin_sz=" << bin_sz
               << " bins_param_bytes=" << bins_param_bytes;
        return {false, detail.str(), {}};
    }

    return {true, std::to_string(bin_sz) + " bytes", std::move(device_bins[bin_idx])};
}

SaveResult save_program_binary(OpenCLApi& api, cl_program program, const std::string& path) {
    SaveResult extracted = extract_program_binary(api, program);
    if (!extracted.ok) {
        return extracted;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        extracted.ok = false;
        extracted.detail = std::string("open failed errno=") + std::strerror(errno);
        return extracted;
    }
    out.write(reinterpret_cast<const char*>(extracted.bytes.data()),
              static_cast<std::streamsize>(extracted.bytes.size()));
    if (!out.good()) {
        extracted.ok = false;
        extracted.detail = "write failed";
        return extracted;
    }
    extracted.detail = std::to_string(extracted.bytes.size()) + " bytes";
    return extracted;
}

bool try_load_cached_program(
        OpenCLApi& api,
        cl_context ctx,
        cl_device_id dev,
        const char* opts,
        const std::vector<unsigned char>& bin,
        std::ostringstream& log,
        cl_program& out_program) {
    if (bin.empty()) {
        return false;
    }
    size_t bin_size = bin.size();
    const unsigned char* bin_ptr = bin.data();
    cl_int err = CL_SUCCESS;
    cl_int bin_status = CL_SUCCESS;
    out_program = api.clCreateProgramWithBinary(ctx, 1, &dev, &bin_size, &bin_ptr, &bin_status, &err);
    if (out_program == nullptr || err != CL_SUCCESS || bin_status != CL_SUCCESS) {
        log << "cache load failed err=" << err << " bin_status=" << bin_status << " ("
            << cl_err_str(bin_status) << ")\n";
        if (out_program != nullptr) {
            api.clReleaseProgram(out_program);
            out_program = nullptr;
        }
        return false;
    }
    err = api.clBuildProgram(out_program, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        log << "cache load build failed err=" << err << " (" << cl_err_str(err) << ")\n";
        log << program_build_log(api, out_program, dev) << "\n";
        api.clReleaseProgram(out_program);
        out_program = nullptr;
        return false;
    }
    return true;
}

int count_cache_bin_files() {
    const std::string dir = cache_dir_path();
    if (dir.empty()) {
        return -1;
    }
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
        return -1;
    }
    int count = 0;
    while (const dirent* ent = readdir(d)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        const size_t len = std::strlen(ent->d_name);
        if (len > 4 && std::strcmp(ent->d_name + len - 4, ".bin") == 0) {
            ++count;
        }
    }
    closedir(d);
    return count;
}

void append_cache_file_list(std::ostringstream& out, int max_files) {
    const std::string dir = cache_dir_path();
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
        out << "  (cannot open dir errno=" << errno << ")\n";
        return;
    }
    int shown = 0;
    while (const dirent* ent = readdir(d)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        const size_t len = std::strlen(ent->d_name);
        if (len <= 4 || std::strcmp(ent->d_name + len - 4, ".bin") != 0) {
            continue;
        }
        struct stat st {};
        const std::string path = dir + "/" + ent->d_name;
        if (stat(path.c_str(), &st) == 0) {
            out << "  " << ent->d_name << " (" << st.st_size << " B)\n";
        } else {
            out << "  " << ent->d_name << "\n";
        }
        if (++shown >= max_files) {
            break;
        }
    }
    closedir(d);
    if (shown == 0) {
        out << "  (empty)\n";
    }
}

} // namespace

void set_opencl_cache_dir(const char* path) {
    g_cache_root = (path != nullptr) ? path : "";
    LOGI("OpenCL cache root: %s", g_cache_root.empty() ? "(disabled)" : g_cache_root.c_str());
}

std::string get_opencl_cache_status() {
    std::ostringstream out;
    out << "=== OpenCL compile cache ===\n";
    if (!cache_enabled()) {
        out << "status: DISABLED (cache root not set; call nativeInitAssets with codeCacheDir)\n";
    } else {
        out << "cache_root: " << g_cache_root << "\n";
        out << "cache_dir:  " << cache_dir_path() << "\n";
        const int count = count_cache_bin_files();
        if (count < 0) {
            out << "cache_dir exists: no (will be created on first compile)\n";
            out << "cached .bin files: 0\n";
        } else {
            out << "cache_dir exists: yes\n";
            out << "cached .bin files: " << count << "\n";
            out << "files:\n";
            append_cache_file_list(out, 8);
        }
    }
    out << "live cached programs (app session): " << g_live_programs.size() << "\n";
    if (!g_live_programs.empty()) {
        out << "note: live cache used when GPU driver cannot export OpenCL program binaries\n";
    }
    out << "adb: adb shell run-as com.example.ecm ls -la code_cache/opencl_cache/\n";
    out << "logcat: adb logcat ECM-OpenCL:I *:S\n";
    return out.str();
}

bool acquire_opencl_cache_session(
        OpenCLApi& api,
        cl_device_id dev,
        cl_context& ctx,
        cl_command_queue& queue,
        std::ostringstream& log) {
    if (g_session.active) {
        ctx = g_session.ctx;
        queue = g_session.q;
        return true;
    }
    if (!create_context_queue(api, dev, g_session.ctx, g_session.q, log)) {
        return false;
    }
    g_session.dev = dev;
    g_session.active = true;
    ctx = g_session.ctx;
    queue = g_session.q;
    log << "opencl session: persistent context for compile cache\n";
    return true;
}

bool opencl_cache_retains_runtime() {
    return g_session.active;
}

void maybe_unload_opencl_api(OpenCLApi& api, bool own_lib) {
    if (!opencl_cache_retains_runtime()) {
        unload_opencl_api(api, own_lib);
    }
}

bool try_live_program_cache(
        OpenCLApi& api,
        const std::string& cache_key,
        std::ostringstream& log,
        double& compile_ms,
        bool& cache_hit,
        cl_program& out_program) {
    const auto it = g_live_programs.find(cache_key);
    if (it == g_live_programs.end()) {
        return false;
    }
    api.clRetainProgram(it->second);
    out_program = it->second;
    cache_hit = true;
    compile_ms = 0.0;
    log << "compile: live cache hit (app session, no driver binary export)\n";
    LOGI("OpenCL live cache hit: %s", cache_key.c_str());
    return true;
}

void store_live_program_cache(OpenCLApi& api, const std::string& cache_key, cl_program program) {
    const auto [it, inserted] = g_live_programs.emplace(cache_key, program);
    if (inserted) {
        api.clRetainProgram(program);
        LOGI("OpenCL live cache store: %s", cache_key.c_str());
    }
}

cl_program build_opencl_program_cached(
        OpenCLApi& api,
        cl_context ctx,
        cl_device_id dev,
        const char* source,
        size_t source_len,
        const char* build_opts,
        std::ostringstream& log,
        double& compile_ms,
        bool& cache_hit) {
    cache_hit = false;
    compile_ms = 0.0;
    if (source == nullptr || source_len == 0) {
        return nullptr;
    }

    const char* opts = build_opts ? build_opts : "";
    const std::string cache_path = make_cache_path(api, dev, source, source_len, opts);
    const std::string cache_dir = cache_dir_path();

    log << "cache_enabled: " << (cache_enabled() ? "yes" : "no") << "\n";
    if (cache_enabled()) {
        log << "cache_dir: " << cache_dir << "\n";
    } else {
        log << "cache_dir: (not set)\n";
    }
    log << "cache_key: " << cache_path << "\n";

    {
        cl_program live = nullptr;
        if (try_live_program_cache(api, cache_path, log, compile_ms, cache_hit, live)) {
            return live;
        }
    }

    if (cache_enabled()) {
        const auto cache_t0 = std::chrono::steady_clock::now();
        if (!ensure_dir(g_cache_root)) {
            log << "cache mkdir root failed errno=" << errno << "\n";
        }
        if (!ensure_dir(cache_dir)) {
            log << "cache mkdir dir failed errno=" << errno << "\n";
        }

        if (const auto it = g_ram_bin_cache.find(cache_path); it != g_ram_bin_cache.end()) {
            cl_program program = nullptr;
            if (try_load_cached_program(api, ctx, dev, opts, it->second, log, program)) {
                compile_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - cache_t0)
                                   .count();
                cache_hit = true;
                log << "compile: cache hit (ram) " << compile_ms << " ms (bin_kib="
                    << (it->second.size() / 1024u) << ")\n";
                LOGI("OpenCL cache hit (ram): %s (%zu bytes, %.1f ms)", cache_path.c_str(),
                     it->second.size(), compile_ms);
                return program;
            }
            log << "cache stale (ram), rebuilding\n";
        }

        std::vector<unsigned char> bin;
        if (load_program_binary(cache_path, bin) && !bin.empty()) {
            cl_program program = nullptr;
            if (try_load_cached_program(api, ctx, dev, opts, bin, log, program)) {
                compile_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - cache_t0)
                                   .count();
                cache_hit = true;
                log << "compile: cache hit (disk) " << compile_ms << " ms (bin_kib="
                    << (bin.size() / 1024u) << ")\n";
                LOGI("OpenCL cache hit (disk): %s (%zu bytes, %.1f ms)", cache_path.c_str(),
                     bin.size(), compile_ms);
                g_ram_bin_cache[cache_path] = std::move(bin);
                return program;
            }
            log << "cache stale (disk), rebuilding\n";
            LOGI("OpenCL cache stale: %s", cache_path.c_str());
        } else {
            log << "cache miss (no file or unreadable)\n";
        }
    }

    const auto build_t0 = std::chrono::steady_clock::now();
    cl_int err = CL_SUCCESS;
    const char* src_ptr = source;
    cl_program program = api.clCreateProgramWithSource(ctx, 1, &src_ptr, &source_len, &err);
    if (!program || err != CL_SUCCESS) {
        log << "clCreateProgramWithSource err=" << err << "\n";
        return nullptr;
    }

    err = api.clBuildProgram(program, 1, &dev, opts, nullptr, nullptr);
    compile_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_t0).count();
    if (err != CL_SUCCESS) {
        log << "clBuildProgram err=" << err << " (" << cl_err_str(err) << ")\n";
        log << program_build_log(api, program, dev) << "\n";
        api.clReleaseProgram(program);
        return nullptr;
    }

    log << "compile: " << compile_ms << " ms (src_kib=" << (source_len / 1024u) << ")\n";
    if (cache_enabled()) {
        SaveResult extracted = extract_program_binary(api, program);
        if (extracted.ok) {
            g_ram_bin_cache[cache_path] = extracted.bytes;
            std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                log << "cache save failed: open errno=" << errno
                    << " (ram cache kept for this session)\n";
                LOGI("OpenCL cache disk open failed, ram cache kept: %s", cache_path.c_str());
            } else {
                out.write(reinterpret_cast<const char*>(extracted.bytes.data()),
                          static_cast<std::streamsize>(extracted.bytes.size()));
                if (out.good()) {
                    log << "cache save: " << cache_path << " (" << extracted.detail << ")\n";
                    LOGI("OpenCL cache save: %s (%s)", cache_path.c_str(), extracted.detail.c_str());
                } else {
                    log << "cache save failed: write (ram cache kept for this session)\n";
                    LOGI("OpenCL cache disk write failed, ram cache kept: %s", cache_path.c_str());
                }
            }
        } else {
            log << "cache extract failed: " << extracted.detail << "\n";
            log << "cache: keeping live cl_program for this app session\n";
            LOGI("OpenCL cache extract failed: %s", extracted.detail.c_str());
            store_live_program_cache(api, cache_path, program);
        }
    } else {
        store_live_program_cache(api, cache_path, program);
    }
    return program;
}
