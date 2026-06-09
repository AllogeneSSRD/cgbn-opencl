#include "opencl_probe.h"

#include <android/log.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "ECM-OpenCL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Minimal OpenCL types (no Khronos headers required at build time).
using cl_int = int;
using cl_uint = unsigned int;
using cl_ulong = unsigned long long;
using cl_bool = cl_uint;
using cl_platform_id = void*;
using cl_device_id = void*;
using cl_context = void*;
using cl_command_queue = void*;
using cl_mem = void*;
using cl_program = void*;
using cl_kernel = void*;

using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_device_type = cl_ulong;
using cl_context_properties = cl_ulong;
using cl_command_queue_properties = cl_ulong;
using cl_mem_flags = cl_ulong;

constexpr cl_uint CL_SUCCESS = 0;
constexpr cl_uint CL_DEVICE_NOT_FOUND = -1;
constexpr cl_ulong CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_ulong CL_DEVICE_TYPE_CPU = 1 << 1;
constexpr cl_ulong CL_DEVICE_TYPE_DEFAULT = 1 << 0;
constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_PLATFORM_VERSION = 0x0901;
constexpr cl_uint CL_PLATFORM_VENDOR = 0x0903;
constexpr cl_uint CL_DEVICE_NAME = 0x1002;
constexpr cl_uint CL_DEVICE_VENDOR = 0x1004;
constexpr cl_uint CL_DEVICE_VERSION = 0x1003;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_DRIVER_VERSION = 0x1101;
constexpr cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1001;
constexpr cl_ulong CL_MEM_READ_WRITE = 1 << 0;

struct OpenCLApi {
    void* lib = nullptr;
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*) = nullptr;
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*) = nullptr;
    cl_context (*clCreateContext)(
        const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*),
        void*, cl_int*) = nullptr;
    cl_int (*clReleaseContext)(cl_context) = nullptr;
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
    cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
    cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*) = nullptr;
    cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
    cl_int (*clEnqueueWriteBuffer)(
        cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void* const*, const void* const*) =
        nullptr;
    cl_int (*clEnqueueReadBuffer)(
        cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void* const*, const void* const*) =
        nullptr;
    cl_int (*clFinish)(cl_command_queue) = nullptr;
};

static bool load_symbol(void* lib, const char* name, void** out) {
    *out = dlsym(lib, name);
    return *out != nullptr;
}

static void preload_vendor_deps(std::ostringstream& log) {
    // Load from device vendor partition (never from APK — pulled .so may fail 16 KB page alignment).
    static const char* kDeps[] = {
        "/vendor/lib64/libvndksupport.so",
        "/system/vendor/lib64/libvndksupport.so",
        "/vendor/lib64/libcutils.so",
        "/system/vendor/lib64/libcutils.so",
    };
    for (const char* path : kDeps) {
        void* h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            log << "preload OK: " << path << "\n";
        }
    }
}

static bool load_opencl_api(OpenCLApi& api, std::ostringstream& log) {
    preload_vendor_deps(log);

    // Device vendor ICD only — do not bundle libOpenCL.so in APK (16 KB page size alignment).
    static const char* kPaths[] = {
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
    };

    for (const char* path : kPaths) {
        api.lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (api.lib) {
            log << "dlopen OK: " << path << "\n";
            break;
        }
        log << "dlopen fail: " << path << " -> " << dlerror() << "\n";
    }
    if (!api.lib) {
        return false;
    }

#define LOAD(field, name)                                                                                              \
    if (!load_symbol(api.lib, name, reinterpret_cast<void**>(&api.field))) {                                           \
        log << "dlsym missing: " << name << "\n";                                                                      \
        return false;                                                                                                  \
    }

    LOAD(clGetPlatformIDs, "clGetPlatformIDs");
    LOAD(clGetPlatformInfo, "clGetPlatformInfo");
    LOAD(clGetDeviceIDs, "clGetDeviceIDs");
    LOAD(clGetDeviceInfo, "clGetDeviceInfo");
    LOAD(clCreateContext, "clCreateContext");
    LOAD(clReleaseContext, "clReleaseContext");
    LOAD(clCreateCommandQueue, "clCreateCommandQueue");
    LOAD(clReleaseCommandQueue, "clReleaseCommandQueue");
    LOAD(clCreateBuffer, "clCreateBuffer");
    LOAD(clReleaseMemObject, "clReleaseMemObject");
    LOAD(clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
    LOAD(clEnqueueReadBuffer, "clEnqueueReadBuffer");
    LOAD(clFinish, "clFinish");
#undef LOAD
    return true;
}

static std::string query_platform_string(OpenCLApi& api, cl_platform_id plat, cl_uint param) {
    size_t need = 0;
    if (api.clGetPlatformInfo(plat, param, 0, nullptr, &need) != CL_SUCCESS || need == 0) {
        return {};
    }
    std::vector<char> buf(need);
    if (api.clGetPlatformInfo(plat, param, need, buf.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!buf.empty() && buf.back() == '\0') {
        return std::string(buf.data());
    }
    return std::string(buf.data(), buf.size());
}

static std::string query_device_string(OpenCLApi& api, cl_device_id dev, cl_uint param) {
    size_t need = 0;
    if (api.clGetDeviceInfo(dev, param, 0, nullptr, &need) != CL_SUCCESS || need == 0) {
        return {};
    }
    std::vector<char> buf(need);
    if (api.clGetDeviceInfo(dev, param, need, buf.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    if (!buf.empty() && buf.back() == '\0') {
        return std::string(buf.data());
    }
    return std::string(buf.data(), buf.size());
}

static const char* device_type_str(cl_ulong type) {
    if (type & CL_DEVICE_TYPE_GPU) {
        return "GPU";
    }
    if (type & CL_DEVICE_TYPE_CPU) {
        return "CPU";
    }
    if (type & CL_DEVICE_TYPE_DEFAULT) {
        return "DEFAULT";
    }
    return "OTHER";
}

std::string probe_opencl() {
    std::ostringstream out;
    out << "=== OpenCL probe (Android) ===\n";

    OpenCLApi api{};
    if (!load_opencl_api(api, out)) {
        out << "RESULT: FAIL (cannot load OpenCL library or symbols)\n";
        LOGI("%s", out.str().c_str());
        return out.str();
    }

    cl_uint num_platforms = 0;
    cl_int err = api.clGetPlatformIDs(0, nullptr, &num_platforms);
    out << "clGetPlatformIDs: err=" << err << " platforms=" << num_platforms << "\n";
    if (err != CL_SUCCESS || num_platforms == 0) {
        out << "RESULT: FAIL (no OpenCL platforms)\n";
        dlclose(api.lib);
        LOGI("%s", out.str().c_str());
        return out.str();
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = api.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        out << "RESULT: FAIL (platform enum err=" << err << ")\n";
        dlclose(api.lib);
        return out.str();
    }

    bool any_device_ok = false;
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_platform_id plat = platforms[pi];
        out << "\n-- Platform " << pi << " --\n";
        out << "  name: " << query_platform_string(api, plat, CL_PLATFORM_NAME) << "\n";
        out << "  vendor: " << query_platform_string(api, plat, CL_PLATFORM_VENDOR) << "\n";
        out << "  version: " << query_platform_string(api, plat, CL_PLATFORM_VERSION) << "\n";

        cl_uint num_devices = 0;
        err = api.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        out << "  devices: err=" << err << " count=" << num_devices << "\n";
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(num_devices);
        err = api.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            out << "  device enum failed err=" << err << "\n";
            continue;
        }

        for (cl_uint di = 0; di < num_devices; ++di) {
            cl_device_id dev = devices[di];
            cl_ulong dev_type = 0;
            api.clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr);

            out << "  -- Device " << di << " (" << device_type_str(dev_type) << ") --\n";
            out << "    name: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";
            out << "    vendor: " << query_device_string(api, dev, CL_DEVICE_VENDOR) << "\n";
            out << "    version: " << query_device_string(api, dev, CL_DEVICE_VERSION) << "\n";
            out << "    driver: " << query_device_string(api, dev, CL_DRIVER_VERSION) << "\n";

            cl_uint cu = 0;
            api.clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            out << "    compute_units: " << cu << "\n";

            cl_int ctx_err = 0;
            cl_context ctx = api.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &ctx_err);
            out << "    clCreateContext: err=" << ctx_err << (ctx ? " OK" : " FAIL") << "\n";
            if (!ctx || ctx_err != CL_SUCCESS) {
                continue;
            }

            cl_int q_err = 0;
            cl_command_queue q = api.clCreateCommandQueue(ctx, dev, 0, &q_err);
            out << "    clCreateCommandQueue: err=" << q_err << (q ? " OK" : " FAIL") << "\n";
            if (!q || q_err != CL_SUCCESS) {
                api.clReleaseContext(ctx);
                continue;
            }

            constexpr size_t kBytes = 16;
            cl_int b_err = 0;
            cl_mem buf = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, kBytes, nullptr, &b_err);
            out << "    clCreateBuffer(16B): err=" << b_err << (buf ? " OK" : " FAIL") << "\n";
            if (!buf || b_err != CL_SUCCESS) {
                api.clReleaseCommandQueue(q);
                api.clReleaseContext(ctx);
                continue;
            }

            unsigned char host_in[kBytes] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            unsigned char host_out[kBytes] = {};
            cl_int w_err = api.clEnqueueWriteBuffer(q, buf, 1, 0, kBytes, host_in, 0, nullptr, nullptr);
            cl_int r_err = api.clEnqueueReadBuffer(q, buf, 1, 0, kBytes, host_out, 0, nullptr, nullptr);
            cl_int f_err = api.clFinish(q);
            const bool data_ok = (w_err == CL_SUCCESS && r_err == CL_SUCCESS && f_err == CL_SUCCESS &&
                                  std::memcmp(host_in, host_out, kBytes) == 0);
            out << "    buffer R/W test: w=" << w_err << " r=" << r_err << " finish=" << f_err
                << (data_ok ? " PASS" : " FAIL") << "\n";

            api.clReleaseMemObject(buf);
            api.clReleaseCommandQueue(q);
            api.clReleaseContext(ctx);
            any_device_ok = any_device_ok || data_ok;
        }
    }

    out << "\nRESULT: " << (any_device_ok ? "PASS (OpenCL usable)" : "PARTIAL (loaded but no working device)") << "\n";
    LOGI("%s", out.str().c_str());
    dlclose(api.lib);
    return out.str();
}
