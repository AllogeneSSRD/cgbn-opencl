#include "opencl_loader.h"

static bool load_symbol(void* lib, const char* name, void** out) {
    *out = dlsym(lib, name);
    return *out != nullptr;
}

static bool try_dlopen_name(const char* name, void** handle, std::ostringstream& log) {
    dlerror();
    *handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (*handle != nullptr) {
        log << "dlopen OK: " << name << "\n";
        return true;
    }
    const char* err = dlerror();
    log << "dlopen fail: " << name << " -> " << (err != nullptr ? err : "?") << "\n";
    return false;
}

bool load_opencl_api(OpenCLApi& api, bool& own_lib, std::ostringstream& log) {
    own_lib = false;
    static const char* kNames[] = {"libOpenCL.so", "OpenCL"};
    for (const char* name : kNames) {
        if (try_dlopen_name(name, &api.lib, log)) {
            own_lib = true;
            break;
        }
    }
    if (api.lib == nullptr) {
        dlerror();
        api.lib = dlopen("libOpenCL.so", RTLD_NOLOAD | RTLD_GLOBAL);
        if (api.lib != nullptr) {
            log << "dlopen NOLOAD OK: libOpenCL.so\n";
        }
    }
    if (api.lib == nullptr && dlsym(nullptr, "clGetPlatformIDs") != nullptr) {
        log << "symbols OK via RTLD_DEFAULT\n";
    } else if (api.lib == nullptr) {
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
    LOAD(clCreateProgramWithSource, "clCreateProgramWithSource");
    LOAD(clCreateProgramWithBinary, "clCreateProgramWithBinary");
    LOAD(clBuildProgram, "clBuildProgram");
    LOAD(clGetProgramInfo, "clGetProgramInfo");
    LOAD(clGetProgramBuildInfo, "clGetProgramBuildInfo");
    LOAD(clRetainProgram, "clRetainProgram");
    LOAD(clReleaseProgram, "clReleaseProgram");
    LOAD(clCreateKernel, "clCreateKernel");
    LOAD(clSetKernelArg, "clSetKernelArg");
    LOAD(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    LOAD(clReleaseKernel, "clReleaseKernel");
#undef LOAD
    return true;
}

void unload_opencl_api(OpenCLApi& api, bool own_lib) {
    if (own_lib && api.lib != nullptr) {
        dlclose(api.lib);
        api.lib = nullptr;
    }
}

std::string query_platform_string(OpenCLApi& api, cl_platform_id plat, cl_uint param) {
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

std::string query_device_string(OpenCLApi& api, cl_device_id dev, cl_uint param) {
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

const char* cl_err_str(cl_int err) {
    switch (err) {
        case 0:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -30:
            return "CL_INVALID_VALUE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -48:
            return "CL_INVALID_KERNEL";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        default:
            return "CL_ERROR";
    }
}

bool collect_devices(
        OpenCLApi& api,
        cl_platform_id plat,
        cl_ulong dev_type,
        const char* label,
        std::vector<cl_device_id>& devices,
        std::ostringstream& out) {
    cl_uint count = 0;
    cl_int err = api.clGetDeviceIDs(plat, dev_type, 0, nullptr, &count);
    out << "  query " << label << ": err=" << err << " (" << cl_err_str(err) << ") count=" << count << "\n";
    if (err != CL_SUCCESS || count == 0) {
        return false;
    }
    std::vector<cl_device_id> batch(count);
    err = api.clGetDeviceIDs(plat, dev_type, count, batch.data(), nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }
    for (cl_device_id dev : batch) {
        bool seen = false;
        for (cl_device_id existing : devices) {
            if (existing == dev) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            devices.push_back(dev);
        }
    }
    return true;
}

const char* device_type_str(cl_ulong type) {
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

bool acquire_gpu_device(OpenCLApi& api, cl_device_id& dev, std::ostringstream& log) {
    cl_uint num_platforms = 0;
    if (api.clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        log << "no platforms\n";
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (api.clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    for (cl_platform_id plat : platforms) {
        std::vector<cl_device_id> devices;
        collect_devices(api, plat, CL_DEVICE_TYPE_GPU, "GPU", devices, log);
        collect_devices(api, plat, CL_DEVICE_TYPE_DEFAULT, "DEFAULT", devices, log);
        if (!devices.empty()) {
            dev = devices[0];
            return true;
        }
    }
    return false;
}

bool create_context_queue(
        OpenCLApi& api,
        cl_device_id dev,
        cl_context& ctx,
        cl_command_queue& queue,
        std::ostringstream& log) {
    cl_int err = 0;
    ctx = api.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!ctx || err != CL_SUCCESS) {
        log << "clCreateContext err=" << err << "\n";
        return false;
    }
    queue = api.clCreateCommandQueue(ctx, dev, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        log << "clCreateCommandQueue err=" << err << "\n";
        api.clReleaseContext(ctx);
        ctx = nullptr;
        return false;
    }
    return true;
}
