#include "opencl_runtime.h"

#include "opencl_loader.h"

#include <cstring>

std::string probe_opencl() {
    std::ostringstream out;
    out << "=== OpenCL probe ===\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "RESULT: FAIL (cannot load OpenCL)\n";
        return out.str();
    }

    cl_uint num_platforms = 0;
    cl_int err = api.clGetPlatformIDs(0, nullptr, &num_platforms);
    out << "platforms: " << num_platforms << " (err=" << err << ")\n";
    if (err != CL_SUCCESS || num_platforms == 0) {
        unload_opencl_api(api, own_lib);
        out << "RESULT: FAIL\n";
        return out.str();
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    api.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    bool any_ok = false;
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_platform_id plat = platforms[pi];
        out << "\n-- Platform " << pi << " --\n";
        out << "  name: " << query_platform_string(api, plat, CL_PLATFORM_NAME) << "\n";
        out << "  vendor: " << query_platform_string(api, plat, CL_PLATFORM_VENDOR) << "\n";
        out << "  version: " << query_platform_string(api, plat, CL_PLATFORM_VERSION) << "\n";

        std::vector<cl_device_id> devices;
        collect_devices(api, plat, CL_DEVICE_TYPE_GPU, "GPU", devices, out);
        collect_devices(api, plat, CL_DEVICE_TYPE_CPU, "CPU", devices, out);
        collect_devices(api, plat, CL_DEVICE_TYPE_DEFAULT, "DEFAULT", devices, out);
        out << "  unique devices: " << devices.size() << "\n";

        for (cl_uint di = 0; di < devices.size(); ++di) {
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

            cl_context ctx = nullptr;
            cl_command_queue q = nullptr;
            if (!create_context_queue(api, dev, ctx, q, out)) {
                continue;
            }
            constexpr size_t kBytes = 16;
            cl_int b_err = 0;
            cl_mem buf = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, kBytes, nullptr, &b_err);
            unsigned char host_in[kBytes] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            unsigned char host_out[kBytes] = {};
            api.clEnqueueWriteBuffer(q, buf, 1, 0, kBytes, host_in, 0, nullptr, nullptr);
            api.clEnqueueReadBuffer(q, buf, 1, 0, kBytes, host_out, 0, nullptr, nullptr);
            api.clFinish(q);
            const bool ok = std::memcmp(host_in, host_out, kBytes) == 0;
            out << "    buffer R/W: " << (ok ? "PASS" : "FAIL") << "\n";
            api.clReleaseMemObject(buf);
            api.clReleaseCommandQueue(q);
            api.clReleaseContext(ctx);
            any_ok = any_ok || ok;
        }
    }

    out << "\nRESULT: " << (any_ok ? "PASS (OpenCL usable)" : "PARTIAL") << "\n";
    unload_opencl_api(api, own_lib);
    return out.str();
}

std::string run_short_test() {
    std::ostringstream out;
    out << "=== Short test ===\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "FAIL: OpenCL not loaded\n";
        return out.str();
    }

    cl_device_id dev = nullptr;
    if (!acquire_gpu_device(api, dev, out)) {
        unload_opencl_api(api, own_lib);
        out << "FAIL: no GPU device\n";
        return out.str();
    }

    out << "device: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    if (!create_context_queue(api, dev, ctx, q, out)) {
        unload_opencl_api(api, own_lib);
        out << "FAIL: context/queue\n";
        return out.str();
    }

    cl_int b_err = 0;
    cl_mem buf = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, nullptr, &b_err);
    cl_uint val = 0xA5A5A5A5u;
    cl_uint readback = 0;
    api.clEnqueueWriteBuffer(q, buf, 1, 0, sizeof(val), &val, 0, nullptr, nullptr);
    api.clEnqueueReadBuffer(q, buf, 1, 0, sizeof(readback), &readback, 0, nullptr, nullptr);
    api.clFinish(q);
    const bool ok = (readback == val);
    out << "uint32 ping: " << (ok ? "PASS" : "FAIL") << "\n";
    out << "RESULT: " << (ok ? "PASS" : "FAIL") << "\n";

    api.clReleaseMemObject(buf);
    api.clReleaseCommandQueue(q);
    api.clReleaseContext(ctx);
    unload_opencl_api(api, own_lib);
    return out.str();
}
