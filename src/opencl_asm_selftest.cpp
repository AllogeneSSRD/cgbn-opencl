#include "cl_probe.h"
#include "cgbn_opencl.h"

#include <CL/cl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct InstrCase {
    int what;
    const char *name;
    int latency_ops_per_iter;
    int throughput_ops_per_iter;
};

const InstrCase kCases[] = {
    {0, "V_NOP", 1, 4},
    {1, "V_ADD_I32", 1, 4},
    {2, "V_FMA_F32", 1, 4},
    {3, "V_ADD_F64", 1, 4},
    {4, "V_FMA_F64", 1, 4},
    {5, "V_MUL_F64", 1, 4},
    {6, "V_MAD_U64_U32", 1, 4},
};

bool has_amd_gpu(const cgbn::opencl::context_t &ctx) {
    char pname[1024] = {0};
    char dname[1024] = {0};
    clGetPlatformInfo(ctx.platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
    clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    std::string p = pname;
    std::string d = dname;
    for (char &c : p) c = (char)tolower((unsigned char)c);
    for (char &c : d) c = (char)tolower((unsigned char)c);
    return p.find("amd") != std::string::npos || d.find("amd") != std::string::npos ||
           d.find("gfx") != std::string::npos || d.find("radeon") != std::string::npos;
}

bool has_valid_sink(const std::vector<int64_t> &sink) {
    for (int64_t v : sink) if (v >= 0) return true;
    return false;
}

} // namespace

int main() {
    if (!configureFirstAmdGpuDevice(true)) {
        return EXIT_FAILURE;
    }

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return EXIT_FAILURE;
    }
    if (!has_amd_gpu(ctx)) {
        std::cerr << "Selected device is not AMD GPU; aborting selftest." << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_int qerr = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue prof_queue = clCreateCommandQueueWithProperties(ctx.ctx, ctx.device, props, &qerr);
#else
    cl_command_queue prof_queue = clCreateCommandQueue(ctx.ctx, ctx.device, CL_QUEUE_PROFILING_ENABLE, &qerr);
#endif
    if (qerr != CL_SUCCESS || prof_queue == nullptr) {
        std::cerr << "Failed to create profiling queue: " << qerr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    std::string src = cgbn::opencl::load_text_file("cgbn/backends/opencl/kernels/selftest.cl");
    if (src.empty()) {
        std::cerr << "Failed to load selftest.cl" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_int buildErr = CL_SUCCESS;
    const char *opts = "-DSELFTEST_USE_BUILTIN_CLOCK=1 -DSELFTEST_DISABLE_MAD_U64=1";
    cl_program prog = cgbn::opencl::build_program_from_source(ctx, src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build selftest.cl: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    cl_kernel k_lat = clCreateKernel(prog, "testLatency", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testLatency: " << err << std::endl;
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_kernel k_thr = clCreateKernel(prog, "testThroughput", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testThroughput: " << err << std::endl;
        clReleaseKernel(k_lat);
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    const size_t groups = 4096;
    const size_t global = groups * 64;
    const size_t local = 64;
    const int it_latency = 256;
    const int it_throughput = 512;
    std::vector<int64_t> host(groups, 0);
    cl_mem buf = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(int64_t) * host.size(), host.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create timing buffer: " << err << std::endl;
        clReleaseKernel(k_lat);
        clReleaseKernel(k_thr);
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    std::cout << "Running AMD ISA selftest on first AMD GPU..." << std::endl;
    std::cout << "Build options: " << opts << std::endl;
    std::cout << "Global=" << global << " Local=" << local
              << " it_latency=" << it_latency
              << " it_throughput=" << it_throughput << std::endl;
    for (const auto &tc : kCases) {
        int what = tc.what;
        std::fill(host.begin(), host.end(), 0);
        clEnqueueWriteBuffer(prof_queue, buf, CL_TRUE, 0, sizeof(int64_t) * host.size(),
                             host.data(), 0, nullptr, nullptr);

        // Latency-style dependent chain timing
        cl_event ev_lat = nullptr;
        int iters = it_latency;
        err = clSetKernelArg(k_lat, 0, sizeof(int), &what);
        err |= clSetKernelArg(k_lat, 1, sizeof(int), &iters);
        err |= clSetKernelArg(k_lat, 2, sizeof(cl_mem), &buf);
        if (err != CL_SUCCESS) {
            std::cerr << "clSetKernelArg latency failed for " << tc.name << ": " << err << std::endl;
            continue;
        }
        err = clEnqueueNDRangeKernel(prof_queue, k_lat, 1, nullptr, &global, &local, 0, nullptr, &ev_lat);
        if (err != CL_SUCCESS) {
            std::cerr << "Kernel enqueue latency failed for " << tc.name << ": " << err << std::endl;
            if (ev_lat) clReleaseEvent(ev_lat);
            continue;
        }
        clFinish(prof_queue);
        err = clEnqueueReadBuffer(prof_queue, buf, CL_TRUE, 0, sizeof(int64_t) * host.size(),
                                  host.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Read latency sink failed for " << tc.name << ": " << err << std::endl;
            if (ev_lat) clReleaseEvent(ev_lat);
            continue;
        }
        if (!has_valid_sink(host)) {
            std::cout << tc.name << " : unsupported on this compiler/device (skipped)" << std::endl;
            if (ev_lat) clReleaseEvent(ev_lat);
            continue;
        }
        cl_ulong lat_start = 0, lat_end = 0;
        clGetEventProfilingInfo(ev_lat, CL_PROFILING_COMMAND_START, sizeof(lat_start), &lat_start, nullptr);
        clGetEventProfilingInfo(ev_lat, CL_PROFILING_COMMAND_END, sizeof(lat_end), &lat_end, nullptr);
        clReleaseEvent(ev_lat);
        double lat_ns_total = (double)(lat_end - lat_start);
        double lat_ns_per_op = lat_ns_total / (double(global) * double(it_latency) * double(tc.latency_ops_per_iter));

        // Throughput-style independent streams timing
        cl_event ev_thr = nullptr;
        iters = it_throughput;
        err = clSetKernelArg(k_thr, 0, sizeof(int), &what);
        err |= clSetKernelArg(k_thr, 1, sizeof(int), &iters);
        err |= clSetKernelArg(k_thr, 2, sizeof(cl_mem), &buf);
        if (err != CL_SUCCESS) {
            std::cerr << "clSetKernelArg throughput failed for " << tc.name << ": " << err << std::endl;
            continue;
        }
        err = clEnqueueNDRangeKernel(prof_queue, k_thr, 1, nullptr, &global, &local, 0, nullptr, &ev_thr);
        if (err != CL_SUCCESS) {
            std::cerr << "Kernel enqueue throughput failed for " << tc.name << ": " << err << std::endl;
            if (ev_thr) clReleaseEvent(ev_thr);
            continue;
        }
        clFinish(prof_queue);
        cl_ulong thr_start = 0, thr_end = 0;
        clGetEventProfilingInfo(ev_thr, CL_PROFILING_COMMAND_START, sizeof(thr_start), &thr_start, nullptr);
        clGetEventProfilingInfo(ev_thr, CL_PROFILING_COMMAND_END, sizeof(thr_end), &thr_end, nullptr);
        clReleaseEvent(ev_thr);
        double thr_ns_total = (double)(thr_end - thr_start);
        double thr_total_ops = double(global) * double(it_throughput) * double(tc.throughput_ops_per_iter);
        double gops = thr_total_ops / thr_ns_total;

        cl_uint mhz = 0;
        clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
        double cycles_per_op_est = lat_ns_per_op * (double(mhz) * 1e-3);

        std::cout << tc.name
                  << " : latency " << lat_ns_per_op << " ns/op"
                  << " (~" << cycles_per_op_est << " cycles/op @ " << mhz << "MHz)"
                  << "; throughput " << gops << " Gops/s" << std::endl;
    }

    clReleaseMemObject(buf);
    clReleaseKernel(k_lat);
    clReleaseKernel(k_thr);
    clReleaseProgram(prog);
    clReleaseCommandQueue(prof_queue);
    cgbn::opencl::destroy_context(ctx);
    return EXIT_SUCCESS;
}
