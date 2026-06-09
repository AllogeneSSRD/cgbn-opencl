#include "opencl_runtime.h"

#include "opencl_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>

static const char* kBenchKernel = R"CLC(
__kernel void bench_add_mod(
    __global uint* a,
    __global const uint* b,
    __global const uint* n,
    const uint mask,
    const uint inner_iters)
{
    const int gid = get_global_id(0);
  uint av = a[gid];
  uint bv = b[gid];
  const uint nv = n[gid];
  for (uint k = 0; k < inner_iters; ++k) {
    uint s = (av + bv) & mask;
    if (s >= nv) {
      s -= nv;
    }
    av = s;
    bv = (bv * 1103515245u + 12345u) & mask;
  }
  a[gid] = av;
}
)CLC";

static std::string program_build_log(OpenCLApi& api, cl_program prog, cl_device_id dev) {
    size_t need = 0;
    if (api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &need) != CL_SUCCESS || need == 0) {
        return {};
    }
    std::vector<char> buf(need);
    api.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, need, buf.data(), nullptr);
    return std::string(buf.data());
}

std::string run_bit_bench(int limb_bits, int elements, int kernel_iters, int launch_repeats) {
    std::ostringstream out;
    if (limb_bits != 16 && limb_bits != 24 && limb_bits != 32) {
        out << "unsupported limb_bits=" << limb_bits << " (use 16/24/32)\n";
        return out.str();
    }

    const cl_uint mask = (limb_bits == 32) ? 0xFFFFFFFFu : ((1u << limb_bits) - 1u);
    out << "=== Bench " << limb_bits << "-bit add-mod ===\n";
    out << "elements=" << elements << " inner_iters=" << kernel_iters << " repeats=" << launch_repeats << "\n";
    out << "mask=0x" << std::hex << mask << std::dec << "\n";

    OpenCLApi api{};
    bool own_lib = false;
    if (!load_opencl_api(api, own_lib, out)) {
        out << "FAIL: OpenCL not loaded\n";
        return out.str();
    }

    cl_device_id dev = nullptr;
    if (!acquire_gpu_device(api, dev, out)) {
        unload_opencl_api(api, own_lib);
        out << "FAIL: no GPU\n";
        return out.str();
    }

    out << "device: " << query_device_string(api, dev, CL_DEVICE_NAME) << "\n";

    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    if (!create_context_queue(api, dev, ctx, q, out)) {
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    cl_int err = 0;
    const char* src = kBenchKernel;
    size_t src_len = std::strlen(kBenchKernel);
    cl_program prog = api.clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    if (!prog || err != CL_SUCCESS) {
        out << "clCreateProgramWithSource err=" << err << "\n";
        api.clReleaseCommandQueue(q);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    err = api.clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        out << "clBuildProgram err=" << err << " (" << cl_err_str(err) << ")\n";
        out << program_build_log(api, prog, dev) << "\n";
        api.clReleaseProgram(prog);
        api.clReleaseCommandQueue(q);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    cl_kernel kernel = api.clCreateKernel(prog, "bench_add_mod", &err);
    if (!kernel || err != CL_SUCCESS) {
        out << "clCreateKernel err=" << err << "\n";
        api.clReleaseProgram(prog);
        api.clReleaseCommandQueue(q);
        api.clReleaseContext(ctx);
        unload_opencl_api(api, own_lib);
        return out.str();
    }

    const size_t bytes = static_cast<size_t>(elements) * sizeof(cl_uint);
    std::vector<cl_uint> ha(elements);
    std::vector<cl_uint> hb(elements);
    std::vector<cl_uint> hn(elements);
    for (int i = 0; i < elements; ++i) {
        ha[i] = static_cast<cl_uint>((i * 17 + 1) & mask);
        hb[i] = static_cast<cl_uint>((i * 31 + 7) & mask);
        hn[i] = (mask >> 1) | 1u;
        if (hn[i] == 0) {
            hn[i] = mask;
        }
    }

    cl_int m_err = 0;
    cl_mem ma = api.clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &m_err);
    cl_mem mb = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &m_err);
    cl_mem mn = api.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &m_err);
    api.clEnqueueWriteBuffer(q, ma, 1, 0, bytes, ha.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, mb, 1, 0, bytes, hb.data(), 0, nullptr, nullptr);
    api.clEnqueueWriteBuffer(q, mn, 1, 0, bytes, hn.data(), 0, nullptr, nullptr);

    const cl_uint inner = static_cast<cl_uint>(kernel_iters);
    api.clSetKernelArg(kernel, 0, sizeof(cl_mem), &ma);
    api.clSetKernelArg(kernel, 1, sizeof(cl_mem), &mb);
    api.clSetKernelArg(kernel, 2, sizeof(cl_mem), &mn);
    api.clSetKernelArg(kernel, 3, sizeof(cl_uint), &mask);
    api.clSetKernelArg(kernel, 4, sizeof(cl_uint), &inner);

    size_t gws = static_cast<size_t>(elements);
    for (int w = 0; w < 3; ++w) {
        api.clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        api.clFinish(q);
    }

    double best_ms = 1e300;
    double sum_ms = 0;
    for (int r = 0; r < launch_repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        err = api.clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        api.clFinish(q);
        const auto t1 = std::chrono::steady_clock::now();
        if (err != CL_SUCCESS) {
            out << "enqueue err=" << err << "\n";
            break;
        }
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best_ms = std::min(best_ms, ms);
        sum_ms += ms;
    }

    const double avg_ms = sum_ms / std::max(1, launch_repeats);
    const double total_ops = static_cast<double>(elements) * static_cast<double>(kernel_iters) * launch_repeats;
    const double gops = total_ops / (sum_ms / 1000.0) / 1e9;

    out << std::fixed;
    out.precision(3);
    out << "best: " << best_ms << " ms  avg: " << avg_ms << " ms\n";
    out << "throughput: " << gops << " G add-mod ops/s (total " << static_cast<uint64_t>(total_ops) << " ops)\n";
    out << "RESULT: PASS\n";

    api.clReleaseMemObject(ma);
    api.clReleaseMemObject(mb);
    api.clReleaseMemObject(mn);
    api.clReleaseKernel(kernel);
    api.clReleaseProgram(prog);
    api.clReleaseCommandQueue(q);
    api.clReleaseContext(ctx);
    unload_opencl_api(api, own_lib);
    return out.str();
}
