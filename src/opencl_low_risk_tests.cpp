#include "opencl_low_risk_tests.h"

#include "cgbn_opencl.h"

#include <CL/cl.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using LimbVec = std::vector<cl_uint>;
constexpr size_t kLimbs = 4;

void checkCl(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        std::cerr << message << " Error: " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

LimbVec cpuSetUi32(cl_uint value) {
    LimbVec result(kLimbs, 0u);
    result[0] = value;
    return result;
}

LimbVec cpuBitwiseAnd(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] & b[i];
    }
    return result;
}

LimbVec cpuBitwiseIor(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] | b[i];
    }
    return result;
}

LimbVec cpuBitwiseXor(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

LimbVec cpuAddUi32(const LimbVec &a, cl_uint value) {
    LimbVec result(kLimbs, 0u);
    std::uint64_t carry = value;
    for (size_t i = 0; i < kLimbs; ++i) {
        std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + carry;
        result[i] = static_cast<cl_uint>(sum);
        carry = sum >> 32;
    }
    return result;
}

LimbVec cpuSubUi32(const LimbVec &a, cl_uint value) {
    LimbVec result(kLimbs, 0u);
    std::uint64_t borrow = value;
    for (size_t i = 0; i < kLimbs; ++i) {
        std::uint64_t minuend = static_cast<std::uint64_t>(a[i]);
        std::uint64_t wide = minuend - borrow;
        result[i] = static_cast<cl_uint>(wide);
        borrow = minuend < borrow ? 1u : 0u;
    }
    return result;
}

bool sameLimbs(const LimbVec &lhs, const LimbVec &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}

bool runUnarySetUi32(cl_context ctx, cl_command_queue queue, cl_program program, cl_uint value) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, "cgbn_set_ui32", &err);
    checkCl(err, "Failed to create cgbn_set_ui32 kernel");

    LimbVec output(kLimbs, 0xDEADBEEFu);
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set output arg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &value);
    checkCl(err, "Failed to set value arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue cgbn_set_ui32");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read cgbn_set_ui32 output");

    const LimbVec expected = cpuSetUi32(value);
    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runBinaryBitwise(cl_context ctx, cl_command_queue queue, cl_program program,
                      const char *kernelName, const LimbVec &lhs, const LimbVec &rhs,
                      const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem bBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(rhs.data()), &err);
    checkCl(err, "Failed to create rhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 2");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue bitwise kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read bitwise output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runUnaryAddSub(cl_context ctx, cl_command_queue queue, cl_program program,
                    const char *kernelName, const LimbVec &lhs, cl_uint value,
                    const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &value);
    checkCl(err, "Failed to set value arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue add/sub kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read add/sub output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

std::string loadKernelSource() {
    const std::array<const char *, 3> candidates = {
        "cgbn/backends/opencl/kernels/base.cl",
        "../cgbn/backends/opencl/kernels/base.cl",
        "../../cgbn/backends/opencl/kernels/base.cl"
    };

    for (const char *candidate : candidates) {
        std::string source = cgbn::opencl::load_text_file(candidate);
        if (!source.empty()) {
            return source;
        }
    }

    return R"CLC(
__kernel void cgbn_set_ui32(__global uint *out, uint value, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }
    if (limbs > 0u) {
        out[0] = value;
    }
}

__kernel void cgbn_bitwise_and(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] & b[i];
    }
}

__kernel void cgbn_bitwise_ior(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] | b[i];
    }
}

__kernel void cgbn_bitwise_xor(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

__kernel void cgbn_add_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong carry = (ulong)value;
    for (uint i = 0; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + carry;
        out[i] = (uint)sum;
        carry = sum >> 32;
    }
}

__kernel void cgbn_sub_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong borrow = (ulong)value;
    for (uint i = 0; i < limbs; ++i) {
        ulong minuend = (ulong)a[i];
        ulong wide = minuend - borrow;
        out[i] = (uint)wide;
        borrow = minuend < borrow ? 1u : 0u;
    }
}
)CLC";
}

} // namespace

bool runOpenClLowRiskOperatorTests() {
    std::cout << "\n[OpenCL low-risk tests] starting" << std::endl;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    std::string source = loadKernelSource();
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, source.c_str(), "", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build OpenCL kernel program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return false;
    }

    const LimbVec lhs = {0xFFFF0000u, 0x12345678u, 0xAAAAAAAAu, 0x00000001u};
    const LimbVec rhs = {0x0000FFFFu, 0x87654321u, 0x55555555u, 0x00000002u};

    bool ok = true;
    ok = runUnarySetUi32(ctx.ctx, ctx.queue, program, 0xDEADBEEFu) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_and", lhs, rhs, cpuBitwiseAnd(lhs, rhs)) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_ior", lhs, rhs, cpuBitwiseIor(lhs, rhs)) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_xor", lhs, rhs, cpuBitwiseXor(lhs, rhs)) && ok;
    ok = runUnaryAddSub(ctx.ctx, ctx.queue, program, "cgbn_add_ui32", lhs, 0x11111111u, cpuAddUi32(lhs, 0x11111111u)) && ok;
    ok = runUnaryAddSub(ctx.ctx, ctx.queue, program, "cgbn_sub_ui32", lhs, 0x11111111u, cpuSubUi32(lhs, 0x11111111u)) && ok;

    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    if (ok) {
        std::cout << "[OpenCL low-risk tests] all tests passed" << std::endl;
    } else {
        std::cerr << "[OpenCL low-risk tests] at least one test failed" << std::endl;
    }

    return ok;
}
