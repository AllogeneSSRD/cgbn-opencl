#include "cgbn_opencl.h"
#include <cstring>
#include <fstream>
#include <vector>
#include <iostream>

namespace cgbn {
namespace opencl {

cl_int create_context(context_t &out) {
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) return err ? err : -1;

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) return err;

    // pick first platform and first device (DEVICE_TYPE_ALL)
    out.platform = platforms[0];
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(out.platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) return err ? err : -2;

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(out.platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
    if (err != CL_SUCCESS) return err;

    out.device = devices[0];
    out.ctx = clCreateContext(NULL, 1, &out.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // create a command queue (non-deprecated compatible API)
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = { 0 };
    out.queue = clCreateCommandQueueWithProperties(out.ctx, out.device, props, &err);
#else
    out.queue = clCreateCommandQueue(out.ctx, out.device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        clReleaseContext(out.ctx);
        out.ctx = nullptr;
        return err;
    }

    return CL_SUCCESS;
}

cl_int destroy_context(context_t &c) {
    cl_int err = CL_SUCCESS;
    if (c.queue) { err = clReleaseCommandQueue(c.queue); c.queue = nullptr; }
    if (c.ctx) { err = clReleaseContext(c.ctx); c.ctx = nullptr; }
    c.device = nullptr;
    c.platform = nullptr;
    return err;
}

cl_program build_program_from_source(context_t &ctx, const char *source, const char *options, cl_int &errcode) {
    size_t src_len = strlen(source);
    const char *src = source;
    cl_int err;
    cl_program program = clCreateProgramWithSource(ctx.ctx, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS) { errcode = err; return nullptr; }

    err = clBuildProgram(program, 1, &ctx.device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        // fetch build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        std::cerr << "OpenCL build error:\n" << log << std::endl;
        errcode = err;
        return program; // still return program so caller can inspect log
    }
    errcode = CL_SUCCESS;
    return program;
}

std::string load_text_file(const char *path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) return std::string();
    std::string content;
    ifs.seekg(0, std::ios::end);
    content.resize((size_t)ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    ifs.read(&content[0], content.size());
    return content;
}

} // namespace opencl
} // namespace cgbn
