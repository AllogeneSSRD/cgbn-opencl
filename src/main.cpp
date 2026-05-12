#include "cl_probe.h"
#include "cgbn_opencl.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }


int main() {
    probePlatforms();

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL init failed: " << err << std::endl;
        return EXIT_FAILURE;
    }

    char deviceName[256] = {0};
    clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "\n[OpenCL smoke test] Selected device: " << deviceName << std::endl;

    const char *kernelSource = R"CLC(
        __kernel void smoke_test(__global int *data) {
            int gid = get_global_id(0);
            data[gid] = data[gid] + 1;
        }
    )CLC";

    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, kernelSource, "", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "OpenCL program build failed: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    std::cout << "[OpenCL smoke test] Program build succeeded." << std::endl;

    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    return 0;
}
