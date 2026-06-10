#pragma once

// Android: single OpenCL type source shared with opencl_loader.h and the dlsym shim.
#include "../opencl_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// Extra constants used by cgbn_stage1 / impl_opencl (not needed by bench-only paths).
#ifndef CL_MEM_COPY_HOST_PTR
#define CL_MEM_COPY_HOST_PTR (1 << 3)
#endif
#ifndef CL_DEVICE_LOCAL_MEM_SIZE
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1010
#endif
#ifndef CL_DEVICE_MAX_WORK_GROUP_SIZE
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1014
#endif
#ifndef CL_DEVICE_MAX_MEM_ALLOC_SIZE
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1018
#endif

cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*);
cl_context clCreateContext(
    const cl_context_properties*, cl_uint, const cl_device_id*,
    void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
cl_int clReleaseContext(cl_context);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_ulong, cl_int*);
cl_int clReleaseCommandQueue(cl_command_queue);
cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*);
cl_int clReleaseMemObject(cl_mem);
cl_int clEnqueueWriteBuffer(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*,
    cl_event*);
cl_int clEnqueueReadBuffer(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*,
    cl_event*);
cl_int clFinish(cl_command_queue);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
cl_program clCreateProgramWithBinary(
    cl_context, cl_uint, const cl_device_id*, const size_t*, const unsigned char**, cl_int*, cl_int*);
cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*,
                      void (*)(cl_program, void*), void*);
cl_int clGetProgramInfo(cl_program, cl_program_info, size_t, void*, size_t*);
cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
cl_int clReleaseProgram(cl_program);
cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int clEnqueueNDRangeKernel(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,
    const cl_event*, cl_event*);
cl_int clReleaseKernel(cl_kernel);

#ifdef __cplusplus
}
#endif
