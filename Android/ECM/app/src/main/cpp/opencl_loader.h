#pragma once

#include <android/log.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "ECM-OpenCL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

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
using cl_event = void*;

using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_device_type = cl_ulong;
using cl_context_properties = cl_ulong;
using cl_command_queue_properties = cl_ulong;
using cl_mem_flags = cl_ulong;
using cl_program_build_info = cl_uint;

constexpr cl_uint CL_SUCCESS = 0;
constexpr cl_uint CL_DEVICE_NOT_FOUND = -1;
constexpr cl_int CL_INVALID_DEVICE_TYPE = -31;
constexpr cl_ulong CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_ulong CL_DEVICE_TYPE_CPU = 1 << 1;
constexpr cl_ulong CL_DEVICE_TYPE_DEFAULT = 1 << 0;
constexpr cl_ulong CL_DEVICE_TYPE_ALL = 0xFFFFFFFFUL;
constexpr cl_uint CL_PLATFORM_NAME = 0x0902;
constexpr cl_uint CL_PLATFORM_VERSION = 0x0901;
constexpr cl_uint CL_PLATFORM_VENDOR = 0x0903;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VENDOR = 0x102C;
constexpr cl_uint CL_DEVICE_VERSION = 0x102D;
constexpr cl_uint CL_DRIVER_VERSION = 0x1101;
constexpr cl_ulong CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_ulong CL_MEM_READ_ONLY = 1 << 2;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

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
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*) =
        nullptr;
    cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*) = nullptr;
    cl_int (*clReleaseProgram)(cl_program) = nullptr;
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
    cl_int (*clEnqueueNDRangeKernel)(
        cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*,
        cl_event*) = nullptr;
    cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
};

bool load_opencl_api(OpenCLApi& api, bool& own_lib, std::ostringstream& log);
void unload_opencl_api(OpenCLApi& api, bool own_lib);

std::string query_platform_string(OpenCLApi& api, cl_platform_id plat, cl_uint param);
std::string query_device_string(OpenCLApi& api, cl_device_id dev, cl_uint param);
const char* cl_err_str(cl_int err);
bool collect_devices(
        OpenCLApi& api,
        cl_platform_id plat,
        cl_ulong dev_type,
        const char* label,
        std::vector<cl_device_id>& devices,
        std::ostringstream& out);
const char* device_type_str(cl_ulong type);

bool acquire_gpu_device(OpenCLApi& api, cl_device_id& dev, std::ostringstream& log);
bool create_context_queue(
        OpenCLApi& api,
        cl_device_id dev,
        cl_context& ctx,
        cl_command_queue& queue,
        std::ostringstream& log);
