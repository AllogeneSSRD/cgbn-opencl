#include "opencl_android_shim.h"

namespace {

OpenCLApi* g_api = nullptr;

template <typename Fn>
Fn require(Fn* fn, const char* name) {
    if (fn == nullptr) {
        return nullptr;
    }
    return fn;
}

} // namespace

void android_ecm_bind_opencl_api(OpenCLApi* api) {
    g_api = api;
}

void android_ecm_unbind_opencl_api() {
    g_api = nullptr;
}

extern "C" cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    return g_api != nullptr ? g_api->clGetPlatformIDs(num_entries, platforms, num_platforms) : -1;
}

extern "C" cl_int clGetPlatformInfo(
    cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value,
    size_t* param_value_size_ret) {
    return g_api != nullptr ? g_api->clGetPlatformInfo(platform, param_name, param_value_size,
                                                         param_value, param_value_size_ret)
                            : -1;
}

extern "C" cl_int clGetDeviceIDs(
    cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices,
    cl_uint* num_devices) {
    return g_api != nullptr ? g_api->clGetDeviceIDs(platform, device_type, num_entries, devices,
                                                    num_devices)
                            : -1;
}

extern "C" cl_int clGetDeviceInfo(
    cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value,
    size_t* param_value_size_ret) {
    return g_api != nullptr ? g_api->clGetDeviceInfo(device, param_name, param_value_size,
                                                     param_value, param_value_size_ret)
                            : -1;
}

extern "C" cl_context clCreateContext(
    const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices,
    void (*pfn_notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateContext(properties, num_devices, devices, pfn_notify, user_data,
                                  errcode_ret);
}

extern "C" cl_int clReleaseContext(cl_context context) {
    return g_api != nullptr ? g_api->clReleaseContext(context) : -1;
}

extern "C" cl_command_queue clCreateCommandQueue(
    cl_context context, cl_device_id device, cl_ulong properties, cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateCommandQueue(context, device, properties, errcode_ret);
}

extern "C" cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return g_api != nullptr ? g_api->clReleaseCommandQueue(command_queue) : -1;
}

extern "C" cl_mem clCreateBuffer(
    cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

extern "C" cl_int clReleaseMemObject(cl_mem memobj) {
    return g_api != nullptr ? g_api->clReleaseMemObject(memobj) : -1;
}

extern "C" cl_int clEnqueueWriteBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
    size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
    return g_api != nullptr ? g_api->clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                                          offset, size, ptr, num_events_in_wait_list,
                                                          event_wait_list, event)
                            : -1;
}

extern "C" cl_int clEnqueueReadBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size,
    void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return g_api != nullptr ? g_api->clEnqueueReadBuffer(command_queue, buffer, blocking_read,
                                                         offset, size, ptr, num_events_in_wait_list,
                                                         event_wait_list, event)
                            : -1;
}

extern "C" cl_int clFinish(cl_command_queue command_queue) {
    return g_api != nullptr ? g_api->clFinish(command_queue) : -1;
}

extern "C" cl_program clCreateProgramWithSource(
    cl_context context, cl_uint count, const char** strings, const size_t* lengths,
    cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

extern "C" cl_program clCreateProgramWithBinary(
    cl_context context, cl_uint num_devices, const cl_device_id* device_list,
    const size_t* lengths, const unsigned char** binaries, cl_int* binary_status,
    cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries,
                                            binary_status, errcode_ret);
}

extern "C" cl_int clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
    void (*pfn_notify)(cl_program, void*), void* user_data) {
    return g_api != nullptr ? g_api->clBuildProgram(program, num_devices, device_list, options,
                                                    pfn_notify, user_data)
                            : -1;
}

extern "C" cl_int clGetProgramInfo(
    cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value,
    size_t* param_value_size_ret) {
    return g_api != nullptr ? g_api->clGetProgramInfo(program, param_name, param_value_size,
                                                      param_value, param_value_size_ret)
                            : -1;
}

extern "C" cl_int clGetProgramBuildInfo(
    cl_program program, cl_device_id device, cl_program_build_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    return g_api != nullptr ? g_api->clGetProgramBuildInfo(program, device, param_name,
                                                           param_value_size, param_value,
                                                           param_value_size_ret)
                            : -1;
}

extern "C" cl_int clReleaseProgram(cl_program program) {
    return g_api != nullptr ? g_api->clReleaseProgram(program) : -1;
}

extern "C" cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    if (g_api == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = -1;
        }
        return nullptr;
    }
    return g_api->clCreateKernel(program, kernel_name, errcode_ret);
}

extern "C" cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                 const void* arg_value) {
    return g_api != nullptr ? g_api->clSetKernelArg(kernel, arg_index, arg_size, arg_value) : -1;
}

extern "C" cl_int clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset,
    const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
    return g_api != nullptr
               ? g_api->clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
                                               global_work_size, local_work_size,
                                               num_events_in_wait_list, event_wait_list, event)
               : -1;
}

extern "C" cl_int clReleaseKernel(cl_kernel kernel) {
    return g_api != nullptr ? g_api->clReleaseKernel(kernel) : -1;
}
