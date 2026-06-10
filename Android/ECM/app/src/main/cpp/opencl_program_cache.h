#pragma once

#include "opencl_loader.h"

#include <cstddef>
#include <sstream>
#include <string>

// App cache root (e.g. Context.getCodeCacheDir()); subdir opencl_cache/ is used.
void set_opencl_cache_dir(const char* path);
std::string get_opencl_cache_dir();

// Human-readable cache dir status (for probe UI / adb debugging).
std::string get_opencl_cache_status();

// Persistent OpenCL context/queue for compile cache (must not clReleaseContext per bench run).
bool acquire_opencl_cache_session(
        OpenCLApi& api,
        cl_device_id dev,
        cl_context& ctx,
        cl_command_queue& queue,
        std::ostringstream& log);

// When true, do not dlclose OpenCL — live cached programs depend on the loaded runtime.
bool opencl_cache_retains_runtime();

void maybe_unload_opencl_api(OpenCLApi& api, bool own_lib);

// Build from source with device-binary cache (same key scheme as cgbn opencl impl).
// On cache hit, compile_ms is load+clBuildProgram time; cache_hit=true.
// When the driver cannot export program binaries, a live cl_program is kept for the app session.
cl_program build_opencl_program_cached(
        OpenCLApi& api,
        cl_context ctx,
        cl_device_id dev,
        const char* source,
        size_t source_len,
        const char* build_opts,
        std::ostringstream& log,
        double& compile_ms,
        bool& cache_hit);
