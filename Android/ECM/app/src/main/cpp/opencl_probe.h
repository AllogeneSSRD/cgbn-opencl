#pragma once

#include <string>

// Probe OpenCL via dlopen(libOpenCL.so). Returns multi-line report for UI/logcat.
std::string probe_opencl();
