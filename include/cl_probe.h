#ifndef CL_PROBE_H
#define CL_PROBE_H

#include <CL/cl.h>
#include <stdbool.h>

void probePlatforms();
// Interactive device selection and return selected device ID.
cl_device_id chooseDeviceInteractive();

// Non-interactive helpers for ECM main:
// Print all available OpenCL devices and return count.
int listOpenclDevices();
// Validate and set selected device index for current process.
// Returns true on success. If printDevices is true, prints the list first.
bool configureOpenclDeviceIndex(int deviceIndex, bool printDevices);

#endif
