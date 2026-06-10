#pragma once

#include "opencl_loader.h"

// Bind dynamically loaded OpenCL API so cgbn_stage1 (direct cl* calls) works on Android.
void android_ecm_bind_opencl_api(OpenCLApi* api);
void android_ecm_unbind_opencl_api();
