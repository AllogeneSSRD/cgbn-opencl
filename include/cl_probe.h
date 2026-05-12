#ifndef CL_PROBE_H
#define CL_PROBE_H

#include <CL/cl.h>

void probePlatforms();
// 交互式选择设备并返回选中的设备 ID
cl_device_id chooseDeviceInteractive();

#endif
