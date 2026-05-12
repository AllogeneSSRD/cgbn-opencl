#include "cl_probe.h"
#include <iostream>
#include <vector>

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }


int main() {
    probePlatforms();
    chooseDeviceInteractive();

    return 0;
}
