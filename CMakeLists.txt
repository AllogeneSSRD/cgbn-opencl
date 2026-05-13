cmake_minimum_required(VERSION 3.20)
project(main)

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)
link_directories("D:/code/vcpkg/installed/x64-windows/lib")

add_executable(main
    src/main.cpp
    src/cl_probe.cpp
    src/opencl_low_risk_tests.cpp
)

add_executable(opencl_addsub
    src/opencl_addsub_tests.cpp
)

add_executable(cuda_addsub_cpu
    src/cuda_addsub_tests.cpp
)

target_include_directories(cuda_addsub_cpu PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_addsub_cpu
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_include_directories(opencl_addsub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_addsub
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_compile_definitions(opencl_addsub PRIVATE BUILD_OPENCL_ADDSUB_MAIN=1)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# OpenCL backend library (static) for tests and later integration
add_library(cgbn_opencl STATIC
    cgbn/backends/opencl/impl_opencl.cpp
)

target_include_directories(cgbn_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}/cgbn/backends/opencl
)

target_link_libraries(cgbn_opencl PRIVATE
    OpenCL::OpenCL
)

target_link_libraries(main
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)
