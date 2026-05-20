cmake_minimum_required(VERSION 3.20)
project(main)

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)
link_directories("D:/code/vcpkg/installed/x64-windows/lib")

add_executable(main
    src/main.c
    src/cl_probe.cpp
    src/opencl_low_risk_tests.cpp
)

add_library(opencl_ecm_entry STATIC
    src/opencl_ecm_entry.cpp
)

add_executable(ecm
    src/ecm_driver.cpp
    src/cgbn_stage1_opencl.cpp
)
target_sources(ecm PRIVATE
    src/ecm_params_init.cpp
)
target_sources(ecm PRIVATE
)

target_include_directories(ecm PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    # ${CMAKE_SOURCE_DIR}/ecm
)

target_link_libraries(ecm
    opencl_ecm_entry
    cgbn_opencl
    OpenCL::OpenCL
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_include_directories(ecm PRIVATE
    ${CMAKE_SOURCE_DIR}/cgbn/backends/opencl
)

add_executable(opencl_addsub
    src/opencl_addsub_tests.cpp
)

add_executable(opencl_mont
    src/opencl_mont_tests.cpp
)

add_executable(opencl_mont_throughput
    src/opencl_mont_throughput_tests.cpp
)

add_executable(opencl_mont_wg
    src/opencl_mont_wg_tests.cpp
)

add_executable(opencl_ecm_addsub
    src/opencl_ecm_addsub_bench.cpp
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

target_include_directories(opencl_ecm_entry PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    # ${CMAKE_SOURCE_DIR}/ecm
)

target_include_directories(opencl_mont PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(opencl_mont_throughput PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(opencl_mont_wg PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(opencl_ecm_addsub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_addsub
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_link_libraries(opencl_mont
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_link_libraries(opencl_mont_throughput
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_link_libraries(opencl_mont_wg
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_link_libraries(opencl_ecm_addsub
    cgbn_opencl
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

target_compile_definitions(opencl_addsub PRIVATE BUILD_OPENCL_ADDSUB_MAIN=1)
target_compile_definitions(opencl_mont PRIVATE BUILD_OPENCL_MONT_MAIN=1)
target_compile_definitions(opencl_mont_throughput PRIVATE BUILD_OPENCL_MONT_THROUGHPUT_MAIN=1)
target_compile_definitions(opencl_mont_wg PRIVATE BUILD_OPENCL_MONT_WG_MAIN=1)
target_compile_definitions(opencl_ecm_addsub PRIVATE BUILD_OPENCL_ECM_ADDSUB_MAIN=1)

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
