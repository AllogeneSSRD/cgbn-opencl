cmake_minimum_required(VERSION 3.20)
project(main)

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)
link_directories("D:/code/vcpkg/installed/x64-windows/lib")

# OpenCL backend library (static) for tests and later integration
add_library(cgbn_opencl STATIC
    cgbn/backends/opencl/impl_opencl.cpp
)

target_include_directories(cgbn_opencl PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cgbn/backends/opencl
)

target_link_libraries(cgbn_opencl PRIVATE
    OpenCL::OpenCL
)

add_library(opencl_ecm_entry STATIC
    src/opencl_ecm_entry.cpp
)

target_include_directories(opencl_ecm_entry PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(ecm
    src/cgbn_stage1_opencl.cpp
    src/cl_probe.cpp
    src/ecm_driver.cpp
    src/ecm_params_init.cpp
    src/opencl_ecm_debug_utils.cpp
    src/opencl_ecm_log.cpp
)

target_include_directories(ecm PRIVATE
    ${CMAKE_SOURCE_DIR}/cgbn/backends/opencl
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ecm
    OpenCL::OpenCL
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
    opencl_ecm_entry
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

add_executable(main
    src/cl_probe.cpp
    src/main_test.cpp
    src/opencl_low_risk_tests.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(main
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

add_executable(opencl_ecm_addsub
    src/opencl_ecm_addsub_bench.cpp
)

target_include_directories(opencl_ecm_addsub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(opencl_ecm_addsub PRIVATE BUILD_OPENCL_ECM_ADDSUB_MAIN=1)

target_link_libraries(opencl_ecm_addsub
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)

add_executable(opencl_asm_selftest
    src/cl_probe.cpp
    src/opencl_asm_selftest.cpp
)

target_include_directories(opencl_asm_selftest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_asm_selftest
    OpenCL::OpenCL
    cgbn_opencl
)

add_executable(opencl_mont_isa_export
    src/cl_probe.cpp
    src/opencl_mont_isa_export.cpp
)

target_include_directories(opencl_mont_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_mont_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)
