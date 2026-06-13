cmake_minimum_required(VERSION 3.20)
project(main)

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)

# Desktop GMP (vcpkg x64-windows). Android uses ECM_ANDROID_GMP_ROOT in app/CMakeLists.txt.
set(ECM_WINDOWS_GMP_ROOT "" CACHE PATH
    "Prefix with include/gmp.h and lib/gmp.lib (e.g. vcpkg installed/x64-windows)")
if(NOT ECM_WINDOWS_GMP_ROOT)
    if(DEFINED ENV{VCPKG_ROOT})
        set(ECM_WINDOWS_GMP_ROOT "$ENV{VCPKG_ROOT}/installed/x64-windows")
    elseif(EXISTS "D:/code/vcpkg/installed/x64-windows/include/gmp.h")
        set(ECM_WINDOWS_GMP_ROOT "D:/code/vcpkg/installed/x64-windows")
    endif()
endif()
find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS "${ECM_WINDOWS_GMP_ROOT}/include")
find_library(GMP_LIBRARY NAMES gmp libgmp HINTS "${ECM_WINDOWS_GMP_ROOT}/lib")
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR
        "GMP not found (include=${GMP_INCLUDE_DIR}, lib=${GMP_LIBRARY}). "
        "Install: vcpkg install gmp:x64-windows "
        "or pass -DECM_WINDOWS_GMP_ROOT=<prefix>")
endif()
message(STATUS "GMP: include=${GMP_INCLUDE_DIR} lib=${GMP_LIBRARY}")

function(ecm_target_link_gmp target)
    target_include_directories(${target} PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(${target} ${GMP_LIBRARY})
endfunction()

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

target_compile_definitions(cgbn_opencl PRIVATE
    "ECM_KERNEL_ROOT_DEFAULT=\"${CMAKE_SOURCE_DIR}/kernels/opencl\""
)

add_library(opencl_ecm_entry STATIC
    src/opencl_ecm_entry.cpp
)

target_include_directories(opencl_ecm_entry PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(opencl_ecm_entry ${GMP_LIBRARY})

add_executable(ecm
    src/cgbn_stage1_opencl.cpp
    src/cl_probe.cpp
    src/ecm_driver.cpp
    src/ecm_params_init.cpp
    src/opencl_ecm_checkpoint.cpp
    src/opencl_ecm_save.cpp
    src/opencl_ecm_debug_utils.cpp
    src/opencl_ecm_log.cpp
    src/opencl_ecm_mont.cpp
    src/opencl_ecm_path_registry.cpp
    src/opencl_ecm_selftest.cpp
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
)

ecm_target_link_gmp(ecm)

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
)

ecm_target_link_gmp(main)

add_executable(opencl_ecm_addsub
    src/opencl_ecm_addsub_bench.cpp
    src/opencl_ecm_addsub_manifest.cpp
)

target_include_directories(opencl_ecm_addsub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(opencl_ecm_addsub PRIVATE BUILD_OPENCL_ECM_ADDSUB_MAIN=1)

target_link_libraries(opencl_ecm_addsub
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

ecm_target_link_gmp(opencl_ecm_addsub)

add_executable(opencl_ecm_montsqr
    src/opencl_ecm_montsqr_bench.cpp
)

target_include_directories(opencl_ecm_montsqr PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(opencl_ecm_montsqr PRIVATE BUILD_OPENCL_ECM_MONTSQR_MAIN=1)

target_link_libraries(opencl_ecm_montsqr
    OpenSSL::Crypto
    OpenSSL::SSL
    cgbn_opencl
)

ecm_target_link_gmp(opencl_ecm_montsqr)

if(MSVC)
    # NVIDIA OpenCL JIT of mont_priv_opt.cl can overflow the default 1MB thread stack.
    target_link_options(opencl_ecm_montsqr PRIVATE "/STACK:16777216")
endif()

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

add_executable(opencl_addsub_isa_export
    src/cl_probe.cpp
    src/opencl_addsub_isa_export.cpp
    src/opencl_ecm_addsub_manifest.cpp
)

target_include_directories(opencl_addsub_isa_export PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(opencl_addsub_isa_export
    OpenCL::OpenCL
    cgbn_opencl
)
