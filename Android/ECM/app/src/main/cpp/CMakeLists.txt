cmake_minimum_required(VERSION 3.22.1)
project("ecm")

set(MPA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../..")

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        opencl_loader.cpp
        opencl_program_cache.cpp
        opencl_probe.cpp
        opencl_bench.cpp
        opencl_addsub_bench.cpp
        opencl_montsqr_bench.cpp
        kernel_assets.cpp
        sys_gpu_stats.cpp
        "${MPA_ROOT}/src/opencl_ecm_addsub_manifest.cpp"
        "${MPA_ROOT}/src/opencl_ecm_montsqr_manifest.cpp")

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        "${MPA_ROOT}/include")

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        dl)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
