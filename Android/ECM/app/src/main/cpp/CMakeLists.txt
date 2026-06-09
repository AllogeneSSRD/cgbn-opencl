cmake_minimum_required(VERSION 3.22.1)
project("ecm")

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        opencl_loader.cpp
        opencl_probe.cpp
        opencl_bench.cpp)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log
        dl)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
