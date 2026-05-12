cmake_minimum_required(VERSION 3.20)
project(main)

# set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenSSL REQUIRED)
link_directories("D:/code/vcpkg/installed/x64-windows/lib")

add_executable(main
    src/main.cpp
    src/cl_probe.cpp
    src/initialization.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(main
    OpenCL::OpenCL
    OpenSSL::SSL
    OpenSSL::Crypto
    "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
)
