if(WIN32)
    cmake_minimum_required(VERSION 3.9)
else()
    cmake_minimum_required(VERSION 3.8)
endif()

option(BUILD_GPU "Compile kernels for NVIDIA GPUs" ON)

if(BUILD_GPU)
    project(Ginkgo LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD_REQUIRED true)
    set(CMAKE_CUDA_STANDARD 11)
else()
    project(Ginkgo LANGUAGES CXX)
endif()

option(BUILD_TESTS "Generate build files for unit tests" ON)
option(DEVEL_TOOLS "Add development tools to the build system" ON)
option(BUILD_REFERENCE "Compile reference CPU kernels" OFF)
option(BUILD_CPU "Compile kernels for CPUs" OFF)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if(BUILD_TESTS)
    enable_testing()
endif()

if(BUILD_GPU)
    add_subdirectory(gpu)
endif()

add_subdirectory(third_party)  # Third-party tools and libraries
add_subdirectory(core)         # Core Ginkgo types and top-level functions

