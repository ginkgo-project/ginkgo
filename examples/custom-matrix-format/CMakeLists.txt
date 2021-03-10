cmake_minimum_required(VERSION 3.9)
project(custom-matrix-format CXX CUDA)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.3.0 REQUIRED)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(custom-matrix-format
    custom-matrix-format.cpp
        stencil_kernel.cu)
target_link_libraries(custom-matrix-format Ginkgo::ginkgo)

# workaround for clang-cuda/g++ interaction
set_target_properties(custom-matrix-format PROPERTIES POSITION_INDEPENDENT_CODE ON)
