cmake_minimum_required(VERSION 3.18)
project(custom-matrix-format CXX CUDA)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
    find_package(OpenMP 3.0 REQUIRED)
endif()

if(NOT (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP))
    message(FATAL_ERROR
        "This example needs Ginkgo built with CUDA and OpenMP support")
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(custom-matrix-format custom-matrix-format.cpp stencil_kernel.cu)
target_link_libraries(custom-matrix-format Ginkgo::ginkgo OpenMP::OpenMP_CXX)

# workaround for clang-cuda/g++ interaction
set_target_properties(custom-matrix-format PROPERTIES POSITION_INDEPENDENT_CODE ON)
