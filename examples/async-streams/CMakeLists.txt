cmake_minimum_required(VERSION 3.9)
project(async-streams CXX CUDA)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
    find_package(OpenMP 3.0 REQUIRED)
endif()

if(NOT (GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP))
    message(FATAL_ERROR
        "This example needs Ginkgo built with CUDA and OpenMP support")
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(async-streams async_streams.cu sqrt_kernel.cu)
target_link_libraries(async-streams Ginkgo::ginkgo OpenMP::OpenMP_CXX)

# inherit CUDA architecture flags from Ginkgo
target_compile_options(async-streams
    PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
target_compile_options(async-streams
    PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>")
# we handle CUDA architecture flags for now, disable CMake handling
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_target_properties(async-streams PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

# workaround for clang-cuda/g++ interaction
set_target_properties(async-streams PROPERTIES POSITION_INDEPENDENT_CODE ON)
