cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

include(cmake/hip_helpers.cmake)
include(CheckLanguage)
check_language(HIP)
ginkgo_check_hip_detection_issue()

enable_language(HIP)

# We keep using NVCC/HCC for consistency with previous releases even if AMD
# updated everything to use NVIDIA/AMD in ROCM 4.1
set(GINKGO_HIP_PLATFORM_NVCC 0)
set(GINKGO_HIP_PLATFORM_HCC 0)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    set(GINKGO_HIP_PLATFORM "nvidia")
    set(GINKGO_HIP_PLATFORM_NVIDIA ON)
    set(GINKGO_HIP_PLATFORM_AMD OFF)
    set(GINKGO_HIP_PLATFORM_NVCC 1)
else()
    set(GINKGO_HIP_PLATFORM "amd")
    set(GINKGO_HIP_PLATFORM_NVIDIA OFF)
    set(GINKGO_HIP_PLATFORM_AMD ON)
    set(GINKGO_HIP_PLATFORM_HCC 1)
endif()

ginkgo_find_hip_version()

find_package(hipblas REQUIRED)
find_package(hipfft) # optional dependency
find_package(hiprand REQUIRED)
find_package(hipsparse REQUIRED)
# At the moment, for hiprand to work also rocrand is required.
find_package(rocrand REQUIRED)
find_package(rocthrust REQUIRED)
find_package(ROCTX)

if(GINKGO_HIP_AMD_UNSAFE_ATOMIC AND GINKGO_HIP_VERSION VERSION_GREATER_EQUAL 5)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -munsafe-fp-atomics -Wno-unused-command-line-argument")
endif()
set(CMAKE_HIP_STANDARD 17)
