cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
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

find_program(
    HIP_HIPCONFIG_EXECUTABLE
    NAMES hipconfig
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
if(NOT HIP_HIPCONFIG_EXECUTABLE)
    # Now search in default paths
    find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
endif()

execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
        OUTPUT_VARIABLE GINKGO_HIP_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
)

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
