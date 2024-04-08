cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
enable_language(HIP)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    set(GINKGO_HIP_PLATFORM "nvidia")
    set(GINKGO_HIP_PLATFORM_NVIDIA ON)
    set(GINKGO_HIP_PLATFORM_AMD OFF)
else()
    set(GINKGO_HIP_PLATFORM "amd")
    set(GINKGO_HIP_PLATFORM_NVIDIA OFF)
    set(GINKGO_HIP_PLATFORM_AMD ON)
endif()


if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HIPBLAS_PATH)
    if(DEFINED ENV{HIPBLAS_PATH})
        set(HIPBLAS_PATH $ENV{HIPBLAS_PATH} CACHE PATH "Path to which HIPBLAS has been installed")
    else()
        set(HIPBLAS_PATH "${ROCM_PATH}/hipblas" CACHE PATH "Path to which HIPBLAS has been installed")
    endif()
endif()

if(NOT DEFINED HIPFFT_PATH)
    if(DEFINED ENV{HIPFFT_PATH})
        set(HIPFFT_PATH $ENV{HIPFFT_PATH} CACHE PATH "Path to which HIPFFT has been installed")
    else()
        set(HIPFFT_PATH "${ROCM_PATH}/hipfft" CACHE PATH "Path to which HIPFFT has been installed")
    endif()
endif()

if(NOT DEFINED HIPRAND_PATH)
    if(DEFINED ENV{HIPRAND_PATH})
        set(HIPRAND_PATH $ENV{HIPRAND_PATH} CACHE PATH "Path to which HIPRAND has been installed")
    else()
        set(HIPRAND_PATH "${ROCM_PATH}/hiprand" CACHE PATH "Path to which HIPRAND has been installed")
    endif()
endif()

if(NOT DEFINED ROCRAND_PATH)
    if(DEFINED ENV{ROCRAND_PATH})
        set(ROCRAND_PATH $ENV{ROCRAND_PATH} CACHE PATH "Path to which ROCRAND has been installed")
    else()
        set(ROCRAND_PATH "${ROCM_PATH}/rocrand" CACHE PATH "Path to which ROCRAND has been installed")
    endif()
endif()

if(NOT DEFINED HIPSPARSE_PATH)
    if(DEFINED ENV{HIPSPARSE_PATH})
        set(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH} CACHE PATH "Path to which HIPSPARSE has been installed")
    else()
        set(HIPSPARSE_PATH "${ROCM_PATH}/hipsparse" CACHE PATH "Path to which HIPSPARSE has been installed")
    endif()
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

if(NOT DEFINED ROCTRACER_PATH)
    if(DEFINED ENV{ROCTRACER_PATH})
        set(ROCTRACER_PATH $ENV{ROCTRACER_PATH} CACHE PATH "Path to which ROCTRACER has been installed")
    else()
        set(ROCTRACER_PATH "${ROCM_PATH}/roctracer" CACHE PATH "Path to which ROCTRACER has been installed")
    endif()
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

## Setup all CMAKE variables to find HIP and its dependencies
set(GINKGO_HIP_MODULE_PATH "${HIP_PATH}/cmake")
list(APPEND CMAKE_MODULE_PATH "${GINKGO_HIP_MODULE_PATH}")
if (GINKGO_HIP_PLATFORM_AND)
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake")
endif()
list(APPEND CMAKE_PREFIX_PATH
    "${HIPBLAS_PATH}/lib/cmake"
    "${HIPFFT_PATH}/lib/cmake"
    "${HIPRAND_PATH}/lib/cmake"
    "${HIPSPARSE_PATH}/lib/cmake"
    "${ROCRAND_PATH}/lib/cmake"
)

find_package(hipblas REQUIRED)
find_package(hipfft) # optional dependency
find_package(hiprand REQUIRED)
find_package(hipsparse REQUIRED)
# At the moment, for hiprand to work also rocrand is required.
find_package(rocrand REQUIRED)
find_package(ROCTX)

if(GINKGO_HIP_AMD_UNSAFE_ATOMIC AND GINKGO_HIP_VERSION VERSION_GREATER_EQUAL 5)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -munsafe-fp-atomics -Wno-unused-command-line-argument")
endif()
set(CMAKE_HIP_STANDARD 14)
