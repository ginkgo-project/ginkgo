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

if(NOT DEFINED HIPSPARSE_PATH)
    if(DEFINED ENV{HIPSPARSE_PATH})
        set(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH} CACHE PATH "Path to which HIPSPARSE has been installed")
    else()
        set(HIPSPARSE_PATH "${ROCM_PATH}/hipsparse" CACHE PATH "Path to which HIPSPARSE has been installed")
    endif()
endif()

## Both the definition of `HCC_PATH` and `HIP_HIPCC_CMAKE_LINKER_HELPER` are required
## before including `FindHIP`, as these are essential but not defined in the beginning
## of the `FindHIP` file itself. Not defining these currently results in:
## 1. Without `HCC_PATH`: the `hcc` backend not working properly if it is wrongly set,
##    if it is not set, popentially all compilation could fail.
## 2. Without `HIP_HIPCC_CMAKE_LINKER_HELPER` two compilations are required, since
##    `FindHIP` defines this only in macro calls, which we call much later on after
##    including the file itself.
if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

# Find HIPCC_CMAKE_LINKER_HELPER executable
find_program(
    HIP_HIPCC_CMAKE_LINKER_HELPER
    NAMES hipcc_cmake_linker_helper
    PATHS
    "${HIP_ROOT_DIR}"
    ENV ROCM_PATH
    ENV HIP_PATH
    /opt/rocm
    /opt/rocm/hip
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
if(NOT HIP_HIPCC_CMAKE_LINKER_HELPER)
    # Now search in default paths
    find_program(HIP_HIPCC_CMAKE_LINKER_HELPER hipcc_cmake_linker_helper)
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
set(GINKGO_HIP_VERSION ${GINKGO_HIP_VERSION} PARENT_SCOPE)

if (GINKGO_HIP_PLATFORM MATCHES "nvcc") # ensure ENV{CUDA_PATH} is set by the user
    if (NOT DEFINED ENV{CUDA_PATH})
        find_path(GINKGO_HIP_DEFAULT_CUDA_PATH "cuda.h" PATH /usr/local/cuda/include NO_DEFAULT_PATH)
        if (NOT GINKGO_HIP_DEFAULT_CUDA_PATH)
            message(FATAL_ERROR "HIP nvcc backend was requested but CUDA could not be located. "
                "Set and export the environment variable CUDA_PATH.")
         endif()
     endif()
endif()

if (GINKGO_HIP_PLATFORM STREQUAL "hcc")
    # This is required by hipblas/hipsparse in the case where the platform is hcc.
    # For nvcc platform, these aren't required and only cause trouble.
    list(APPEND CMAKE_PREFIX_PATH
        "${HIP_PATH}/lib/cmake"
        "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies, e.g. hcc
    )
endif()


## Setup all CMAKE variables to find HIP and its dependencies
list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")
list(APPEND CMAKE_PREFIX_PATH
    "${HIPBLAS_PATH}/lib/cmake"
    "${HIPSPARSE_PATH}/lib/cmake"
)
# Set CMAKE_MODULE_PATH and CMAKE_PREFIX_PATH as PARENT_SCOPE to easily find HIP again
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

# setting the default flags like CMAKE_{LANG}_FLAGS_{TYPE}
# the setting is copied from the default CMAKE_CXX_FLAGS_{TYPE}
set(HIP_HIPCC_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Flags used by the HIPCC compiler during DEBUG builds")
set(HIP_HIPCC_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}" CACHE STRING "Flags used by the HIPCC compiler during MINSIZEREL builds")
set(HIP_HIPCC_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "Flags used by the HIPCC compiler during RELEASE builds")
set(HIP_HIPCC_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the HIPCC compiler during RELWITHDEBINFO builds")

find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)
find_path(GINKGO_HIP_THRUST_PATH "thrust/complex.h"
    PATHS "${HIP_PATH}/../include"
    ENV HIP_THRUST_PATH)
if (NOT GINKGO_HIP_THRUST_PATH)
    message(FATAL_ERROR "Could not find the ROCm header thrust/complex.h which is required by Ginkgo HIP.")
endif()

set(GINKGO_HIP_SOURCES
    base/exception.hip.cpp
    base/executor.hip.cpp
    base/version.hip.cpp
    components/fill_array.hip.cpp
    components/precision_conversion.hip.cpp
    components/prefix_sum.hip.cpp
    factorization/ilu_kernels.hip.cpp
    factorization/factorization_kernels.hip.cpp
    factorization/par_ict_kernels.hip.cpp
    factorization/par_ilu_kernels.hip.cpp
    factorization/par_ilut_approx_filter_kernel.hip.cpp
    factorization/par_ilut_filter_kernel.hip.cpp
    factorization/par_ilut_select_common.hip.cpp
    factorization/par_ilut_select_kernel.hip.cpp
    factorization/par_ilut_spgeam_kernel.hip.cpp
    factorization/par_ilut_sweep_kernel.hip.cpp
    matrix/coo_kernels.hip.cpp
    matrix/csr_kernels.hip.cpp
    matrix/dense_kernels.hip.cpp
    matrix/ell_kernels.hip.cpp
    matrix/hybrid_kernels.hip.cpp
    matrix/sellp_kernels.hip.cpp
    matrix/sparsity_csr_kernels.hip.cpp
    preconditioner/isai_kernels.hip.cpp
    preconditioner/jacobi_advanced_apply_kernel.hip.cpp
    preconditioner/jacobi_generate_kernel.hip.cpp
    preconditioner/jacobi_kernels.hip.cpp
    preconditioner/jacobi_simple_apply_kernel.hip.cpp
    solver/bicg_kernels.hip.cpp
    solver/bicgstab_kernels.hip.cpp
    solver/cg_kernels.hip.cpp
    solver/cgs_kernels.hip.cpp
    solver/fcg_kernels.hip.cpp
    solver/gmres_kernels.hip.cpp
    solver/ir_kernels.hip.cpp
    solver/lower_trs_kernels.hip.cpp
    solver/upper_trs_kernels.hip.cpp
    stop/criterion_kernels.hip.cpp
    stop/residual_norm_kernels.hip.cpp)

set(GINKGO_HIP_NVCC_ARCH "")
if (GINKGO_HIP_PLATFORM MATCHES "nvcc")
    if (NOT CMAKE_CUDA_HOST_COMPILER AND NOT GINKGO_CUDA_DEFAULT_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
    elseif(GINKGO_CUDA_DEFAULT_HOST_COMPILER)
        unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    endif()
    if (CMAKE_CUDA_HOST_COMPILER)
        set(GINKGO_HIP_CUDA_HOST_COMPILER "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    # Remove false positive CUDA warnings when calling one<T>() and zero<T>()
    # This creates a compilation bug on nvcc 9.0.102 *with* the new array_deleter
    # merged at commit ed12b3df5d26
    if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
        set(GINKGO_HIP_NVCC_ADDITIONAL_FLAGS --expt-relaxed-constexpr)
    endif()
    # add gpu architecture flags
    include(CudaArchitectureSelector)
    cas_variable_cuda_architectures(GINKGO_HIP_NVCC_ARCH
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21")
endif()
set(GINKGO_HIPCC_OPTIONS ${GINKGO_HIP_COMPILER_FLAGS})
set(GINKGO_HIP_NVCC_OPTIONS ${GINKGO_HIP_NVCC_COMPILER_FLAGS} ${GINKGO_HIP_NVCC_ARCH} ${GINKGO_HIP_NVCC_ADDITIONAL_FLAGS})
set(GINKGO_HIP_HCC_OPTIONS ${GINKGO_HIP_HCC_COMPILER_FLAGS})
set(GINKGO_HIP_CLANG_OPTIONS ${GINKGO_HIP_CLANG_COMPILER_FLAGS})

set_source_files_properties(${GINKGO_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
if (GINKGO_HIP_VERSION VERSION_GREATER_EQUAL "3.5")
    hip_add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES}
        HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} "-std=c++11"
        HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS}
        CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
        NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS} ${GINKGO_HIP_CUDA_HOST_COMPILER}
        ${GINKGO_STATIC_OR_SHARED})
else()
    hip_add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES}
        HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} "-std=c++11"
        HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS}
        NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS} ${GINKGO_HIP_CUDA_HOST_COMPILER}
        ${GINKGO_STATIC_OR_SHARED})
endif()

if(GINKGO_HIP_AMDGPU AND GINKGO_HIP_PLATFORM MATCHES "hcc")
    foreach(target ${GINKGO_HIP_AMDGPU})
        target_compile_options(ginkgo_hip PRIVATE --amdgpu-target=${target})
        target_link_libraries(ginkgo_hip PRIVATE --amdgpu-target=${target})
    endforeach()
endif()

target_compile_options(ginkgo_hip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
if(GINKGO_WITH_CLANG_TIDY AND GINKGO_CLANG_TIDY_PATH)
    set_property(TARGET ginkgo_hip PROPERTY CXX_CLANG_TIDY "${GINKGO_CLANG_TIDY_PATH};-checks=*")
endif()
if(GINKGO_WITH_IWYU AND GINKGO_IWYU_PATH)
    set_property(TARGET ginkgo_hip PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${GINKGO_IWYU_PATH})
endif()

if(GINKGO_HIP_PLATFORM MATCHES "hcc")
    # Fix the exception thrown bug with `hcc` backend and shared libraries
    set_target_properties(ginkgo_hip PROPERTIES LINKER_LANGUAGE HIP)

    # Ban `-hc` flag as INTERFACE_LINK_LIBRARIES since that is propagated when building
    # a static library, and it's definitely not a known option to any compiler.
    ginkgo_hip_ban_link_hcflag(hcc::hccrt)

    if (NOT BUILD_SHARED_LIBS)
        # Do not let hip::device flags propagate to executables which don't
        # directly use HIP
        ginkgo_hip_clang_ban_hip_device_flags()
    endif()
    target_link_libraries(ginkgo_hip PRIVATE hip::device)
elseif(GINKGO_HIP_PLATFORM MATCHES "nvcc")
    find_package(CUDA 9.0 REQUIRED)
    target_link_libraries(ginkgo_hip PUBLIC ${CUDA_LIBRARIES})
    set(HIP_CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
endif()

target_link_libraries(ginkgo_hip PRIVATE roc::hipblas roc::hipsparse)

target_include_directories(ginkgo_hip
    PUBLIC
        ${HIP_INCLUDE_DIRS}
    PRIVATE
        ${GINKGO_HIP_THRUST_PATH}
        ${HIPBLAS_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>)

ginkgo_compile_features(ginkgo_hip)
ginkgo_default_includes(ginkgo_hip)
ginkgo_install_library(ginkgo_hip hip)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_hip)
endif()

if(GINKGO_BUILD_TESTS)
  # Here, we go through all of Ginkgo's dependencies to build a `-Wl,-rpath` string since for
  # some reason `hipcc` through CMake does not have rpath settings unlike the other compilers.
    get_target_property(GINKGO_LINK_LIBRARIES ginkgo LINK_LIBRARIES)
    set(GINKGO_RPATH_FOR_HIP "-Wl,-rpath,$<TARGET_FILE_DIR:ginkgo>")
    foreach(target ${GINKGO_LINK_LIBRARIES})
        if("${target}" MATCHES "^ginkgo")
            set(GINKGO_RPATH_FOR_HIP "${GINKGO_RPATH_FOR_HIP}:$<TARGET_FILE_DIR:${target}>")
        endif()
    endforeach()

    add_subdirectory(test)
endif()
