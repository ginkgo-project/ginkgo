if (GINKGO_HIP_PLATFORM MATCHES "nvcc" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.2)
    message(FATAL_ERROR "Ginkgo HIP backend requires CUDA >= 9.2.")
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

## Setup all CMAKE variables to find HIP and its dependencies
list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")
if (GINKGO_HIP_PLATFORM STREQUAL "hcc")
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake")
endif()
list(APPEND CMAKE_PREFIX_PATH
    "${HIPBLAS_PATH}/lib/cmake"
    "${HIPRAND_PATH}/lib/cmake"
    "${HIPSPARSE_PATH}/lib/cmake"
    "${ROCRAND_PATH}/lib/cmake"
)
# Set CMAKE_MODULE_PATH and CMAKE_PREFIX_PATH as PARENT_SCOPE to easily find HIP again
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

# NOTE: without this, HIP jacobi build takes a *very* long time. The reason for
# that is that these variables are seemingly empty by default, thus there is no
# proper optimization applied to the HIP builds otherwise.
set(HIP_HIPCC_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Flags used by the HIPCC compiler during DEBUG builds")
set(HIP_HIPCC_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}" CACHE STRING "Flags used by the HIPCC compiler during MINSIZEREL builds")
set(HIP_HIPCC_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "Flags used by the HIPCC compiler during RELEASE builds")
set(HIP_HIPCC_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the HIPCC compiler during RELWITHDEBINFO builds")

find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hiprand REQUIRED)
find_package(hipsparse REQUIRED)
# At the moment, for hiprand to work also rocrand is required.
find_package(rocrand REQUIRED)
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
    components/absolute_array.hip.cpp
    components/fill_array.hip.cpp
    components/precision_conversion.hip.cpp
    components/prefix_sum.hip.cpp
    factorization/factorization_kernels.hip.cpp
    factorization/ic_kernels.hip.cpp
    factorization/ilu_kernels.hip.cpp
    factorization/par_ic_kernels.hip.cpp
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
    matrix/diagonal_kernels.hip.cpp
    matrix/ell_kernels.hip.cpp
    matrix/hybrid_kernels.hip.cpp
    matrix/sellp_kernels.hip.cpp
    matrix/sparsity_csr_kernels.hip.cpp
    multigrid/amgx_pgm_kernels.hip.cpp
    preconditioner/isai_kernels.hip.cpp
    preconditioner/jacobi_advanced_apply_kernel.hip.cpp
    preconditioner/jacobi_generate_kernel.hip.cpp
    preconditioner/jacobi_kernels.hip.cpp
    preconditioner/jacobi_simple_apply_kernel.hip.cpp
    reorder/rcm_kernels.hip.cpp
    solver/bicg_kernels.hip.cpp
    solver/bicgstab_kernels.hip.cpp
    solver/cg_kernels.hip.cpp
    solver/cgs_kernels.hip.cpp
    solver/fcg_kernels.hip.cpp
    solver/gmres_kernels.hip.cpp
    solver/cb_gmres_kernels.hip.cpp
    solver/idr_kernels.hip.cpp
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
        list(APPEND GINKGO_HIP_NVCC_ADDITIONAL_FLAGS "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    # Remove false positive CUDA warnings when calling one<T>() and zero<T>()
    list(APPEND GINKGO_HIP_NVCC_ADDITIONAL_FLAGS "--expt-relaxed-constexpr")

    if (GINKGO_HIP_PLATFROM MATCHES "nvcc" AND CMAKE_CUDA_COMPILER_VERSION
            MATCHES "9.2" AND CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*" )
        ginkgo_extract_clang_version(${CMAKE_CUDA_HOST_COMPILER} GINKGO_CUDA_HOST_CLANG_VERSION)

        if (GINKGO_CUDA_HOST_CLANG_VERSION MATCHES "5\.0.*")
            message(FATAL_ERROR "There is a bug between nvcc 9.2 and clang 5.0 which create a compiling issue."
                "Consider using a different CUDA host compiler or CUDA version.")
        endif()
    endif()
    # add gpu architecture flags
    include(CudaArchitectureSelector)
    cas_variable_cuda_architectures(GINKGO_HIP_NVCC_ARCH
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21")
endif()

# `target_compile_options` do not work with hip_add_library
# Thus, we need to pass the flags to `hip_add_library` itself
if(GINKGO_HIP_AMDGPU)
    foreach(target ${GINKGO_HIP_AMDGPU})
        list(APPEND GINKGO_AMD_ARCH_FLAGS --amdgpu-target=${target})
    endforeach()
endif()

set(GINKGO_HIPCC_OPTIONS ${GINKGO_HIP_COMPILER_FLAGS} "-std=c++14")
set(GINKGO_HIP_NVCC_OPTIONS ${GINKGO_HIP_NVCC_COMPILER_FLAGS} ${GINKGO_HIP_NVCC_ARCH} ${GINKGO_HIP_NVCC_ADDITIONAL_FLAGS})
set(GINKGO_HIP_CLANG_OPTIONS ${GINKGO_HIP_CLANG_COMPILER_FLAGS} ${GINKGO_AMD_ARCH_FLAGS})

set_source_files_properties(${GINKGO_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
hip_add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES}
    HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
    CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
    NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS}
    ${GINKGO_STATIC_OR_SHARED})

target_include_directories(ginkgo_hip
    PUBLIC
        ${HIP_INCLUDE_DIRS}
    PRIVATE
        ${GINKGO_HIP_THRUST_PATH}
        ${HIPBLAS_INCLUDE_DIRS}
        ${hiprand_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>)

target_link_libraries(ginkgo_hip PUBLIC ginkgo_device)
target_link_libraries(ginkgo_hip PRIVATE roc::hipblas roc::hipsparse hip::hiprand roc::rocrand)

target_compile_options(ginkgo_hip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)

if(GINKGO_HIP_PLATFORM MATCHES "hcc")
    find_package(hip REQUIRED)
    # To save a bit of pain, we directly link against the `library` instead of
    # linking against the target.
    get_target_property(HIP_LIBAMDHIP64_CONFIGURATIONS hip::amdhip64 IMPORTED_CONFIGURATIONS)
    if (CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" AND HIP_LIBAMDHIP64_CONFIGURATIONS MATCHES "[Dd][Ee][Bb][Uu][Gg]")
        get_target_property(HIP_LIBAMDHIP64_LIBRARIES hip::amdhip64 IMPORTED_LOCATION_DEBUG)
    else()
        get_target_property(HIP_LIBAMDHIP64_LIBRARIES hip::amdhip64 IMPORTED_LOCATION_RELEASE)
    endif()
    target_link_libraries(ginkgo_hip PUBLIC ${HIP_LIBAMDHIP64_LIBRARIES})
elseif(GINKGO_HIP_PLATFORM MATCHES "nvcc")
    find_package(CUDA 9.2 REQUIRED)
    target_link_libraries(ginkgo_hip PUBLIC ${CUDA_LIBRARIES})
endif()

# Try to find everything in /opt/rocm/lib first.
set(GKO_HIP_RPATH "${ROCM_PATH}/lib" )
if (GINKGO_HIP_PLATFORM MATCHES "nvcc")
    list(GET CUDA_LIBRARIES 0 CUDA_FIRST_LIB)
    get_filename_component(GKO_CUDA_LIBDIR "${CUDA_FIRST_LIB}" DIRECTORY)
    list(APPEND GKO_HIP_RPATH "${GKO_CUDA_LIBDIR}")
else()
    list(APPEND GKO_HIP_RPATH "${HIP_PATH}/lib")
endif()
list(APPEND GKO_HIP_RPATH "${HIPBLAS_PATH}/lib" "${HIPRAND_PATH}/lib"
    "${HIPSPARSE_PATH}/lib" "${ROCRAND_PATH}/lib")

ginkgo_compile_features(ginkgo_hip)
ginkgo_default_includes(ginkgo_hip)
ginkgo_install_library(ginkgo_hip "${GKO_HIP_RPATH}")

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_hip)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
