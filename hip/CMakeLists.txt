if(NOT DEFINED HIPBLAS_PATH)
    if(NOT DEFINED ENV{HIPBLAS_PATH})
        set(HIPBLAS_PATH "${HIP_PATH}/../hipblas" CACHE PATH "Path to which HIPBLAS has been installed")
    else()
        set(HIPBLAS_PATH $ENV{HIPBLAS_PATH} CACHE PATH "Path to which HIPBLAS has been installed")
    endif()
endif()

if(NOT DEFINED HIPSPARSE_PATH)
    if(NOT DEFINED ENV{HIPSPARSE_PATH})
        set(HIPSPARSE_PATH "${HIP_PATH}/../hipsparse" CACHE PATH "Path to which HIPSPARSE has been installed")
    else()
        set(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH} CACHE PATH "Path to which HIPSPARSE has been installed")
    endif()
endif()

# With platform = hcc, manage HCC_PATH and HCC_HOME setting
if (GINKGO_HIP_PLATFORM MATCHES "hcc")
    if(NOT DEFINED HCC_PATH)
        if(NOT DEFINED ENV{HCC_PATH})
            set(HCC_PATH "/opt/rocm/hcc" CACHE PATH "Path to which HCC has been installed")
        else()
            set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
        endif()
        set(HCC_HOME "${HCC_PATH}")
    endif()
elseif (GINKGO_HIP_PLATFORM MATCHES "nvcc") # ensure ENV{CUDA_PATH} is set by the user
    if (NOT DEFINED ENV{CUDA_PATH})
        find_path(GINKGO_HIP_DEFAULT_CUDA_PATH "cuda.h" PATH /usr/local/cuda/include NO_DEFAULT_PATH)
        if (NOT GINKGO_HIP_DEFAULT_CUDA_PATH)
            message(FATAL_ERROR "HIP nvcc backend was requested but CUDA could not be located. "
                "Set and export the environment variable CUDA_PATH.")
         endif()
     endif()
endif()

if (GINKGO_HIP_PLATFORM STREQUAL "hcc" AND NOT GINKGO_HIP_AMDGPU)
    message(WARNING "GINKGO_HIP_AMDGPU is not set. Set this flag if HIP complains about missing"
        " `--amdgpu_target` option.")
endif()


## Setup all CMAKE variables to find HIP and its dependencies
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies, e.g. hcc
    "${HIPBLAS_PATH}/lib/cmake"
    "${HIPSPARSE_PATH}/lib/cmake"
)

find_package(HIP REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)
find_path(GINKGO_HIP_THRUST_PATH "thrust/complex.h"
    PATHS "${HIP_PATH}/../include" NO_DEFAULT_PATH)
if (NOT GINKGO_HIP_THRUST_PATH)
    message(FATAL_ERROR "Could not find the ROCm header thrust/complex.h which is required by Ginkgo HIP.")
endif()


set(GINKGO_HIP_SOURCES
    base/exception.hip.cpp
    base/executor.hip.cpp
    base/version.hip.cpp
    components/zero_array.hip.cpp
    factorization/par_ilu_kernels.hip.cpp
    matrix/coo_kernels.hip.cpp
    matrix/csr_kernels.hip.cpp
    matrix/dense_kernels.hip.cpp
    matrix/ell_kernels.hip.cpp
    matrix/hybrid_kernels.hip.cpp
    matrix/sellp_kernels.hip.cpp
    matrix/sparsity_csr_kernels.hip.cpp
    preconditioner/jacobi_kernels.hip.cpp
    solver/bicgstab_kernels.hip.cpp
    solver/cg_kernels.hip.cpp
    solver/cgs_kernels.hip.cpp
    solver/fcg_kernels.hip.cpp
    solver/gmres_kernels.hip.cpp
    solver/ir_kernels.hip.cpp
    solver/lower_trs_kernels.hip.cpp
    solver/upper_trs_kernels.hip.cpp
    stop/criterion_kernels.hip.cpp
    stop/residual_norm_reduction_kernels.hip.cpp)

set(GINKGO_HIPCC_OPTIONS ${GINKGO_HIP_COMPILER_FLAGS})
set(GINKGO_HIP_NVCC_OPTIONS ${GINKGO_HIP_NVCC_COMPILER_FLAGS})
set(GINKGO_HIP_HCC_OPTIONS ${GINKGO_HIP_HCC_COMPILER_FLAGS})

set_source_files_properties(${GINKGO_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
hip_add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES} HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS} HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS} ${GINKGO_STATIC_OR_SHARED})
set_target_properties(ginkgo_hip PROPERTIES LINKER_LANGUAGE HIP)

if(GINKGO_HIP_AMDGPU AND GINKGO_HIP_PLATFORM STREQUAL "hcc")
    foreach(target ${GINKGO_HIP_AMDGPU})
        target_compile_options(ginkgo_hip PRIVATE --amdgpu-target=${target})
    endforeach()
endif()

target_compile_options(ginkgo_hip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
if(GINKGO_WITH_CLANG_TIDY AND GINKGO_CLANG_TIDY_PATH)
    set_property(TARGET ginkgo_hip PROPERTY CXX_CLANG_TIDY "${GINKGO_CLANG_TIDY_PATH};-checks=*")
endif()
if(GINKGO_WITH_IWYU AND GINKGO_IWYU_PATH)
    set_property(TARGET ginkgo_hip PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${GINKGO_IWYU_PATH})
endif()

target_link_libraries(ginkgo_hip PRIVATE hip::device roc::hipblas roc::hipsparse)

if(GINKGO_HIP_PLATFORM MATCHES "nvcc")
    find_package(CUDA 9.0 REQUIRED)
    target_link_libraries(ginkgo_hip PRIVATE ${CUDA_LIBRARIES})
endif()

target_include_directories(ginkgo_hip
    PUBLIC
        ${HIP_PATH}/include
    PRIVATE
        ${GINKGO_HIP_THRUST_PATH}
        $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>)

ginkgo_compile_features(ginkgo_hip)
ginkgo_default_includes(ginkgo_hip)
ginkgo_install_library(ginkgo_hip hip)

if(GINKGO_BUILD_TESTS)
    get_target_property(GINKGO_LINK_LIBRARIES ginkgo LINK_LIBRARIES)
    set(GINKGO_RPATH_FOR_HIP "-Wl,-rpath,$<TARGET_FILE_DIR:ginkgo>")
    foreach(target ${GINKGO_LINK_LIBRARIES})
        if("${target}" MATCHES "^ginkgo")
            set(GINKGO_RPATH_FOR_HIP "${GINKGO_RPATH_FOR_HIP}:$<TARGET_FILE_DIR:${target}>")
        endif()
    endforeach()

    add_subdirectory(test)
endif()
